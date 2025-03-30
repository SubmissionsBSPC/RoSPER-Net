import torch

import torch.nn as nn


from models.Attention import EdgeEnhancement

class MSFEBlock(nn.Module):
    def __init__(self,in_channels,kernel,sample1=None,sample2=None):
        super().__init__()
        self.sample1=sample1
        self.sample2=sample2
        self.extract=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel,padding=kernel//2,groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self,x):
        if self.sample1!=None:
            x=self.sample1(x)
        x=self.extract(x)
        if self.sample2!=None:
            x=self.sample2(x)
        return x
    
    
class DCT(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.attn=EdgeEnhancement(in_channels)

    def forward(self,x):
        x=self.attn(x)
        return x

class MSFE(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.c=in_channels

        self.msfe1=MSFEBlock(in_channels,3)
        self.msfe2=MSFEBlock(in_channels,7)
        self.msfe3=MSFEBlock(in_channels,11)
        self.msfe4=MSFEBlock(in_channels,15)
        self.msfe5=MSFEBlock(in_channels,3,nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),nn.MaxPool2d(kernel_size=2,stride=2))
        self.msfe6=MSFEBlock(in_channels,11,nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),nn.MaxPool2d(kernel_size=2,stride=2))
        self.msfe7=MSFEBlock(in_channels,3,nn.MaxPool2d(kernel_size=2,stride=2),nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.msfe8=MSFEBlock(in_channels,11,nn.MaxPool2d(kernel_size=2,stride=2),nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        
        self.extract=nn.Sequential(
            nn.Conv2d(8*in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels)
        )
        
        
    def forward(self,x):
        x1=self.msfe1(x)
        x2=self.msfe2(x)
        x3=self.msfe3(x)
        x4=self.msfe4(x)
        x5=self.msfe5(x)
        x6=self.msfe6(x)
        x7=self.msfe7(x)
        x8=self.msfe8(x)
        out=torch.cat([x1,x2,x3,x4,x5,x6,x7,x8],dim=1)
        out=self.extract(out)
        return out
    

class CSED(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.msfe=MSFE(in_channels)
        self.dct=DCT(in_channels)

    def forward(self,x):
        x=self.msfe(x)
        x=self.dct(x)
        return x

    

class Connect(nn.Module):
    def __init__(self,in_channels_e,in_channels_d):
        super().__init__()
        self.pro=nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels_d,in_channels_e,1),
            nn.BatchNorm2d(in_channels_e)
        )
    
    def forward(self,xe,xd):
        xd=self.pro(xd)
        return xd+xe


class DecoderBlock(nn.Module):
    def __init__(self,in_channel_e,in_channel_d=None):
        super().__init__()
        if in_channel_d!=None:
            self.csed=CSED(in_channel_e)
            self.con=Connect(in_channel_e,in_channel_d)
        else:
            self.csed=CSED(in_channel_e)
            self.caf=None
        
    def forward(self,x_e,x_d=None):
        if x_d==None:
            x=self.csed(x_e)
        else:
            x_caf=self.con(x_e,x_d)
            x=self.csed(x_caf)
        return x
    


class Decoder(nn.Module):
    def __init__(self,in_channels=[64,128,256,512,512]):
        super().__init__()
        self.db4=DecoderBlock(in_channels[3])
        self.db3=DecoderBlock(in_channels[2],in_channels[3])
        self.db2=DecoderBlock(in_channels[1],in_channels[2])
        self.db1=DecoderBlock(in_channels[0],in_channels[1])
        
    def forward(self,x):
        x1,x2,x3,x4=x
        x_list=[]
        x4=self.db4(x4)
        x_list.append(x4)
        x3=self.db3(x3,x4)
        x_list.append(x3)
        x2=self.db2(x2,x3)
        x_list.append(x2)
        x1=self.db1(x1,x2)
        x_list.append(x1)
        return x_list



class PH_Block(nn.Module):
    def __init__(self,in_channels,scale_factor=1):
        super().__init__()
        if scale_factor>1:
            self.upsample=nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        else:
            self.upsample=None
        self.pro=nn.Conv2d(in_channels,1,1)

    def forward(self,x:torch.Tensor):
        if self.upsample!=None:
            x=self.upsample(x)
        x=self.pro(x)
        return x

class PH(nn.Module):
    def __init__(self,in_channels=[64,128,256,512,512],scale_factor=[1,2,4,8,16]):
        super().__init__()
        self.final=nn.ModuleList()
        self.ph1=PH_Block(in_channels[0],scale_factor[0])
        self.ph2=PH_Block(in_channels[1],scale_factor[1])
        self.ph3=PH_Block(in_channels[2],scale_factor[2])
        self.ph4=PH_Block(in_channels[3],scale_factor[3])

    def forward(self,x):
        x4,x3,x2,x1=x
        x_list=[]
        x1=self.ph1(x1)
        x_list.append(x1.sigmoid())
        x2=self.ph2(x2)
        x_list.append(x2.sigmoid())
        x3=self.ph3(x3)
        x_list.append(x3.sigmoid())
        x4=self.ph4(x4)
        x_list.append(x4.sigmoid())
        return x_list


