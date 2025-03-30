import torch
import torch.nn as nn

class EdgeEnhancement(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.ns=NSB(in_channels)
        self.dcb=DADCB(in_channels)
        self.balance=nn.Sequential(
            nn.Conv2d(in_channels*2,in_channels,1),
            nn.BatchNorm2d(in_channels)
        )
    def forward(self,x):
        x_dcb=self.dcb(x)
        x_ns=self.ns(x)
        x_out=torch.cat([x_dcb,x_ns],dim=1)
        x_out=self.balance(x_out)
        return x_out

class NSB(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.context=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=15,padding=7,groups=in_channels),
            nn.BatchNorm2d(in_channels)
        )
        self.AP=nn.AvgPool2d(3,1,1)

        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        x_con=self.context(x)
        x_ap=self.AP(x_con)
        weight=self.conv(x_con-x_ap)#B,C,H,W
        x_ee=weight*x
        return x_ee

import torchvision

class DirectionOffsets(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        
        self.offset1=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,(1,15),1,(0,7),groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        self.offset2=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,(15,1),padding=(7,0),groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        self.offset3=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,padding=1,groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        self.balance=nn.Sequential(
            nn.Conv2d(in_channels,2 * kernel_size * kernel_size,1),
            nn.BatchNorm2d(2 * kernel_size * kernel_size)
        )
        
    def forward(self, x):
        offsets1=self.offset1(x)*x
        offsets2=self.offset2(x)*x
        offsets3=self.offset3(x)*x
        offsets=offsets1+offsets2+offsets3
        offsets=self.balance(offsets)
        return offsets

class DADCB(nn.Module):
    def __init__(self, in_channels, kernel_size=3,padding=1,dilation=1):
        super().__init__()
        self.offset=DirectionOffsets(in_channels)
        self.deform=torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=in_channels,
                                                        dilation=dilation,
                                                        bias=False)
        self.balance=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        offsets = self.offset(x)
        out = self.deform(x, offsets)
        out = self.balance(out)*x
        return out
