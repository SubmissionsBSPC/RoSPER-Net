import torch
from torch import nn
from mamba_ssm import Mamba

class PMM(nn.Module):
    
    def __init__(self, in_channels,pool_scale=4):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.mamba=Mamba(
                d_model=in_channels//4, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
        )
        self.bn=nn.BatchNorm2d(in_channels)
        self.pro1=nn.Conv2d(in_channels, in_channels,1)
        self.dwconv=nn.Conv2d(in_channels,in_channels,7,padding=3,groups=in_channels)
        self.gelu=nn.GELU()
        self.pro2=nn.Conv2d(in_channels, in_channels,1)
        

    def forward(self,x):
        B,C,H,W=x.shape
        x=x.view(B,C,H*W).permute(0,2,1).contiguous()
        x_norm=self.norm(x)
        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + x1
        x_mamba2 = self.mamba(x2) + x2
        x_mamba3 = self.mamba(x3) + x3
        x_mamba4 = self.mamba(x4) + x4
        x_mamba = torch.cat([x_mamba1, x_mamba2,x_mamba3,x_mamba4], dim=2)
        x_mamba=x_mamba.permute(0,2,1).contiguous().view(B,C,H,W)
        x_mamba=self.bn(x_mamba)
        x_ffn=self.pro1(x_mamba)

        x_ffn=self.dwconv(x_ffn)
        x_ffn=self.gelu(x_ffn)
        x_ffn=self.pro2(x_ffn)
        return x_mamba,x_ffn

class CAB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.score1=nn.AdaptiveMaxPool2d(1)
        self.score2=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        score=self.score1(x) + self.score2(x)
        score=self.conv(score)*x
        return score

class SAB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ave=nn.Sequential(
            nn.AvgPool2d(3,1,1),
            nn.Conv2d(in_channels,in_channels//2,1),
            nn.BatchNorm2d(in_channels//2),
            nn.GELU(),
            nn.Conv2d(in_channels//2,in_channels//2,1),
            nn.BatchNorm2d(in_channels//2),
        )
        self.max=nn.Sequential(
            nn.MaxPool2d(3,1,1),
            nn.Conv2d(in_channels,in_channels//2,1),
            nn.BatchNorm2d(in_channels//2),
            nn.GELU(),
            nn.Conv2d(in_channels//2,in_channels//2,1),
            nn.BatchNorm2d(in_channels//2),
        )
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,1,7,padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        x_ave=self.ave(x)
        x_max=self.max(x)
        x_=torch.cat([x_ave,x_max],dim=1)
        x_=self.conv(x_)*x
        return x_

class SPE(nn.Module):
    
    def __init__(self, input_channels=3, out_channels=3,pool=True):
        super().__init__()
        if pool:
            self.pool=nn.MaxPool2d(2, stride=2)
        else:
            self.pool=None
        self.conv=nn.Sequential(
            nn.Conv2d(input_channels,out_channels,3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.pmb=PMM(in_channels=out_channels)
        self.prompt1=SPB(out_channels)
        self.prompt2=SPB(out_channels)

    
    def forward(self, x):
        if self.pool!=None:
            x=self.pool(x)
        x=self.conv(x)
        p1=self.prompt1(x)
        x_mamba,x_ffn=self.pmb(x)
        p2=self.prompt2(x_mamba)
        x=x_ffn+p1+p2
        return x

class SPM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca=CAB(in_channels)
        self.sa=SAB(in_channels)
        self.prompt=nn.Sequential(
            nn.Conv2d(in_channels,1,1),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        x_ca=self.ca(x)
        x_sa=self.sa(x_ca)
        p=self.prompt(x_sa)
        return p     

class Encoder(nn.Module):
    
    def __init__(self, input_channels=3, out_channels=[8,16,24,32,40,48]):
        super().__init__()
        self.block1=SPE(input_channels,out_channels[0],pool=False)
        self.block2=SPE(out_channels[0],out_channels[1])
        self.block3=SPE(out_channels[1],out_channels[2])
        self.block4=SPE(out_channels[2],out_channels[3])

    def forward(self, x):
        x_list=[]
        x=self.block1(x)
        x_list.append(x)
        x=self.block2(x)
        x_list.append(x)
        x=self.block3(x)
        x_list.append(x)
        x=self.block4(x)
        x_list.append(x)
        return x_list