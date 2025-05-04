import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention
import yaml
from patch_embd import Patcher,dePatcher

with open("config.yaml", "r") as file:
    data = yaml.safe_load(file)

config = data["config"]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList()
        
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    nn.BatchNorm2d(channels),
                    nn.SiLU(),
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    nn.BatchNorm2d(channels//2),
                    nn.SiLU(),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + self.use_residual * x

        return x

class Attention4VAE(nn.Module):
    def __init__(self, channels,n_head=1,patch=16,request=False):
        super().__init__()
        self.b_norm=nn.BatchNorm2d(channels)
        self.attention=SelfAttention(128,n_head,request)
        self.depatcher=dePatcher(in_channels=channels,patch_size=patch)
        self.patcher=Patcher(channels,patch)
    def forward(self,x):
        B,C,H,W=x.shape
        residue = x 
        x=self.b_norm(x)
        x=self.patcher(x)
        x = self.attention(x)
        x=self.depatcher(x,H)
        x += residue
        
        return x 
    
class Attention4VAE2(nn.Module):
    def __init__(self, channels,n_head=1,patch=16,request=False):
        super().__init__()
        self.b_norm=nn.BatchNorm2d(channels)
        self.attention=SelfAttention(channels,n_head,request)
        
    def forward(self,x):
        residue=x
        B,C,H,W=x.shape
        x=x.view((B,C,H*W))
        x=x.transpose(-1,-2)
        x=self.attention(x)
        x=x.transpose(-1,-2)
        x=x.view((B,C,H,W))
        x+=residue
        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes, anchors_per_scale=3):
        super(ScalePrediction, self).__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2*in_channels, kernel_size=3, padding=1),
            CNNBlock(2*in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1),
        )
        self.num_classes = num_classes
        self.anchors_per_scale = anchors_per_scale

    def forward(self, x):
        return (
            self.pred(x)
                .reshape(x.shape[0], self.anchors_per_scale, self.num_classes + 5, x.shape[2], x.shape[3])
                .permute(0, 1, 3, 4, 2)
        )


class PPLO(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.ConvTranspose2d):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:


            if isinstance(module, list):
                if(module[0]=="B"):
                    num_repeats = module[1]

                    layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))
                elif(module[0]=="C"):
                    out_channels, kernel_size, stride = module[1],module[2],module[3]
                    layers.append(
                        CNNBlock(
                            in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=1 if kernel_size == 3 else 0,
                        )
                    )
                    in_channels = out_channels
                elif(module[0]=="A"):
                    layers.append(Attention4VAE2(in_channels))
            

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        Attention4VAE2(in_channels),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.ConvTranspose2d(in_channels,in_channels,2,2),)
                    in_channels = in_channels * 3
        return layers
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 20
    IMAGE_SIZE = 224
    model = PPLO(num_classes=num_classes).to(device)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE)).to(device)
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    print("Success!")