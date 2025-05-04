import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Patcher(nn.Module):
    def __init__(self,channel, patch=16,n_embd=128):
        super().__init__()
        self.patch=patch
        self.n_embd=n_embd
        
        self.proj = nn.Conv2d(channel, n_embd, kernel_size=patch, stride=patch)
    def forward(self,x):
        x=self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    

class dePatcher(nn.Module):
    
    def __init__(self,in_channels=3,patch_size=16,emb_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.unproj = nn.ConvTranspose2d(emb_dim, in_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x,image_size):
        B, N, D = x.shape
        H = W = image_size
        x = x.transpose(1, 2).view(B, D, H // self.patch_size, W // self.patch_size)  
        x = self.unproj(x) 
        return x
