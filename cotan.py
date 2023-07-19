import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VFNet(nn.Module):
    """
    A 3D U-Net to predit multiscale stationary velocity fields (SVFs).

    Args:
    - C_in: number of input channels
    - layers: number of hidden channels
    - M: number of SVFs for each resolution
    - R: number of scales
    - K: kernel size

    Inputs:
    - x: 3D brain MRI, (B,1,D1,D2,D3)

    Returns:
    - vf: multiscale SVFs, (M*R,3,D1,D2,D3)
    """
    def __init__(self, C_in=1, layers=[16,32,32,32,32], M=4, R=3, K=3):
        super(VFNet, self).__init__()

        assert R <= 3, 'number of scales should be <= 3'
        self.R = R
        self.M = M
        self.conv1 = nn.Conv3d(in_channels=C_in, out_channels=layers[0],
                               kernel_size=K, stride=1, padding=K//2)
        self.conv2 = nn.Conv3d(in_channels=layers[0], out_channels=layers[1],
                               kernel_size=K, stride=2, padding=K//2)
        self.conv3 = nn.Conv3d(in_channels=layers[1], out_channels=layers[2],
                               kernel_size=K, stride=2, padding=K//2)
        self.conv4 = nn.Conv3d(in_channels=layers[2], out_channels=layers[3],
                               kernel_size=K, stride=2, padding=K//2)
        self.conv5 = nn.Conv3d(in_channels=layers[3], out_channels=layers[4],
                               kernel_size=K, stride=1, padding=K//2)

        self.deconv4 = nn.Conv3d(in_channels=layers[4]+layers[3], out_channels=layers[3],
                                 kernel_size=K, stride=1, padding=K//2)
        self.deconv3 = nn.Conv3d(in_channels=layers[3]+layers[2], out_channels=layers[2],
                                 kernel_size=K, stride=1, padding=K//2)
        self.deconv2 = nn.Conv3d(in_channels=layers[2]+layers[1], out_channels=layers[1],
                                 kernel_size=K, stride=1, padding=K//2)
        self.deconv1 = nn.Conv3d(in_channels=layers[1]+layers[0], out_channels=layers[0],
                                 kernel_size=K, stride=1, padding=K//2)

        self.flow1 = nn.Conv3d(in_channels=layers[2], out_channels=3*M,
                               kernel_size=K, stride=1, padding=K//2)
        self.flow2 = nn.Conv3d(in_channels=layers[1], out_channels=3*M,
                               kernel_size=K, stride=1, padding=K//2)
        self.flow3 = nn.Conv3d(in_channels=layers[0], out_channels=3*M,
                               kernel_size=K, stride=1, padding=K//2)
        
        nn.init.normal_(self.flow1.weight, 0, 1e-5)
        nn.init.constant_(self.flow1.bias, 0.0)
        nn.init.normal_(self.flow2.weight, 0, 1e-5)
        nn.init.constant_(self.flow2.bias, 0.0)
        nn.init.normal_(self.flow3.weight, 0, 1e-5)
        nn.init.constant_(self.flow3.bias, 0.0)
        
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
    def forward(self, x):
        
        x1 = F.leaky_relu(self.conv1(x), 0.2)
        x2 = F.leaky_relu(self.conv2(x1), 0.2)
        x3 = F.leaky_relu(self.conv3(x2), 0.2)
        x4 = F.leaky_relu(self.conv4(x3), 0.2)
        x  = F.leaky_relu(self.conv5(x4), 0.2)
                
        x = torch.cat([x, x4], dim=1)
        x = F.leaky_relu(self.deconv4(x), 0.2)

        x = self.up(x)
        x = torch.cat([x, x3], dim=1)
        x = F.leaky_relu(self.deconv3(x), 0.2)
        vf1 = self.up(self.up(self.flow1(x)))
        # reshape to (M,3,D1,D2,D3)
        vf1 = vf1.reshape(self.M,3,*vf1.shape[2:])
        
        x = self.up(x)
        x = torch.cat([x, x2], dim=1)
        x = F.leaky_relu(self.deconv2(x), 0.2)
        vf2 = self.up(self.flow2(x))
        vf2 = vf2.reshape(self.M,3,*vf2.shape[2:])

        x = self.up(x)
        x = torch.cat([x, x1], dim=1)
        x = F.leaky_relu(self.deconv1(x), 0.2)
        vf3 = self.flow3(x)
        vf3 = vf3.reshape(self.M,3,*vf3.shape[2:])

        if self.R == 3:
            vf = torch.cat([vf1, vf2, vf3], dim=0)
        elif self.R == 2:
            vf = torch.cat([vf2, vf3], dim=0)
        elif self.R == 1:
            vf = torch.cat([vf3], dim=0)
        return vf  # velocity field (M*R,3,D1,D2,D3)

    
class AttentionNet(nn.Module):
    """
    Channel-wise Attention Network.

    Args:
    - C_in: number of input channels
    - C: number of hidden channels
    - M: number of SVFs for each resolution
    - R: number of scales

    Inputs:
    - T: time sequence, (N,1)
    e.g., [0, 0.1, 0.2, ..., 1.0]
    - age: age (week) of the neonates, (N,1)
    e.g., [32] * N
    
    Returns:
    - conditional time-varying attention maps (N, M*R)
    """
    def __init__(self, C_in=2, C=16, M=4, R=3):
        super(AttentionNet, self).__init__()
        self.fc1 = nn.Linear(C_in,C*4)
        self.fc2 = nn.Linear(C*4,C*8)
        self.fc3 = nn.Linear(C*8,C*8)
        self.fc4 = nn.Linear(C*8,C*4)
        self.fc5 = nn.Linear(C*4,M*R)
        
    def forward(self, T, age):
        x = torch.cat([T, age], dim=1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.leaky_relu(self.fc4(x), 0.2)
        x = self.fc5(x)
        return F.softmax(x, dim=-1)  # (N, M*R)
    
    
class CoTAN(nn.Module):
    """
    Conditional Temporal Attention Network (CoTAN).

    Args:
    - layers: number of hidden channels for VFNet
    - C: number of hidden channels for AttentionNet
    - M: number of SVFs for each resolution
    - R: number of scales

    Inputs:
    - x: vertices of input mesh (1,|V|,3)
    - T: time sequence, (N,1)
    e.g., [0, 0.1, 0.2, ..., 1.0]
    - age: age (week) of the neonates, (N,1)
    e.g., [32] * N
    - vol: 3D brain MRI, (1,1,D1,D2,D3)
    
    Returns:
    - x: vertices of deformed mesh (1,|V|,3)
    """
    def __init__(self, layers=[16,32,32,32,32], C=16, M=4, R=3):
        super(CoTAN, self).__init__()
        self.vf_net = VFNet(C_in=1, layers=layers, M=M, R=R)
        self.att_net = AttentionNet(C_in=2, C=C, M=M, R=R)
        self.RM = R * M
        
    def forward(self, x, T, age, vol):
        # ------ conditional temporal attention ------ 
        N = T.shape[0]  # total time steps
        h = 1. / N  # step size
        # learn attention maps to weight SVFs
        weight = self.att_net(T, age).unsqueeze(-1).unsqueeze(-1)  # (N,M*R,1,1)

        # ------ multiscale velocity fields ------
        vel_field = self.vf_net(vol)  # (M*R,3,D1,D2,D3)
        # vel_field = torch.tanh(vel_field)  # clip the large deformation if needed
        D1,D2,D3 = vol[0,0].shape
        D = max([D1,D2,D3])
        # rescale the vert to [-1, 1]
        self.rescale = torch.Tensor([D1/D, D2/D, D3/D]).to(vol.device)
        
        # ------ integration ------
        for n in range(N):
            # sample velocity for each vertex
            v = self.interpolate(x, vel_field)  # (M*R,|V|,3)
            # CTVF: weighted by attention for different integration step
            v = (weight[n] * v).sum(0, keepdim=True)
            x = x + h*v  # deformation
        return x

    def interpolate(self, x, vel_field):
        grid = x / self.rescale
        grid = grid.repeat(self.RM,1,1).unsqueeze(dim=-2).unsqueeze(dim=-2).flip(-1)
        v = F.grid_sample(vel_field, grid, mode='bilinear',
                          padding_mode='border', align_corners=True)
        return v[...,0,0].permute(0,2,1)  # sampled velocity
