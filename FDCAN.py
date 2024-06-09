import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from einops import rearrange
import warnings
warnings.filterwarnings('ignore')

class Embedding(nn.Module):
    def __init__(self, P=8, S=4, D=2):
        super(Embedding, self).__init__()
        self.P = P
        self.S = S
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=D,
            kernel_size=P,
            stride=S
        )

    def forward(self, x):
        B = x.shape[0]
        x = x.unsqueeze(2)
        x = rearrange(x, 'b m r l -> (b m) r l')
        x_pad = F.pad(
            x,
            pad=(0, self.P - self.S),
            mode='replicate'
        )
        x_emb = self.conv(x_pad)
        x_emb = rearrange(x_emb, '(b m) d n -> b m d n', b=B)
        return x_emb

class gc_module(nn.Module):
    def __init__(self, M, D, r, one=True):
        super(gc_module, self).__init__()
        groups_num = M if one else D
        self.pw_con1 = nn.Conv1d(
            in_channels=M * D,
            out_channels=r * M * D,
            kernel_size=1,
            groups=groups_num
        )
        self.pw_con2 = nn.Conv1d(
            in_channels=r * M * D,
            out_channels=M * D,
            kernel_size=1,
            groups=groups_num
        )

        self.bn = nn.BatchNorm1d(M * D)

    def forward(self, x):
        x = self.pw_con2(F.gelu(self.bn(self.pw_con1(x))))
        return x

class GC_module(nn.Module):
    def __init__(self, M, D, kernel_size, r):
        super(GC_module, self).__init__()
        self.dw_conv = nn.Conv1d(
            in_channels=M * D,
            out_channels=M * D,
            kernel_size=kernel_size,
            groups=M * D,
            padding='same'
        )
        self.conv_ffn1 = gc_module(M, D, r, one=True)
        self.conv_ffn2 = gc_module(M, D, r, one=False)

    def forward(self, x_emb):
        D = x_emb.shape[-2]
        x = rearrange(x_emb, 'b m d n -> b (m d) n')
        x = self.conv_ffn1(x)

        x = rearrange(x, 'b (m d) n -> b m d n', d=D)
        x = x.permute(0, 2, 1, 3)
        x = rearrange(x, 'b d m n -> b (d m) n')
        x = self.conv_ffn2(x)
        x = rearrange(x, 'b (d m) n -> b d m n', d=D)
        x = x.permute(0, 2, 1, 3)
        out = x + x_emb
        return out

class DC_module(nn.Module):
    def __init__(self, M, L, T, D=2, P=8, S=4, kernel_size=51, r=1, num_layers=2):
        super(DC_module, self).__init__()
        self.num_layers = num_layers
        N = L // S
        self.embed_layer = Embedding(P, S, D)
        self.backbone = nn.ModuleList([GC_module(M, D, kernel_size, r) for _ in range(num_layers)])
        self.head = nn.Linear(D * N, T)

    def forward(self, x):
        x_emb = self.embed_layer(x)

        for i in range(self.num_layers):
            x_emb = self.backbone[i](x_emb)

        z = rearrange(x_emb, 'b m d n -> b m (d n)')
        pred = self.head(z)

        return pred

# ==============Predictor=====================
class Predictor(nn.Module):
    def __init__(self, args,sta_h=64,T = 20):
        super(Predictor, self).__init__()
        self.T = T
        self.sta_dense = nn.Sequential(
            nn.Linear(args.static_dim,sta_h),
            nn.ReLU(),
        )
        self.args = args

        self.dc_module = DC_module(M=args.dynamic_dim, L=args.N_time,
                              T=self.T, D=2,num_layers=1)

        self.attention = nn.Sequential(
            nn.Linear(args.dynamic_dim+sta_h,args.dynamic_dim+sta_h),
            nn.ReLU()
        )

        self.all_dense0 = nn.Sequential(
            nn.Linear(args.dynamic_dim+sta_h, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.all_dense1 = nn.Sequential(
            nn.Linear(args.dynamic_dim+sta_h, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.all_dense2 = nn.Sequential(
            nn.Linear(args.dynamic_dim+sta_h, 32),  
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x_sta,x_dyn):
        b,t,q  = x_dyn.shape
        y1 = self.sta_dense(x_sta)
        y2 = self.dc_module(x_dyn.transpose(2,1))
        y3 = torch.concat((y1,torch.mean(y2, 2)),dim=1)
        score = self.attention(y3)
        y3 = torch.mul(y3,score)
        out = torch.concat(
            (self.all_dense0(y3),
             self.all_dense1(y3),
             self.all_dense2(y3)),dim=1
        )

        return out
