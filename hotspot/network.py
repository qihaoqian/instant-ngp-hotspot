import torch
import torch.nn as nn
import torch.nn.functional as F

from hotspot.encoding import get_encoder


class SDFNetwork(nn.Module):
    def __init__(self,
                 encoding="hashgrid",
                 num_layers=3,
                 skips=[],
                 hidden_dim=64,
                 clip_sdf=None,
                 num_levels = 8,
                 base_resolution = 16,
                 desired_resolution = 2048,
                 sphere_radius=1.6,
                 sphere_scale=1.0,
                 use_sphere_post_processing=False,
                 ):
        super().__init__()


        self.num_layers = num_layers
        self.skips = skips
        self.hidden_dim = hidden_dim
        self.clip_sdf = clip_sdf
        self.sphere_radius = sphere_radius
        self.sphere_scale = sphere_scale
        self.use_sphere_post_processing = use_sphere_post_processing

        self.encoder, self.in_dim = get_encoder(encoding, 
                                                num_levels=num_levels, 
                                                base_resolution=base_resolution,
                                                desired_resolution=desired_resolution)

        backbone = []

        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            elif l in self.skips:
                in_dim = self.hidden_dim + self.in_dim
            else:
                in_dim = self.hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1
            else:
                out_dim = self.hidden_dim
            use_bias = (l == num_layers - 1 and self.use_sphere_post_processing)
            backbone.append(nn.Linear(in_dim, out_dim, bias=use_bias))

        self.backbone = nn.ModuleList(backbone)

        if self.use_sphere_post_processing:
            self._init_weights()

    def _init_weights(self):
        # 对所有隐藏层使用小幅随机初始化
        for l, layer in enumerate(self.backbone):
            if l < self.num_layers - 1:
                # 隐藏层：Xavier 正态或小幅度均匀
                nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
            else:
                # 输出层：weight 清零，bias 填 sphere_radius
                nn.init.zeros_(layer.weight)
                nn.init.constant_(layer.bias, self.sphere_radius ** 2)
            
    def forward(self, x):
        # x: [B, 3]

        x = self.encoder(x)

        h = x
        for l in range(self.num_layers):
            if l in self.skips:
                h = torch.cat([h, x], dim=-1)
            h = self.backbone[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        if self.clip_sdf is not None:
            h = h.clamp(-self.clip_sdf, self.clip_sdf)
            
        # 球面后处理：符号+开方 -> 减半径 -> 缩放
        if self.use_sphere_post_processing:
            eps = 1e-8
            h_sqrt     = torch.sign(h) * torch.sqrt(h.abs() + eps)
            h_centered = h_sqrt - self.sphere_radius
            h = h_centered * self.sphere_scale

        return h