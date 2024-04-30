## referenced by LightViT: Towards Light-Weight Convolution-Free Vision Transformers
## https://github.com/hunto/image_classification_sota/blob/36539b63cc8b851bd3fc93251bba60528813bb36/lib/models/lightvit.py#L264
import time

import torch
import torch.nn as nn
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
import torch.nn.functional as F
from model import common
from torch.autograd import Variable
from einops import rearrange
from model.refinetor.util import clones, Mlp, _get_activation_fn, PatchEmbed, PatchUnEmbed,  window_reverse, get_relative_position_index, LocalityFeedForward
from model.attention import LightViTAttention as Attention

class TSAB(nn.Module):
    def __init__(self, dim, num_heads, m_attn, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, query_num=256,
                 act_layer=nn.GELU, shift=0, ):
        super().__init__()
        self.dim = dim

        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.query_num = query_num
        self.attn = m_attn(dim=dim, query_num=query_num, num_heads=num_heads, window_size=window_size,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale, shift=shift)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
 
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x, global_mask, local_mask):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.attn(self.norm1(x), (H, W), global_mask, local_mask)
        x = x + self.mlp(self.norm2(x)) # mlp -> conv
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x

    def flops(self, x_size, local_mask, global_mask):
        flops = 0
        H, W = x_size
        ws = self.window_size
        # norm1
        flops += self.dim * H * W
        flops += self.attn.flops(x_size, local_mask, global_mask)
        # # lightVit
        # ## 1. local
        # hg, wg = H // ws, W // ws
        # flops += (H * W * self.dim * self.dim * 3) # qkv
        #
        # local_mask = rearrange(local_mask, 'b c (h hs) (w ws) -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws).squeeze(-1)
        # local_mask = local_mask.max(dim=-1).values
        #
        # flops += (local_mask.sum() * ws * ws * ws * ws * self.dim) * 2
        #
        # ## 2. global
        # # W-MSA/SW-MSA
        # flops += self.dim * self.query_num # norm
        # flops += (H * W * self.dim * self.dim * 3) # qkv
        #
        # flops += (global_mask.sum() * self.query_num * self.dim // self.num_heads * self.num_heads) * 2
        # flops += (H * W * self.dim * self.dim * 2)
        # flops += (global_mask.sum() * self.query_num * self.dim // self.num_heads * self.num_heads) * 2

        # mlp
        flops += 2 *  H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class PSPL(nn.Module):
    def __init__(self, dim=64, n_feats=64, num_heads=4, m_attn=None, window_size=8, depths=6, mlp_ratio=2., query_num=256, att_patch_size=1,
                 qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, t_eps=0.01, conv=common.default_conv):
        super().__init__()
        self.dim = dim
        self.depth = depths
        self.window_size = window_size
        self.t_eps = t_eps
        self.sig = torch.sigmoid
        self.n_feats = n_feats
        if not isinstance(query_num, list):
            query_num = [query_num] * depths
        # query_num = [64, 64, 32, 16, 8, 8]
        self.embed = nn.Conv2d(n_feats, dim, kernel_size=att_patch_size, stride=att_patch_size)
        self.blocks = nn.ModuleList([
                             TSAB(dim=dim,
                                  num_heads=num_heads,
                                  m_attn=m_attn,
                                  window_size=window_size,
                                  mlp_ratio=mlp_ratio,
                                  query_num=query_num[i],
                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  norm_layer=norm_layer
                                  ) for i in range(depths)])

        self.var = nn.Sequential(*[nn.Conv2d(dim, dim // 2, 3, 1, 1), nn.Tanh(),
                                   nn.Conv2d(dim // 2, 1, 3, 1, 1)])
        self.unembed = nn.Conv2d(dim, n_feats, kernel_size=att_patch_size, stride=att_patch_size)
        # self.norm = nn.LayerNorm(n_feats)
    def forward_feature(self, x):
        bs, c, h, w = x.shape
        ws = self.window_size

        cum_score = Variable(torch.zeros((bs, 1, h, w), device=x.device))
        global_mask = Variable(torch.ones((bs, 1, h, w), device=x.device)) # ()
        local_mask = Variable(torch.ones((bs * (h // ws) * (w // ws), 1), device=x.device))

        halting_mask = []
        global_mask_list = []
        local_mask_list = []

        out = self.embed(x)

        for blk in self.blocks:
            block_output = blk(out, global_mask.float(), local_mask.float()) 
            out = block_output.clone()  # Deep copy needed for the next layer
            #1.
            # var = self.var(block_output.detach())
            # halting_mask.append(var)

            #2.
            # var = self.var(block_output)
            # halting_mask.append(var)

            #3.
            var = self.var(block_output)
            if self.training:
                halting_mask.append(self.var(block_output.detach()))
            else:
                halting_mask.append(var)
            cum_score = cum_score + self.sig(var)  
            # Update the mask
            global_mask = (cum_score < 1 - self.t_eps).float()
            local_mask = cum_score.reshape(bs, h // ws, ws, w // ws, ws).permute(0, 1, 3, 2, 4)
            local_mask = local_mask.reshape(-1, self.window_size * self.window_size)
            local_mask = torch.mean(local_mask, dim=-1) # B,
            local_mask = (local_mask < 1 - self.t_eps).float()

            global_mask_list.append(global_mask)
            local_mask_list.append(window_reverse(local_mask.reshape(-1, 1, 1).expand(-1, ws * ws, -1), ws, h, w).permute(0, 3, 1, 2))

        # for blk in self.blocks:
        #     var = self.var(out.detach())
        #     halting_mask.append(var)
        #
        #     cum_score = cum_score + self.sig(var)  
        #     # Update the mask
        #     global_mask = (cum_score < 1 - self.eps).float()
        #     local_mask = cum_score.reshape(bs, h // ws, ws, w // ws, ws).permute(0, 1, 3, 2, 4)
        #     local_mask = local_mask.reshape(-1, self.window_size * self.window_size)
        #     local_mask = torch.mean(local_mask, dim=-1)  # B,
        #     local_mask = (local_mask < 1 - self.eps).float()
        #
        #     global_mask_list.append(global_mask)
        #     local_mask_list.append(
        #         window_reverse(local_mask.reshape(-1, 1, 1).expand(-1, ws * ws, -1), ws, h, w).permute(0, 3, 1, 2))
        #
        #     block_output = blk(out, global_mask.float(), local_mask.float()) 
        #     out = block_output.clone()  # Deep copy needed for the next layer

        return out, halting_mask, global_mask_list, local_mask_list

    def forward(self, x):
        out, halting_mask, global_mask_list, local_mask_list = self.forward_feature(x)
        h, w = x.shape[2:]
        out = self.unembed(out)
        # out = rearrange(self.norm(rearrange(out, 'b c h w -> b (h w) c')), 'b (h w) c -> b c h w', h = h, w = w)
        # print("?")
        return x + out, halting_mask, global_mask_list, local_mask_list


    def flops(self, x_size, dict):
        h, w = x_size
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        H = h + mod_pad_h
        W = w + mod_pad_w

        x_size = (H, W)
        flops = 0
        flops += H * W * self.n_feats * self.dim * 1 * 2
        global_mask_list = dict['global_mask']
        local_mask_list = dict['local_mask']

        for i, blk in enumerate(self.blocks):
            flops += blk.flops(x_size, local_mask_list[i], global_mask_list[i])

        return flops


