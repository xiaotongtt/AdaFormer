import math

import torch
import torch.nn as nn
from torch.nn import Module
from timm.models.layers import  trunc_normal_
from model.refinetor.util import window_reverse, get_relative_position_index
# from kmeans_pytorch import kmeans
from einops import rearrange
# import ACT
# from ACT.extensions import ada_cluster, broadcast, weighted_sum

# class LightViTAttention(nn.Module):
#     def __init__(self, dim, query_class=1, query_num=256, num_heads=8, window_size=8,
#                  qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.query_num = query_num
#         self.query_class = query_class
#         self.window_size = window_size
#         self.attn_area = window_size * window_size
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.qkv_local = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.proj = nn.Linear(dim, dim)
#         self.querynorm = nn.Sequential(
#             nn.BatchNorm2d(dim)
#         )
#         self.norm1 = norm_layer(dim)
#         self.norm2 = norm_layer(dim)
#         self.norm3 = norm_layer(dim)
#         self.prototype = nn.Parameter(torch.randn((self.query_class, query_num, dim)))
#
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
#
#         # Get pair-wise relative position index for each token inside the window
#         self.register_buffer("relative_position_index", get_relative_position_index(window_size,
#                                                                                     window_size).view(-1))
#         # Init relative positional bias
#         trunc_normal_(self.relative_position_bias_table, std=.02)
#         trunc_normal_(self.prototype, std=.02)
#
#     def _get_relative_positional_bias(
#             self
#     ) -> torch.Tensor:
#         """ Returns the relative positional bias.
#         Returns:
#             relative_position_bias (torch.Tensor): Relative positional bias.
#         """
#         relative_position_bias = self.relative_position_bias_table[
#             self.relative_position_index].view(self.attn_area, self.attn_area, -1)
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
#         return relative_position_bias.unsqueeze(0)
#
#     def softmax_with_policy(self, attn, policy, eps=1e-6):
#         B, H, N, HW = attn.size()
#         attn_policy = policy.reshape(B, 1, 1, HW)
#
#         max_att = torch.max(attn, dim=-1, keepdim=True)[0]
#         attn = attn - max_att
#
#         # for stable training
#         attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
#         attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
#         return attn.type_as(max_att)
#
#     def forward_global_aggregation(self, q, k, v, attn_mask=None):
#         """
#         q: global tokens
#         k: image tokens
#         v: image tokens
#         """
#         B, _, N, _ = q.shape
#         _, h, HW, c = k.shape
#         q = q * self.scale
#         if attn_mask is not None:
#             attn = (q @ k.transpose(-2, -1)) * attn_mask.reshape(B, 1, 1, attn_mask.shape[1]).expand(-1, h, N, -1)
#         else:
#             attn = (q @ k.transpose(-2, -1))
#
#         if attn_mask is not None:
#             m = attn_mask.reshape(B, 1, 1, attn_mask.shape[1])
#             attn = self.softmax_with_policy(attn, attn_mask) * m
#         else:
#             attn = attn.softmax(dim=-1)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
#         t_attn = x
#         return x, t_attn
#
#     def forward_local(self, x, x_size, mask):
#         """
#         q: image tokens
#         k: image tokens
#         v: image tokens
#         """
#         if mask.sum() == 0.0: return x
#         res = x
#         B, N, C_ = x.shape
#         ws = self.window_size
#         H, W = x_size
#         h = self.num_heads
#         hg, wg = H // ws, W // ws
#         x = rearrange(x, 'b (h hs w ws) c -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws)  # N, ws*ws, c
#         if self.training:
#             x = x * mask.reshape(-1, 1, 1)
#         else:
#             # x = x * mask.reshape(-1, 1, 1)
#             out_ = x
#             # x = x[mask==1.0, ...].contiguous()
#             # out_ = x
#             index = torch.nonzero(mask)
#             x = torch.gather(x, dim=0, index=index[:, :, None].expand(-1, ws * ws, C_))
#
#         qkv = self.qkv_local(x)
#         q, k, v = rearrange(qkv, 'n w (d h c) -> d n h w c', d=3, h=h).unbind(0)
#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))
#         pos_bias = self._get_relative_positional_bias()
#         attn = (attn + pos_bias).softmax(dim=-1)
#         out = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws * ws, -1)
#
#         if not self.training:
#             # out_ = torch.zeros(B * hg * wg, ws * ws, C_).cuda()
#             # out_[mask == 1.0, ...] = out
#             # out = out_
#
#             out = out_.scatter(0, index[:, :, None].expand(-1, ws * ws, C_), out)
#
#         # reverse
#         mask = mask.reshape(-1, 1, 1).expand(-1, ws * ws, -1)
#         out = window_reverse(out, ws, H, W).flatten(1, 2)
#         mask = window_reverse(mask, ws, H, W).flatten(1, 2)
#         x = out * mask + res * (1.0 - mask)
#         return x
#
#     def forward_global_broadcast(self, q, k, v):
#         """
#         q: image tokens
#         k: global tokens
#         v: global tokens
#         """
#         B, num_heads, N, _ = q.shape
#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))
#         attn = attn.softmax(dim=-1)
#         # attn = self.attn_drop(attn)
#         x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
#
#         return x
#
#     def forward_global_feature(self, x, attn_mask=None):
#         B, N, C = x.shape
#         NT = self.query_num
#
#         # qkv
#         qkv = self.qkv(x)
#         q, k, v = qkv.view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)  # B, h, N, c
#
#         # split img tokens & global tokens
#         q_img, k_img, v_img = q[:, :, NT:], k[:, :, NT:], v[:, :, NT:]
#         q_glb, _, _ = q[:, :, :NT], k[:, :, :NT], v[:, :, :NT]
#
#         # global aggregation
#         x_glb, t_img = self.forward_global_aggregation(q_glb, k_img, v_img, attn_mask)
#         k_glb, v_glb = self.kv_global(x_glb).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
#                                                                                                          4).unbind(0)
#         x_img = self.forward_global_broadcast(q_img, k_glb, v_glb)
#         return x_img
#
#     def forward_global(self, x, mask=None):
#         B, L, C = x.shape
#         if mask.sum() == 0.0: return x
#         query = self.prototype.unsqueeze(1).expand((-1, B, -1, -1)).flatten(0, 1)  # 2B, q_n, dim
#         query = self.norm1(query)
#         if self.training:
#             out = torch.cat([query, x], dim=1)
#             out = self.forward_global_feature(out, attn_mask=mask)
#         else:
#             # x = x.flatten(0, 1)  # (N, dim)
#             # mask = mask.flatten(0, 1).squeeze()
#             # select_x = x[mask == 1.0, :].unsqueeze(0)
#             # select_x = torch.cat([query, select_x], dim=1)
#             # select_x = self.forward_global_feature(select_x)
#             # out = torch.zeros_like(x)
#             # out[mask == 1.0, :] = select_x
#             # out = out.reshape(B, L, C)
#
#             out = x.clone().flatten(0, 1)
#             index = torch.nonzero(mask.flatten(0, 1))
#             select_x = x.flatten(0, 1)
#             select_x = select_x.gather(dim=0, index=index.expand(-1, C)).unsqueeze(0)
#             select_x = torch.cat([query, select_x], dim=1)
#             select_x = self.forward_global_feature(select_x).flatten(0, 1)
#
#             out.scatter_(0, index.expand(-1, C), select_x)
#             out = out.reshape(B, L, C)
#
#         x_global = out * mask.unsqueeze(-1) + x * (1.0 - mask).unsqueeze(-1)
#         return x_global
#
#     def forward(self, x, x_size, global_mask=None, local_mask=None):
#         H, W = x_size
#         # local
#         x_local = self.forward_local(x, (H, W), local_mask.squeeze(-1))
#         # global
#         global_mask = rearrange(global_mask, 'b c h w -> b (h w) c').squeeze(-1)
#         x_global = self.forward_global(x, global_mask)
#
#         return x_local + x_global  # x_global #x_local # x_local + x_global
#
#     def flops(self, x_size, local_mask, global_mask):
#         flops = 0
#         H, W = x_size
#         ws = self.window_size
#
#         ## 1. local
#         hg, wg = H // ws, W // ws
#         flops += (H * W * self.dim * self.dim * 3) # qkv
#
#         local_mask = rearrange(local_mask, 'b c (h hs) (w ws) -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws).squeeze(-1)
#         local_mask = local_mask.max(dim=-1).values
#
#         flops += (local_mask.sum() * ws * ws * ws * ws * self.dim) * 2
#
#         ## 2. global
#         # W-MSA/SW-MSA
#         flops += self.dim * self.query_num # norm
#         flops += (H * W * self.dim * self.dim * 3) # qkv
#
#         flops += (global_mask.sum() * self.query_num * self.dim // self.num_heads * self.num_heads) * 2
#         flops += (H * W * self.dim * self.dim * 2)
#         flops += (global_mask.sum() * self.query_num * self.dim // self.num_heads * self.num_heads) * 2
#
#         return flops

class LightViTAttention(nn.Module):
    def __init__(self, dim, query_class=1, query_num=256, num_heads=8, window_size=8,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm, shift=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        self.query_num = query_num
        self.query_class = query_class
        self.window_size = window_size
        self.attn_area = window_size * window_size
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_local = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.prototype = nn.Parameter(torch.randn((self.query_class, query_num, dim)))

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.prototype, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, H, N, HW = attn.size()
        attn_policy = policy.reshape(B, 1, 1, HW)

        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward_global_aggregation(self, q, k, v, attn_mask=None):
        """
        q: global tokens
        k: image tokens
        v: image tokens
        """
        B, _, N, _ = q.shape
        _, h, HW, c = k.shape
        q = q * self.scale
        if attn_mask is not None:
            attn = (q @ k.transpose(-2, -1)) * attn_mask.reshape(B, 1, 1, attn_mask.shape[1]).expand(-1, h, N, -1)
            m = attn_mask.reshape(B, 1, 1, attn_mask.shape[1])
            attn = self.softmax_with_policy(attn, attn_mask) * m
        else:
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x

    def forward_local(self, x, x_size, mask):
        """
        q: image tokens
        k: image tokens
        v: image tokens
        """
        if mask.sum() == 0.0: return x
        res = x
        B, N, C_ = x.shape
        ws = self.window_size
        H, W = x_size
        h = self.num_heads
        hg, wg = H // ws, W // ws
        x = rearrange(x, 'b (h hs w ws) c -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws)  # N, ws*ws, c
        if self.training:
            x = x * mask.reshape(-1, 1, 1)
        else:
            # x = x * mask.reshape(-1, 1, 1)
            out_ = x.clone()
            # x = x[mask==1.0, ...].contiguous()
            # out_ = x
            index = torch.nonzero(mask)
            x = torch.gather(x, dim=0, index=index[:, :, None].expand(-1, ws * ws, C_))

        qkv = self.qkv_local(x)
        q, k, v = rearrange(qkv, 'n w (d h c) -> d n h w c', d=3, h=h).unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        pos_bias = self._get_relative_positional_bias()
        attn = (attn + pos_bias).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws * ws, -1)

        if not self.training:
            # out_ = torch.zeros(B * hg * wg, ws * ws, C_).cuda()
            # out_[mask == 1.0, ...] = out
            # out = out_

            out = out_.scatter(0, index[:, :, None].expand(-1, ws * ws, C_), out)

        # reverse
        mask = mask.reshape(-1, 1, 1).expand(-1, ws * ws, -1)
        out = window_reverse(out, ws, H, W).flatten(1, 2)
        mask = window_reverse(mask, ws, H, W).flatten(1, 2)
        x = out * mask + res * (1.0 - mask)
        return x

    def forward_global_broadcast(self, q, k, v):
        """
        q: image tokens
        k: global tokens
        v: global tokens
        """
        B, num_heads, N, _ = q.shape
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        return x

    def forward_global_feature(self, x, attn_mask=None):
        B, N, C = x.shape
        NT = self.query_num

        # qkv
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)  # B, h, N, c

        # split img tokens & global tokens
        q_img, k_img, v_img = q[:, :, NT:], k[:, :, NT:], v[:, :, NT:]
        q_glb, _, _ = q[:, :, :NT], k[:, :, :NT], v[:, :, :NT]

        # global aggregation
        x_glb = self.forward_global_aggregation(q_glb, k_img, v_img, attn_mask)
        k_glb, v_glb = self.kv_global(x_glb).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        x_img = self.forward_global_broadcast(q_img, k_glb, v_glb)
        return x_img

    def forward_global(self, x, x_size=None, mask=None):
        B, L, C = x.shape
        res = x
        if mask.sum() == 0.0: return x
        query = self.prototype #.unsqueeze(1).expand((-1, B, -1, -1)).flatten(0, 1)  # 2B, q_n, dim
        query = self.norm1(query)
        if self.training:
            query = query.expand(B, -1, -1)
            out = torch.cat([query, x], dim=1)
            out = self.forward_global_feature(out, attn_mask=mask)
        else:
            # x = x.flatten(0, 1)  # (N, dim)
            # mask = mask.flatten(0, 1).squeeze()
            # select_x = x[mask == 1.0, :].unsqueeze(0)
            # select_x = torch.cat([query, select_x], dim=1)
            # select_x = self.forward_global_feature(select_x)
            # out = torch.zeros_like(x)
            # out[mask == 1.0, :] = select_x
            # out = out.reshape(B, L, C)

            # ==================
            # query = query.expand(B, -1, -1)
            # out = x.clone().flatten(0, 1)
            # index = torch.nonzero(mask.flatten(0, 1))
            # select_x = x.flatten(0, 1)
            # select_x = select_x.gather(dim=0, index=index.expand(-1, C)).unsqueeze(0)
            # select_x = torch.cat([query, select_x], dim=1)
            # select_x = self.forward_global_feature(select_x).flatten(0, 1)
            #
            # out.scatter_(0, index.expand(-1, C), select_x)
            # out = out.reshape(B, L, C)

            # ==============
            # H, W = x_size
            # split = min(2 ** round(math.log2(min(H * 1.0, W * 1.0) / 48)), 8)
            # chunk_size = 48
            # # print(split)
            # # hw = H // split
            # # ww = W // split
            # x = rearrange(x, 'b (h x w y) c -> (b h w) (x y) c', h=split, x=H // split, w=split, y=W // split)
            # reshape_mask = rearrange(mask, 'b (h x w y) -> (b h w) (x y)', h=split, x=H // split, w=split, y=W // split)
            # out = x.clone().flatten(0, 1)
            # index = torch.nonzero(reshape_mask.flatten(0, 1))
            # select_x = x.flatten(0, 1)
            # select_x = select_x.gather(dim=0, index=index.expand(-1, C))
            # num_select = select_x.size(0)
            # # padding = (split * split) - num_select % (split * split) if num_select % (split * split) != 0 else 0
            # padding = chunk_size - num_select % chunk_size if num_select % chunk_size != 0 else 0
            #
            # if padding and select_x.size(0) >= padding:
            #     pad_x = select_x[-padding:, :].clone()
            #     # pad_x = torch.ones((padding, select_x.size(-1)), device=select_x.device) * -1000.0
            #     select_x = torch.cat([select_x, pad_x], dim=0)
            #
            # group = split * split if select_x.size(0) // (split * split) > 0 else 1 # max(select_x.size(0) // (split * split), 1)
            # select_x = select_x.reshape(B * group, -1, C)  # [(B h w hw ww), C]
            # query = query.expand(B * group, -1, -1)
            #
            # select_x = torch.cat([query, select_x], dim=1)
            # select_x = self.forward_global_feature(select_x).flatten(0, 1)  # [(B h w) (hw ww), C]
            # select_x = select_x[:num_select]
            # out.scatter_(0, index.expand(-1, C), select_x)  # [(B h w hw ww), C]
            # out = rearrange(out, '(b h w x y) c -> b (h x w y) c', b=B, h=split, w=split, x=H // split, y=W // split)
            # # print(out.shape, mask.shape)

            # ============ test 最好 ==============
            H, W = x_size
            # split = 1
            split = min(2 ** round(math.log2(min(H * 1.0, W * 1.0) / 48)), 8)
            # split = 4
            # print(split)
            # split = min(H, W) // 48
            # hw = H // split
            # ww = W // split
            x = rearrange(x, 'b (h x w y) c -> (b h w) (x y) c', h=split, x=H // split, w=split, y=W // split)
            reshape_mask = rearrange(mask, 'b (h x w y) -> (b h w) (x y)', h=split, x=H // split, w=split, y=W // split)
            out = x.clone().flatten(0, 1)
            index = torch.nonzero(reshape_mask.flatten(0, 1))
            select_x = x.flatten(0, 1)
            select_x = select_x.gather(dim=0, index=index.expand(-1, C))
            num_select = select_x.size(0)
            padding = (split * split) - num_select % (split * split) if num_select % (split * split) != 0 else 0

            if padding and select_x.size(0) >= padding:
                pad_x = select_x[-padding:, :].clone()
                select_x = torch.cat([select_x, pad_x], dim=0)

            select_x = select_x.reshape(B * split * split, -1, C)  # [(B h w hw ww), C]
            query = query.expand(split * split, -1, -1)
            select_x = torch.cat([query, select_x], dim=1)
            select_x = self.forward_global_feature(select_x).flatten(0, 1)  # [(B h w) (hw ww), C]
            select_x = select_x[:num_select]
            out.scatter_(0, index.expand(-1, C), select_x)  # [(B h w hw ww), C]
            out = rearrange(out, '(b h w x y) c -> b (h x w y) c', b=B, h=split, w=split, x=H // split, y=W // split)

            # ================================
            # H, W = x_size
            # out = x.clone().flatten(0, 1)
            # index = torch.nonzero(mask.flatten(0, 1))
            # select_x = x.flatten(0, 1)
            # select_x = select_x.gather(dim=0, index=index.expand(-1, C))#.unsqueeze(0)
            # ss = select_x.size(0)
            # group = int( min(2 ** round(math.log2(ss / 48)), 8)) if int( min(2 ** round(math.log2(ss / 48)), 8)) > 0 else 1
            # chunk_size = ss // group
            # # print(ss, group, chunk_size)
            # padding = chunk_size - ss % (chunk_size) if ss % chunk_size != 0 else 0
            # if padding:
            #     pad_x = select_x[-padding:, :].clone()
            #     select_x = torch.cat([select_x, pad_x], dim=0)
            # ss_group = select_x.size(0) // chunk_size
            # select_x = select_x.reshape(ss_group, chunk_size, -1)
            # query = query.expand(ss_group, -1, -1)
            # # print(select_x.shape, query.shape)
            # select_x = torch.cat([query, select_x], dim=1)
            # select_x = self.forward_global_feature(select_x).flatten(0, 1)
            # select_x = select_x[:ss, ...]
            # out.scatter_(0, index.expand(-1, C), select_x)
            # out = out.reshape(B, L, C)

            #  =====================================
            # H, W = x_size
            # # split = 1
            # split = min(2 ** round(math.log2(min(H * 1.0, W * 1.0) / 48)), 8)
            # x = rearrange(x, 'b (h x w y) c -> (b h w) (x y) c', h=split, x=H // split, w=split, y=W // split)
            # reshape_mask = rearrange(mask, 'b (h x w y) -> (b h w) (x y)', h=split, x=H // split, w=split, y=W // split)
            # out = x.clone().flatten(0, 1)
            # index = torch.nonzero(reshape_mask.flatten(0, 1))
            # select_x = x.flatten(0, 1)
            # select_x = select_x.gather(dim=0, index=index.expand(-1, C))
            #
            # num_select = select_x.size(0)
            # ss = select_x.size(0)
            # # padding = (split * split) - num_select % (split * split) if num_select % (split * split) != 0 else 0
            # #
            # # if padding and select_x.size(0) >= padding:
            # #     pad_x = select_x[-padding:, :].clone()
            # #     select_x = torch.cat([select_x, pad_x], dim=0)
            #
            # group = int(min(2 ** round(math.log2(ss / 48)), 8)) if int(
            #     min(2 ** round(math.log2(ss / 48)), 8)) > 0 else 1
            # chunk_size = ss // group
            # # print(ss, group, chunk_size)
            # padding = chunk_size - ss % (chunk_size) if ss % chunk_size != 0 else 0
            # if padding:
            #     pad_x = select_x[-padding:, :].clone()
            #     select_x = torch.cat([select_x, pad_x], dim=0)
            # ss_group = select_x.size(0) // chunk_size
            # select_x = select_x.reshape(ss_group, chunk_size, -1)
            # query = query.expand(ss_group, -1, -1)
            # # print(select_x.shape, query.shape)
            # select_x = torch.cat([query, select_x], dim=1)
            #
            # select_x = self.forward_global_feature(select_x).flatten(0, 1)  # [(B h w) (hw ww), C]
            # select_x = select_x[:num_select]
            # out.scatter_(0, index.expand(-1, C), select_x)  # [(B h w hw ww), C]
            # out = rearrange(out, '(b h w x y) c -> b (h x w y) c', b=B, h=split, w=split, x=H // split, y=W // split)

        x_global = out * mask.unsqueeze(-1) + res * (1.0 - mask).unsqueeze(-1)
        return x_global

    def forward(self, x, x_size, global_mask=None, local_mask=None):
        H, W = x_size
        # local
        x_local = self.forward_local(x, (H, W), local_mask.squeeze(-1))
        # global
        global_mask = rearrange(global_mask, 'b c h w -> b (h w) c').squeeze(-1)
        x_global = self.forward_global(x, x_size, global_mask)

        return x_local + x_global  # x_global #x_local # x_local + x_global

    def flops(self, x_size, local_mask, global_mask):
        flops = 0
        H, W = x_size
        ws = self.window_size

        ## 1. local
        hg, wg = H // ws, W // ws
        flops += (H * W * self.dim * self.dim * 3) # qkv

        local_mask = rearrange(local_mask, 'b c (h hs) (w ws) -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws).squeeze(-1)
        local_mask = local_mask.max(dim=-1).values
        # print("num of local mask:", local_mask.sum(), hg * wg)

        flops += (local_mask.sum() * ws * ws * ws * ws * self.dim) * 2
        # flops += (hg * wg * ws * ws * ws * ws * self.dim) * 2

        ## 2. global
        # W-MSA/SW-MSA
        flops += self.dim * self.query_num # norm
        flops += (H * W * self.dim * self.dim * 3) # qkv

        # print("num of local mask:", global_mask.sum(), H * W)
        flops += (global_mask.sum() * self.query_num * self.dim // self.num_heads * self.num_heads) * 2
        # flops += (H * W * self.query_num * self.dim // self.num_heads * self.num_heads) * 2
        flops += (H * W * self.dim * self.dim * 2)
        flops += (global_mask.sum() * self.query_num * self.dim // self.num_heads * self.num_heads) * 2

        return flops

class LightViTAttentionWithoutAct(nn.Module):
    def __init__(self, dim,  query_class=1, query_num=256, num_heads=8, window_size=8,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm, shift=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        self.query_num = query_num
        self.query_class = query_class
        self.window_size = window_size
        self.attn_area = window_size * window_size
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_local = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.querynorm = nn.Sequential(
            nn.BatchNorm2d(dim)
        )
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.prototype = nn.Parameter(torch.randn((self.query_class, query_num, dim)))

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.prototype, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)


    def forward_global_aggregation(self, q, k, v):
        """
        q: global tokens
        k: image tokens
        v: image tokens
        """
        B, _, N, _ = q.shape
        _, h, HW, c = k.shape
        q = q * self.scale
        attn = (q @ k.transpose(-2,-1))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x

    def forward_local(self, x, x_size):
        """
        q: image tokens
        k: image tokens
        v: image tokens
        """
        B, N, C_ = x.shape
        ws = self.window_size
        H, W = x_size

        qkv = self.qkv_local(x)
        q, k, v = qkv.view(B, N, 3, self.num_heads, C_ // self.num_heads).permute(2, 0, 3, 1, 4).contiguous().unbind(0)
        B, num_heads, N, C = q.shape
        h_group, w_group = H // ws, W // ws

        # partition to windows
        q = q.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        q = q.view(-1, num_heads, ws*ws, C)
        k = k.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        k = k.view(-1, num_heads, ws*ws, C)
        v = v.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        v = v.view(-1, num_heads, ws*ws, v.shape[-1])

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        pos_bias = self._get_relative_positional_bias()
        attn = (attn + pos_bias).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws*ws, -1)

        # reverse
        out = window_reverse(out, ws, H, W).flatten(1, 2)
        return out

    def forward_global_broadcast(self, q, k, v):
        """
        q: image tokens
        k: global tokens
        v: global tokens
        """
        B, num_heads, N, _ = q.shape
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        return x

    def forward_global_feature(self, x):
        B, N, C = x.shape
        NT = self.query_num

        # qkv
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)  # B, h, N, c

        # split img tokens & global tokens
        q_img, k_img, v_img = q[:, :, NT:], k[:, :, NT:], v[:, :, NT:]
        q_glb, _, _ = q[:, :, :NT], k[:, :, :NT], v[:, :, :NT]

        # global aggregation
        x_glb = self.forward_global_aggregation(q_glb, k_img, v_img)
        k_glb, v_glb = self.kv_global(x_glb).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        x_img = self.forward_global_broadcast(q_img, k_glb, v_glb)
        return x_img

    def forward_global(self, x, mask=None):
        B, L, C = x.shape
        query = self.prototype.unsqueeze(1).expand((-1, B, -1, -1)).flatten(0, 1)  # 2B, q_n, dim
        query = self.norm1(query)
        out = torch.cat([query, x], dim=1)
        x_global = self.forward_global_feature(out)

        return x_global

    def forward(self, x, x_size):
        H, W = x_size
        # local
        x_local = self.forward_local(x, (H, W))
        # global
        x_global = self.forward_global(x)

        return x_global + x_local

    def flops(self, x_size, local_mask=None, global_mask=None):
        flops = 0
        H, W = x_size
        ws = self.window_size

        ## 1. local
        hg, wg = H // ws, W // ws
        flops += (H * W * self.dim * self.dim * 3) # qkv

        # local_mask = rearrange(local_mask, 'b c (h hs) (w ws) -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws).squeeze(-1)
        # local_mask = local_mask.max(dim=-1).values

        flops += ( ws * ws * ws * ws * self.dim) * 2

        ## 2. global
        # W-MSA/SW-MSA
        flops += self.dim * self.query_num # norm
        flops += (H * W * self.dim * self.dim * 3) # qkv

        flops += (self.query_num * self.dim // self.num_heads * self.num_heads) * 2
        flops += (H * W * self.dim * self.dim * 2)
        flops += (self.query_num * self.dim // self.num_heads * self.num_heads) * 2

        return flops

class LightViTAttention_v2(nn.Module):
    def __init__(self, dim,  query_class=1, query_num=256, num_heads=8, window_size=8,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.query_num = query_num
        self.query_class = query_class
        self.window_size = window_size
        self.attn_area = window_size * window_size
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_local = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv_global_2 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.querynorm = nn.Sequential(
            nn.BatchNorm2d(dim)
        )
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.prototype = nn.Parameter(torch.randn((self.query_class, query_num, dim)))

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.prototype, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)


    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, H, N, HW = attn.size()
        attn_policy = policy.reshape(B, 1, 1, HW)

        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)


    def forward_local(self, x, x_size, mask):
        """
        q: image tokens
        k: image tokens
        v: image tokens
        """
        B, N, C_ = x.shape
        ws = self.window_size
        H, W = x_size

        qkv = self.qkv_local(x)
        q, k, v = qkv.view(B, N, 3, self.num_heads, C_ // self.num_heads).permute(2, 0, 3, 1, 4).contiguous().unbind(0)
        B, num_heads, N, C = q.shape
        h_group, w_group = H // ws, W // ws

        # partition to windows
        q = q.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        q = q.view(-1, num_heads, ws*ws, C)
        k = k.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        k = k.view(-1, num_heads, ws*ws, C)
        v = v.view(B, num_heads, h_group, ws, w_group, ws, -1).permute(0, 2, 4, 1, 3, 5, 6).contiguous()
        v = v.view(-1, num_heads, ws*ws, v.shape[-1])

        q = q * self.scale
        # if self.training:
        #     q = q * mask.reshape(-1, 1, 1, 1)
        #     k = k * mask.reshape(-1, 1, 1, 1)
        #     v = v * mask.reshape(-1, 1, 1, 1)
        #
        #     attn = (q @ k.transpose(-2, -1))
        #     pos_bias = self._get_relative_positional_bias()
        #     attn = (attn + pos_bias).softmax(dim=-1)
        #     out = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws*ws, -1)
        #
        # else:
        #     out = torch.zeros(B*h_group*w_group, ws*ws, C_).cuda()
        #
        #     # q = q[mask==1.0, ...].contiguous()
        #     # k = k[mask==1.0, ...].contiguous()
        #     # v = v[mask==1.0, ...].contiguous()
        #
        #     # q = torch.masked_select(q, mask.bool()[:, None, None, None]).reshape(-1, num_heads, ws*ws, C)
        #     # k = torch.masked_select(k, mask.bool()[:, None, None, None]).reshape(-1, num_heads, ws*ws, C)
        #     # v = torch.masked_select(v, mask.bool()[:, None, None, None]).reshape(-1, num_heads, ws*ws, C)
        #
        #     # if min(q.shape) != 0:
        #     #     torch.cuda.synchronize()
        #     #     time_start = time.time()
        #     #     attn = (q @ k.transpose(-2, -1))
        #     #     pos_bias = self._get_relative_positional_bias()
        #     #     attn = (attn + pos_bias).softmax(dim=-1)
        #     #     select_x = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws * ws, -1)
        #     #     torch.cuda.synchronize()
        #     #     time_end = time.time()
        #     #     print('select time:',time_end - time_start)
        #     #     out[mask==1.0, ...] = select_x
        #     # torch.cuda.synchronize()
        #     # time_start = time.time()
        #     q_ = q[mask==1.0, ...].contiguous()
        #     k_ = k[mask==1.0, ...].contiguous()
        #     v_ = v[mask==1.0, ...].contiguous()
        #     # index = torch.nonzero(mask).squeeze()
        #     # q_ = torch.index_select(q, dim=0, index=index).contiguous()
        #     # k_ = torch.index_select(k, dim=0, index=index).contiguous()
        #     # v_ = torch.index_select(v, dim=0, index=index).contiguous()
        #     # index_ = index[:, :, None, None].expand(-1, num_heads, ws*ws, C)
        #     # q_ = torch.gather(q, dim=0, index=index_)
        #     # k_ = torch.gather(k, dim=0, index=index_)
        #     # v_ = torch.gather(v, dim=0, index=index_)
        #     if min(q_.shape) != 0:
        #
        #         attn = (q_ @ k_.transpose(-2, -1))
        #         pos_bias = self._get_relative_positional_bias()
        #         attn = (attn + pos_bias).softmax(dim=-1)
        #         select_x = (attn @ v_).transpose(1, 2).reshape(v_.shape[0], ws * ws, -1)
        #
        #         out[mask==1.0, ...] = select_x
        #         # torch.cuda.synchronize()
        #         # time_end = time.time()
        #         # print('select time:', time_end - time_start)

        q = q * mask.reshape(-1, 1, 1, 1)
        k = k * mask.reshape(-1, 1, 1, 1)
        v = v * mask.reshape(-1, 1, 1, 1)
        # print(q.shape)
        # torch.cuda.synchronize()
        # time_start = time.time()
        attn = (q @ k.transpose(-2, -1))
        pos_bias = self._get_relative_positional_bias()
        attn = (attn + pos_bias).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws * ws, -1)
        # torch.cuda.synchronize()
        # time_end = time.time()
        # print('no attn select time:', time_end - time_start)
        # reverse
        mask = mask.reshape(-1, 1, 1).expand(-1, ws*ws, -1)
        out = window_reverse(out, ws, H, W).flatten(1, 2)
        mask = window_reverse(mask, ws, H, W).flatten(1, 2)
        x = out * mask + x * (1.0 - mask)
        return x

    def forward_global_aggregation(self, q, k, v, attn_mask=None):
        """
        q: global tokens
        k: image tokens
        v: image tokens
        """
        B, _, N, _ = q.shape
        _, h, HW, c = k.shape
        q = q * self.scale
        if attn_mask is not None:
            attn = (q @ k.transpose(-2, -1)) * attn_mask.reshape(B, 1, 1, attn_mask.shape[1]).expand(-1, h, N, -1)
        else:
            attn = (q @ k.transpose(-2, -1))

        if attn_mask is not None:
            m = attn_mask.reshape(B, 1, 1, attn_mask.shape[1])
            attn = self.softmax_with_policy(attn, attn_mask) * m
        else:
            attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        t_attn = x
        return x, t_attn

    def forward_global_broadcast(self, q, k, v):
        """
        q: image tokens
        k: global tokens
        v: global tokens
        """
        B, num_heads, N, _ = q.shape
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        return x

    def forward_global_feature(self, x, attn_mask=None):
        B, N, C = x.shape
        NT = self.query_num

        # qkv
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)  # B, h, N, c

        # split img tokens & global tokens
        q_img, k_img, v_img = q[:, :, NT:], k[:, :, NT:], v[:, :, NT:]
        q_glb, k_glb, v_glb = q[:, :, :NT], k[:, :, :NT], v[:, :, :NT]

        # global aggregation
        x_glb, t_img = self.forward_global_aggregation(q_glb, k_img, v_img, attn_mask)
        # k_glb, v_glb = self.kv_global(self.norm2(x_glb)).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        k_glb, v_glb = self.kv_global(x_glb).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        x_glb = self.forward_global_broadcast(k_glb, k_glb, v_glb)
        # k_glb, v_glb = self.kv_global_2(self.norm3(x_glb)).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
        k_glb, v_glb = self.kv_global_2(x_glb).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                         4).unbind(0)
        x_img = self.forward_global_broadcast(q_img, k_glb, v_glb)
        return x_img

    def forward_global(self, x, mask=None):
        B, L, C = x.shape
        query = self.prototype.unsqueeze(1).expand((-1, B, -1, -1)).flatten(0, 1)  # 2B, q_n, dim
        query = self.norm1(query)
        # if self.training:
        #     out = torch.cat([query, x], dim=1)
        #     out = self.forward_global_feature(out, attn_mask=mask)
        # else:
        #     x = x.flatten(0, 1)  # (N, dim)
        #     mask = mask.flatten(0, 1).squeeze()
        #     # sum_ = int(mask.sum())
        #     # _, indices = mask.sort(dim=-1)
        #     # _, undo_sort = indices.sort(dim=-1)
        #     # print(mask.shape, indices.shape, undo_sort.shape)
        #     # index = torch.nonzero(mask).squeeze()
        #     # select_x = torch.index_select(x, dim=0, index=indices[:sum_]).unsqueeze(0)
        #     select_x = x[mask == 1.0, :].unsqueeze(0)
        #     if min(select_x.shape) != 0:
        #         select_x = torch.cat([query, select_x], dim=1)
        #         select_x = self.forward_global_feature(select_x)
        #     out = torch.zeros_like(x)
        #     # print(select_x.shape)
        #     # out = torch.zeros((B, L - sum_, C)).cuda()
        #     # out = torch.cat([select_x, out], dim=1)
        #     # out = out.gather(1, undo_sort[None, :, None].expand(B, -1, C))
        #     out[mask == 1.0, :] = select_x
        #     # out = out.reshape(B, L, C)
        #
        out = torch.cat([query, x], dim=1)
        out = self.forward_global_feature(out, attn_mask=mask)
        x_global = out * mask.unsqueeze(-1) + x * (1.0 - mask).unsqueeze(-1)
        return x_global

    def forward(self, x, x_size, global_mask=None, local_mask=None):
        H, W = x_size
        # local
        # x_local = self.forward_local(x, (H, W), local_mask.squeeze(-1))
        # global
        global_mask = rearrange(global_mask, 'b c h w -> b (h w) c').squeeze(-1)
        x_global = self.forward_global(x, global_mask)

        return  x_global #+ x_global #x_global #x_local # x_local + x_global

class LightViTAttention_LQ(nn.Module):
    def __init__(self, dim,  query_class=1, query_num=256, num_heads=8, window_size=8,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.query_num = query_num
        self.query_class = query_class
        self.window_size = window_size
        self.attn_area = window_size * window_size
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_local = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.querynorm = nn.Sequential(
            nn.BatchNorm2d(dim)
        )
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.prototype = nn.Parameter(torch.randn((self.query_class, query_num, dim)))
        self.local_prototype = nn.Parameter(torch.randn((self.query_class, query_num, dim)))
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.prototype, std=.02)
        trunc_normal_(self.local_prototype, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)


    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, H, N, HW = attn.size()
        attn_policy = policy.reshape(B, 1, 1, HW)

        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward_global_aggregation(self, q, k, v, attn_mask=None):
        """
        q: global tokens
        k: image tokens
        v: image tokens
        """
        B, _, N, _ = q.shape
        _, h, HW, c = k.shape
        q = q * self.scale
        if attn_mask is not None:
            attn = (q @ k.transpose(-2, -1)) * attn_mask.reshape(B, 1, 1, attn_mask.shape[1]).expand(-1, h, N, -1)
        else:
            attn = (q @ k.transpose(-2, -1))

        if attn_mask is not None:
            m = attn_mask.reshape(B, 1, 1, attn_mask.shape[1])
            attn = self.softmax_with_policy(attn, attn_mask) * m
        else:
            attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        t_attn = x
        return x, t_attn

    def forward_local(self, x, x_size, mask):
        """
        q: image tokens
        k: image tokens
        v: image tokens
        """
        if mask.sum() == 0.0: return x
        res = x
        B, N, C_ = x.shape
        ws = self.window_size
        H, W = x_size
        h = self.num_heads
        hg, wg = H // ws, W // ws

        n = H*W if self.training else mask.sum()
        local_query = self.norm2(self.local_prototype.expand(B, n, -1, -1))

        x = rearrange(x, 'b (h hs w ws) c -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws)  # N, ws*ws, c
        if self.training:
            x = x * mask.reshape(-1, 1, 1)
        else:
            x = x[mask == 1.0, ...]

        qkv = self.qkv_local(x)
        q, k, v = rearrange(qkv, 'n w (d h c) -> d n h w c', d=3, h=h).unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        pos_bias = self._get_relative_positional_bias()
        attn = (attn + pos_bias).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws * ws, -1)

        if not self.training:
            out_ = torch.zeros(B * hg * wg, ws * ws, C_).cuda()
            out_[mask == 1.0, ...] = out
            out = out_

        # reverse
        mask = mask.reshape(-1, 1, 1).expand(-1, ws * ws, -1)
        out = window_reverse(out, ws, H, W).flatten(1, 2)
        mask = window_reverse(mask, ws, H, W).flatten(1, 2)
        x = out * mask + res * (1.0 - mask)
        return x

    def forward_global_broadcast(self, q, k, v):
        """
        q: image tokens
        k: global tokens
        v: global tokens
        """
        B, num_heads, N, _ = q.shape
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        return x

    def forward_global_feature(self, x, attn_mask=None):
        B, N, C = x.shape
        NT = self.query_num

        # qkv
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)  # B, h, N, c

        # split img tokens & global tokens
        q_img, k_img, v_img = q[:, :, NT:], k[:, :, NT:], v[:, :, NT:]
        q_glb, _, _ = q[:, :, :NT], k[:, :, :NT], v[:, :, :NT]

        # global aggregation
        x_glb, t_img = self.forward_global_aggregation(q_glb, k_img, v_img, attn_mask)
        k_glb, v_glb = self.kv_global(x_glb).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        x_img = self.forward_global_broadcast(q_img, k_glb, v_glb)
        return x_img

    def forward_global(self, x, mask=None):
        B, L, C = x.shape
        query = self.norm1(self.prototype).unsqueeze(1).expand((-1, B, -1, -1)).flatten(0, 1)  # 2B, q_n, dim

        # if self.training:
        #     out = torch.cat([query, x], dim=1)
        #     out = self.forward_global_feature(out, attn_mask=mask)
        # else:
        #     x = x.flatten(0, 1)  # (N, dim)
        #     mask = mask.flatten(0, 1).squeeze()
        #     select_x = x[mask == 1.0, :].unsqueeze(0)
        #     if min(select_x.shape) != 0:
        #         select_x = torch.cat([query, select_x], dim=1)
        #         select_x = self.forward_global_feature(select_x)
        #     out = torch.zeros_like(x)
        #     out[mask == 1.0, :] = select_x
        #     out = out.reshape(B, L, C)
        out = torch.cat([query, x], dim=1)
        out = self.forward_global_feature(out, attn_mask=mask)
        x_global = out * mask.unsqueeze(-1) + x * (1.0 - mask).unsqueeze(-1)
        return x_global

    def forward(self, x, x_size, global_mask=None, local_mask=None):
        H, W = x_size
        # local
        # x_local = self.forward_local(x, (H, W), local_mask.squeeze(-1))
        # global
        global_mask = rearrange(global_mask, 'b c h w -> b (h w) c').squeeze(-1)
        x_global = self.forward_global(x, global_mask)

        return  x_global #x_global #x_local # x_local + x_global

class LightViTAttention_Global_Group(nn.Module): # local select change
    def __init__(self, dim,  query_class=8, query_num=256, num_heads=8, window_size=8,
                 qkv_bias=False, qk_scale=None, group=2, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.query_num = query_num
        self.query_class = query_class
        self.window_size = window_size
        self.attn_area = window_size * window_size
        self.scale = qk_scale or head_dim ** -0.5
        self.group = group
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_local = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.querynorm = nn.Sequential(
            nn.BatchNorm2d(dim)
        )
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)

        self.prototype = nn.Parameter(torch.randn((self.query_class, query_num, dim)))
        self.linear = nn.Linear(self.query_class, 1)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.prototype, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)


    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, L, H, N, HW = attn.size()
        attn_policy = policy.reshape(B, 1, 1, 1, HW)

        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward_global_aggregation(self, q, k, v, attn_mask=None):
        """
        q: global tokens
        k: image tokens
        v: image tokens
        """
        # B, _, N, _ = q.shape
        B, h, HW, c = k.shape
        q = q * self.scale
        # if attn_mask is not None:
        #     attn = (q @ k.transpose(-2, -1)) * attn_mask.reshape(B, 1, 1, attn_mask.shape[1]).expand(-1, h, N, -1)
        # else:
        # attn = (q @ k.transpose(-2, -1))
        attn = torch.einsum('b n h l d, b h k d -> b n h l k', [q, k])
        if attn_mask is not None:
            m = attn_mask.reshape(B, 1, 1, 1, attn_mask.shape[1])
            attn = self.softmax_with_policy(attn, attn_mask) * m
        else:
            attn = attn.softmax(dim=-1)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, -1) # b h l d
        x = torch.einsum('b n h l k, b h k d -> b n h l d', [attn, v])
        # x = rearrange(x, 'b n h l d -> b n l (h d)')
        # x = x.mean(dim=1)
        x = rearrange(x, 'b n h l d -> b n l (h d)')
        # print(x.shape)
        # x = self.linear(x).squeeze(-1)
        # x = torch.sum(torch.nn.functional.softmax(x, dim=1) * x, dim = 1)
        return x

    def forward_local(self, x, x_size, mask):
        """
        q: image tokens
        k: image tokens
        v: image tokens
        """
        if mask.sum() == 0.0: return x
        res = x
        B, N, C_ = x.shape
        ws = self.window_size
        H, W = x_size
        h  = self.num_heads
        hg, wg = H // ws, W // ws
        x = rearrange(x, 'b (h hs w ws) c -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws) # N, ws*ws, c
        if self.training:
            x = x * mask.reshape(-1, 1, 1)
        else:
            # x = x * mask.reshape(-1, 1, 1)
            out_ = x.clone()
            x = x[mask==1.0, ...]
            # index = torch.nonzero(mask)
            # x = torch.gather(x, dim=0, index=index[:, :, None].expand(-1, ws*ws, C_)).contiguous()

        qkv = self.qkv_local(x)
        q, k, v = rearrange(qkv, 'n w (d h c) -> d n h w c', d = 3, h = h).unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        pos_bias = self._get_relative_positional_bias()
        attn = (attn + pos_bias).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws * ws, -1)

        if not self.training:
            # out_ = torch.zeros(B * hg * wg, ws * ws, C_).cuda()
            out_[mask == 1.0, ...] = out
            out = out_

        # reverse
        mask = mask.reshape(-1, 1, 1).expand(-1, ws*ws, -1)
        out = window_reverse(out, ws, H, W).flatten(1, 2)
        mask = window_reverse(mask, ws, H, W).flatten(1, 2)
        x = out * mask + res * (1.0 - mask)
        return x

    def forward_global_broadcast(self, q, k, v):
        """
        q: image tokens
        k: global tokens
        v: global tokens
        """
        B, num_heads, N, _ = q.shape
        q = q * self.scale
        # attn = (q @ k.transpose(-2, -1))
        attn = torch.einsum('b h l d, b n h k d -> b n h l k', [q, k])
        attn = attn.softmax(dim=-1)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = torch.einsum('b n h l k, b n h k d -> b n h l d', [attn, v])
        x = rearrange(x, 'b n h l d -> b n l (h d)')
        x = torch.sum(torch.nn.functional.softmax(x, dim=1) * x, dim=1)
        # print(x.shape)
        return x

    def forward_global_feature(self, x, attn_mask=None):
        B, N, C = x.shape
        NT = self.query_num
        query = self.norm1(self.prototype).unsqueeze(0).expand((B, -1, -1, -1))  # B, n, q_n, dim
        q_glb = rearrange(query, 'b n q (h d) -> b n h q d', h=self.num_heads)

        # qkv
        qkv = self.qkv(x)
        q_img, k_img, v_img = qkv.view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)  # B, h, N, c

        # split img tokens & global tokens
        # q_img, k_img, v_img = q[:, :, NT:], k[:, :, NT:], v[:, :, NT:]
        # q_glb, _, _ = q[:, :, :NT], k[:, :, :NT], v[:, :, :NT]

        # global aggregation
        x_glb = self.forward_global_aggregation(q_glb, k_img, v_img, attn_mask) # [b n l c]
        k_glb, v_glb = self.kv_global(x_glb).view(B, -1, NT, 2, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5).unbind(0)  # [2, b n h l c]
        x_img = self.forward_global_broadcast(q_img, k_glb, v_glb)
        return x_img

    def forward_global(self, x, mask=None):
        B, L, C = x.shape
        if mask.sum() == 0.0: return x
        if self.training:
            # out = torch.cat([query, x], dim=1)
            # out = self.forward_global_feature(out, attn_mask=mask)
            out = self.forward_global_feature(x, attn_mask=mask)
        else:
            x = x.flatten(0, 1)  # (N, dim)
            mask = mask.flatten(0, 1).squeeze()
            select_x = x[mask == 1.0, :].unsqueeze(0)
            # select_x = torch.cat([query, select_x], dim=1)
            select_x = self.forward_global_feature(select_x)
            out = torch.zeros_like(x)
            out[mask == 1.0, :] = select_x
            out = out.reshape(B, L, C)

        x_global = out * mask.unsqueeze(-1) + x * (1.0 - mask).unsqueeze(-1)
        return x_global

    def forward(self, x, x_size, global_mask=None, local_mask=None):
        H, W = x_size
        # local
        # x_local = self.forward_local(x, (H, W), local_mask.squeeze(-1))


        # global
        global_mask = rearrange(global_mask, 'b c h w -> b (h w) c').squeeze(-1)
        x_global = self.forward_global(x, global_mask)

        return  x_global # x_global # x_local # x_local + x_global

class LightViTAttentionv3(nn.Module):
    def __init__(self, dim,  query_class=1, query_num=256, num_heads=8, window_size=8,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.query_num = query_num
        self.query_class = query_class
        self.window_size = window_size
        self.attn_area = window_size * window_size
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_local = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.querynorm = nn.Sequential(
            nn.BatchNorm2d(dim)
        )
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.prototype = nn.Parameter(torch.randn((self.query_class, query_num, dim)))

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.prototype, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)


    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, H, N, HW = attn.size()
        attn_policy = policy.reshape(B, 1, 1, HW)

        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward_global_aggregation(self, q, k, v, attn_mask=None):
        """
        q: global tokens
        k: image tokens
        v: image tokens
        """
        B, _, N, _ = q.shape
        _, h, HW, c = k.shape
        q = q * self.scale
        if attn_mask is not None:
            attn = (q @ k.transpose(-2, -1)) * attn_mask.reshape(B, 1, 1, attn_mask.shape[1]).expand(-1, h, N, -1)
        else:
            attn = (q @ k.transpose(-2, -1))

        if attn_mask is not None:
            m = attn_mask.reshape(B, 1, 1, attn_mask.shape[1])
            attn = self.softmax_with_policy(attn, attn_mask) * m
        else:
            attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        t_attn = x
        return x, t_attn

    def forward_local(self, x, x_size, mask):
        """
        q: image tokens
        k: image tokens
        v: image tokens
        """
        if mask.sum() == 0.0: return x
        res = x
        B, N, C_ = x.shape
        ws = self.window_size
        H, W = x_size
        h = self.num_heads
        hg, wg = H // ws, W // ws
        x = rearrange(x, 'b (h hs w ws) c -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws)  # N, ws*ws, c
        if self.training:
            x = x * mask.reshape(-1, 1, 1)
        else:
            # x = x * mask.reshape(-1, 1, 1)
            x = x[mask == 1.0, ...].contiguous()
            # index = torch.nonzero(mask)
            # x = torch.gather(x, dim=0, index=index[:, :, None].expand(-1, ws*ws, C_)).contiguous()

        qkv = self.qkv_local(x)
        q, k, v = rearrange(qkv, 'n w (d h c) -> d n h w c', d=3, h=h).unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        pos_bias = self._get_relative_positional_bias()
        attn = (attn + pos_bias).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws * ws, -1)

        if not self.training:
            out_ = torch.zeros(B * hg * wg, ws * ws, C_).cuda()
            out_[mask == 1.0, ...] = out
            out = out_

        # reverse
        mask = mask.reshape(-1, 1, 1).expand(-1, ws * ws, -1)
        out = window_reverse(out, ws, H, W).flatten(1, 2)
        mask = window_reverse(mask, ws, H, W).flatten(1, 2)
        x = out * mask + res * (1.0 - mask)
        return x

    def forward_global_broadcast(self, q, k, v):
        """
        q: image tokens
        k: global tokens
        v: global tokens
        """
        B, num_heads, N, _ = q.shape
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        return x

    def forward_global_feature(self, x, attn_mask=None):
        B, N, C = x.shape
        NT = self.query_num

        # qkv
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)  # B, h, N, c

        # split img tokens & global tokens
        q_img, k_img, v_img = q[:, :, NT:], k[:, :, NT:], v[:, :, NT:]
        q_glb, _, _ = q[:, :, :NT], k[:, :, :NT], v[:, :, :NT]

        # global aggregation
        x_glb, t_img = self.forward_global_aggregation(q_glb, k_img, v_img, attn_mask)
        k_glb, v_glb = self.kv_global(x_glb).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        x_img = self.forward_global_broadcast(q_img, k_glb, v_glb)
        return x_img

    def forward_global(self, x, mask=None):
        B, L, C = x.shape
        query = self.prototype.unsqueeze(1).expand((-1, B, -1, -1)).flatten(0, 1)  # 2B, q_n, dim
        query = self.norm1(query)
        if self.training:
            out = torch.cat([query, x], dim=1)
            out = self.forward_global_feature(out, attn_mask=mask) * mask.unsqueeze(-1)
        else:
            x = x.flatten(0, 1)  # (N, dim)
            mask = mask.flatten(0, 1).squeeze()
            select_x = x[mask == 1.0, :].unsqueeze(0)
            if min(select_x.shape) != 0:
                select_x = torch.cat([query, select_x], dim=1)
                select_x = self.forward_global_feature(select_x)
            out = torch.zeros_like(x)
            out[mask == 1.0, :] = select_x
            out = out.reshape(B, L, C)
        # out = torch.cat([query, x], dim=1)
        # out = self.forward_global_feature(out, attn_mask=mask)
        x_global = out
        return x_global

    def forward(self, x, x_size, global_mask=None, local_mask=None):
        H, W = x_size
        # local
        x_local = self.forward_local(x, (H, W), local_mask.squeeze(-1))
        # global
        global_mask = rearrange(global_mask, 'b c h w -> b (h w) c').squeeze(-1)
        x_global = self.forward_global(x, global_mask)

        return  x_local + x_global # x_global #x_local # x_local + x_global

class LightViTAttentionLocalMask(nn.Module):
    def __init__(self, dim, query_class=1, query_num=256, num_heads=8, window_size=8,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm, shift=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        self.query_num = query_num
        self.query_class = query_class
        self.window_size = window_size
        self.attn_area = window_size * window_size
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_local = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.prototype = nn.Parameter(torch.randn((self.query_class, query_num, dim)))

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.prototype, std=.02)
        print("==== LightViTAttentionLocalMask ===")

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, H, N, HW = attn.size()
        attn_policy = policy.reshape(B, 1, 1, HW)

        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward_global_aggregation(self, q, k, v, attn_mask=None):
        """
        q: global tokens
        k: image tokens
        v: image tokens
        """
        B, _, N, _ = q.shape
        _, h, HW, c = k.shape
        q = q * self.scale
        if attn_mask is not None:
            attn = (q @ k.transpose(-2, -1)) * attn_mask.reshape(B, 1, 1, attn_mask.shape[1]).expand(-1, h, N, -1)
            m = attn_mask.reshape(B, 1, 1, attn_mask.shape[1])
            attn = self.softmax_with_policy(attn, attn_mask) * m
        else:
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x

    def forward_local(self, x, x_size, mask, global_mask):
        """
        q: image tokens
        k: image tokens
        v: image tokens
        """
        if mask.sum() == 0.0: return x
        res = x
        B, N, C_ = x.shape
        ws = self.window_size
        H, W = x_size
        h = self.num_heads
        hg, wg = H // ws, W // ws

        x = rearrange(x, 'b (h hs w ws) c -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws)  # N, ws*ws, c
        glb_mask = rearrange(global_mask, 'b c (h hs) (w ws) -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws)
        if self.training:
            x = x * mask.reshape(-1, 1, 1) * glb_mask
        else:
            # x = x * mask.reshape(-1, 1, 1)
            out_ = x.clone()
            # x = x[mask==1.0, ...].contiguous()
            # out_ = x
            index = torch.nonzero(mask)
            x = torch.gather(x, dim=0, index=index[:, :, None].expand(-1, ws * ws, C_))
            glb_mask = torch.gather(glb_mask, dim=0, index=index[:, :, None].expand(-1, ws * ws, 1))
            x *= glb_mask

        qkv = self.qkv_local(x)
        q, k, v = rearrange(qkv, 'n w (d h c) -> d n h w c', d=3, h=h).unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        pos_bias = self._get_relative_positional_bias()
        attn = attn + pos_bias
        attn = self.softmax_with_policy(attn, glb_mask.squeeze(-1))
        out = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws * ws, -1)

        if not self.training:
            # out_ = torch.zeros(B * hg * wg, ws * ws, C_).cuda()
            # out_[mask == 1.0, ...] = out
            # out = out_

            out = out_.scatter(0, index[:, :, None].expand(-1, ws * ws, C_), out)

        # reverse
        # mask = mask.reshape(-1, 1, 1).expand(-1, ws * ws, -1)
        # out = window_reverse(out, ws, H, W).flatten(1, 2)
        # mask = window_reverse(glb_mask, ws, H, W).flatten(1, 2)
        out = rearrange(out, '(b h w) (hs ws) c -> b (h hs w ws) c', h=hg, hs=ws, w=wg, ws=ws)
        mask = rearrange(global_mask, 'b c h w -> b (h w) c')
        x = out * mask + res * (1.0 - mask)
        return x

    def forward_global_broadcast(self, q, k, v):
        """
        q: image tokens
        k: global tokens
        v: global tokens
        """
        B, num_heads, N, _ = q.shape
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        return x

    def forward_global_feature(self, x, attn_mask=None):
        B, N, C = x.shape
        NT = self.query_num

        # qkv
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)  # B, h, N, c

        # split img tokens & global tokens
        q_img, k_img, v_img = q[:, :, NT:], k[:, :, NT:], v[:, :, NT:]
        q_glb, _, _ = q[:, :, :NT], k[:, :, :NT], v[:, :, :NT]

        # global aggregation
        x_glb = self.forward_global_aggregation(q_glb, k_img, v_img, attn_mask)
        k_glb, v_glb = self.kv_global(x_glb).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        x_img = self.forward_global_broadcast(q_img, k_glb, v_glb)
        return x_img

    def forward_global(self, x, x_size=None, mask=None):
        B, L, C = x.shape
        res = x
        if mask.sum() == 0.0: return x
        # query = self.prototype.unsqueeze(1).expand((-1, B, -1, -1)).flatten(0, 1)  # 2B, q_n, dim
        # query = self.norm1(query)

        # query = self.prototype.unsqueeze(1).expand((-1, B, -1, -1)).flatten(0, 1)  # 2B, q_n, dim
        query = self.norm1(self.prototype)
        if self.training:
            query = self.prototype.unsqueeze(1).expand((-1, B, -1, -1)).flatten(0, 1)  # 2B, q_n, dim
            out = torch.cat([query, x], dim=1)
            out = self.forward_global_feature(out, attn_mask=mask)
        else:
            # x = x.flatten(0, 1)  # (N, dim)
            # mask = mask.flatten(0, 1).squeeze()
            # select_x = x[mask == 1.0, :].unsqueeze(0)
            # select_x = torch.cat([query, select_x], dim=1)
            # select_x = self.forward_global_feature(select_x)
            # out = torch.zeros_like(x)
            # out[mask == 1.0, :] = select_x
            # out = out.reshape(B, L, C)

            # =======================
            out = x.clone().flatten(0, 1)
            index = torch.nonzero(mask.flatten(0, 1))
            select_x = x.flatten(0, 1)
            select_x = select_x.gather(dim=0, index=index.expand(-1, C)).unsqueeze(0)
            select_x = torch.cat([query, select_x], dim=1)
            select_x = self.forward_global_feature(select_x).flatten(0, 1)

            out.scatter_(0, index.expand(-1, C), select_x)
            out = out.reshape(B, L, C)

            # ==========================
            # H, W = x_size
            # # split = 1
            # split = min(2 ** round(math.log2(min(H * 1.0, W * 1.0) / 48)), 8)
            # # split = min(H, W) // 48
            # # hw = H // split
            # # ww = W // split
            # x = rearrange(x, 'b (h x w y) c -> (b h w) (x y) c', h=split, x=H // split, w=split, y=W // split)
            # reshape_mask = rearrange(mask, 'b (h x w y) -> (b h w) (x y)', h=split, x=H // split, w=split, y=W // split)
            # out = x.clone().flatten(0, 1)
            # index = torch.nonzero(reshape_mask.flatten(0, 1))
            # select_x = x.flatten(0, 1)
            # select_x = select_x.gather(dim=0, index=index.expand(-1, C))
            # num_select = select_x.size(0)
            # padding = (split * split) - num_select % (split * split) if num_select % (split * split) != 0 else 0
            #
            # if padding and select_x.size(0) >= padding:
            #     pad_x = select_x[-padding:, :].clone()
            #     select_x = torch.cat([select_x, pad_x], dim=0)
            #
            # select_x = select_x.reshape(B * split * split, -1, C)  # [(B h w hw ww), C]
            # query = query.expand(split * split, -1, -1)
            # select_x = torch.cat([query, select_x], dim=1)
            # select_x = self.forward_global_feature(select_x).flatten(0, 1)  # [(B h w) (hw ww), C]
            # select_x = select_x[:num_select]
            # out.scatter_(0, index.expand(-1, C), select_x)  # [(B h w hw ww), C]
            # out = rearrange(out, '(b h w x y) c -> b (h x w y) c', b=B, h=split, w=split, x=H // split, y=W // split)

            # ===============
            # H, W = x_size
            # out = x.clone().flatten(0, 1)
            # index = torch.nonzero(mask.flatten(0, 1))
            # select_x = x.flatten(0, 1)
            # select_x = select_x.gather(dim=0, index=index.expand(-1, C))#.unsqueeze(0)
            # ss = select_x.size(0)
            # group = int( min(2 ** round(math.log2(ss / 48)), 8)) if int( min(2 ** round(math.log2(ss / 48)), 8)) > 0 else 1
            # chunk_size = ss // group
            # # print(ss, group, chunk_size)
            # padding = chunk_size - ss % (chunk_size) if ss % chunk_size != 0 else 0
            # if padding:
            #     pad_x = select_x[-padding:, :].clone()
            #     select_x = torch.cat([select_x, pad_x], dim=0)
            # ss_group = select_x.size(0) // chunk_size
            # select_x = select_x.reshape(ss_group, chunk_size, -1)
            # query = query.expand(ss_group, -1, -1)
            # # print(select_x.shape, query.shape)
            # select_x = torch.cat([query, select_x], dim=1)
            # select_x = self.forward_global_feature(select_x).flatten(0, 1)
            # select_x = select_x[:ss, ...]
            # out.scatter_(0, index.expand(-1, C), select_x)
            # out = out.reshape(B, L, C)

            #############
            # H, W = x_size
            # # split = 1
            # split = min(2 ** round(math.log2(min(H * 1.0, W * 1.0) / 48)), 8)
            # x = rearrange(x, 'b (h x w y) c -> (b h w) (x y) c', h=split, x=H // split, w=split, y=W // split)
            # reshape_mask = rearrange(mask, 'b (h x w y) -> (b h w) (x y)', h=split, x=H // split, w=split, y=W // split)
            # out = x.clone().flatten(0, 1)
            # index = torch.nonzero(reshape_mask.flatten(0, 1))
            # select_x = x.flatten(0, 1)
            # select_x = select_x.gather(dim=0, index=index.expand(-1, C))
            #
            # num_select = select_x.size(0)
            # ss = select_x.size(0)
            # # padding = (split * split) - num_select % (split * split) if num_select % (split * split) != 0 else 0
            # #
            # # if padding and select_x.size(0) >= padding:
            # #     pad_x = select_x[-padding:, :].clone()
            # #     select_x = torch.cat([select_x, pad_x], dim=0)
            #
            # group = int(min(2 ** round(math.log2(ss / 48)), 8)) if int(
            #     min(2 ** round(math.log2(ss / 48)), 8)) > 0 else 1
            # chunk_size = ss // group
            # # print(ss, group, chunk_size)
            # padding = chunk_size - ss % (chunk_size) if ss % chunk_size != 0 else 0
            # if padding:
            #     pad_x = select_x[-padding:, :].clone()
            #     select_x = torch.cat([select_x, pad_x], dim=0)
            # ss_group = select_x.size(0) // chunk_size
            # select_x = select_x.reshape(ss_group, chunk_size, -1)
            # query = query.expand(ss_group, -1, -1)
            # # print(select_x.shape, query.shape)
            # select_x = torch.cat([query, select_x], dim=1)
            #
            # # select_x = select_x.reshape(B * split * split, -1, C)  # [(B h w hw ww), C]
            # # query = query.expand(split * split, -1, -1)
            # # select_x = torch.cat([query, select_x], dim=1)
            # select_x = self.forward_global_feature(select_x).flatten(0, 1)  # [(B h w) (hw ww), C]
            # select_x = select_x[:num_select]
            # out.scatter_(0, index.expand(-1, C), select_x)  # [(B h w hw ww), C]
            # out = rearrange(out, '(b h w x y) c -> b (h x w y) c', b=B, h=split, w=split, x=H // split, y=W // split)

        x_global = out * mask.unsqueeze(-1) + res * (1.0 - mask).unsqueeze(-1)
        return x_global

    def forward(self, x, x_size, global_mask=None, local_mask=None):
        H, W = x_size
        # local
        x_local = self.forward_local(x, (H, W), local_mask.squeeze(-1), global_mask)
        # global
        global_mask = rearrange(global_mask, 'b c h w -> b (h w) c').squeeze(-1)
        x_global = self.forward_global(x, x_size, global_mask)

        return x_local + x_global  # x_global #x_local # x_local + x_global

    def flops(self, x_size, local_mask, global_mask):
        flops = 0
        H, W = x_size
        ws = self.window_size

        ## 1. local
        hg, wg = H // ws, W // ws
        flops += (H * W * self.dim * self.dim * 3) # qkv

        local_mask = rearrange(local_mask, 'b c (h hs) (w ws) -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws).squeeze(-1)
        local_mask = local_mask.max(dim=-1).values

        flops += (local_mask.sum() * ws * ws * ws * ws * self.dim) * 2

        ## 2. global
        # W-MSA/SW-MSA
        flops += self.dim * self.query_num # norm
        flops += (H * W * self.dim * self.dim * 3) # qkv

        flops += (global_mask.sum() * self.query_num * self.dim // self.num_heads * self.num_heads) * 2
        flops += (H * W * self.dim * self.dim * 2)
        flops += (global_mask.sum() * self.query_num * self.dim // self.num_heads * self.num_heads) * 2

        return flops

# ========= Query Cluster ============
class LightViTAttentionHashCluster(nn.Module): # local select change
    def __init__(self, dim,  query_class=1, query_num=256, num_heads=8, window_size=8,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.query_num = query_num
        self.query_class = query_class
        self.window_size = window_size
        self.attn_area = window_size * window_size
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_local = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.querynorm = nn.Sequential(
            nn.BatchNorm2d(dim)
        )
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.prototype = nn.Parameter(torch.randn((self.query_class, query_num, dim)))
        # self.label = torch.zeros(query_num).cuda()
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.prototype, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def LSH(self, hash_buckets, n_hashes, x):
        # x: [N,H*W,C]
        N = x.shape[0]
        device = x.device

        # generate random rotation matrix
        rotations_shape = (1, x.shape[-1], n_hashes, hash_buckets // 2)  # [1,C,n_hashes,hash_buckets//2]
        random_rotations = torch.randn(rotations_shape, dtype=x.dtype, device=device).expand(N, -1, -1, -1)  # [N, C, n_hashes, hash_buckets//2]

        # locality sensitive hashing
        rotated_vecs = torch.einsum('btf,bfhi->bhti', x, random_rotations)  # [N, n_hashes, H*W, hash_buckets//2]
        rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)  # [N, n_hashes, H*W, hash_buckets]

        # get hash codes
        hash_codes = torch.argmax(rotated_vecs, dim=-1)  # [N,n_hashes,H*W]
        return hash_codes
        # # add offsets to avoid hash codes overlapping between hash rounds
        # offsets = torch.arange(n_hashes, device=device)
        # offsets = torch.reshape(offsets * hash_buckets, (1, -1, 1))
        # hash_codes = torch.reshape(hash_codes + offsets, (N, n_hashes, -1,))  # [N,n_hashes*H*W]
        #
        # return hash_codes

    def getKV(self, x, mask):
        b, n, c = x.shape
        hash_buckets = self.query_num
        n_hashes = 4
        # get assigned hash codes/bucket number
        hash_codes = self.LSH(hash_buckets, n_hashes, x)  # [b, n_hashes, H*W]

        hash_codes = hash_codes.detach()
        label = torch.zeros((b, n_hashes, n, self.query_num), device=x.device)
        # print(hash_codes.shape, label.shape)
        label.scatter_(dim=-1, index=hash_codes.unsqueeze(-1), value=1.) # [b, n_hashes, H*W, n_query]

        if self.training: # 训练过程排除不选的元素
            label = label * mask.reshape(b, 1, n, 1)

        label = label / (label.sum(2, keepdim=True) + 1e-8)
        merge_token = torch.einsum('b h n q, b n c -> b h q c', [label, x])
        return merge_token

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, H, N, HW = attn.size()
        attn_policy = policy.reshape(B, 1, 1, HW)

        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward_global_aggregation(self, q, k, v, attn_mask=None):
        """
        q: global tokens
        k: image tokens
        v: image tokens
        """
        B, _, N, _ = q.shape
        _, h, HW, c = k.shape
        q = q * self.scale
        if attn_mask is not None:
            attn = (q @ k.transpose(-2, -1)) * attn_mask.reshape(B, 1, 1, attn_mask.shape[1]).expand(-1, h, N, -1)
        else:
            attn = (q @ k.transpose(-2, -1))

        if attn_mask is not None:
            m = attn_mask.reshape(B, 1, 1, attn_mask.shape[1])
            attn = self.softmax_with_policy(attn, attn_mask) * m
        else:
            attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        t_attn = x
        return x, t_attn

    def forward_local(self, x, x_size, mask):
        """
        q: image tokens
        k: image tokens
        v: image tokens
        """
        if mask.sum() == 0.0: return x
        res = x
        B, N, C_ = x.shape
        ws = self.window_size
        H, W = x_size
        h  = self.num_heads
        hg, wg = H // ws, W // ws
        x = rearrange(x, 'b (h hs w ws) c -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws) # N, ws*ws, c
        if self.training:
            x = x * mask.reshape(-1, 1, 1)
        else:
            # x = x * mask.reshape(-1, 1, 1)
            out_ = x
            # x = x[mask==1.0, ...].contiguous()
            # out_ = x
            index = torch.nonzero(mask)
            x = torch.gather(x, dim=0, index=index[:, :, None].expand(-1, ws*ws, C_))

        qkv = self.qkv_local(x)
        q, k, v = rearrange(qkv, 'n w (d h c) -> d n h w c', d = 3, h = h).unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        pos_bias = self._get_relative_positional_bias()
        attn = (attn + pos_bias).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws * ws, -1)

        if not self.training:
            # out_ = torch.zeros(B * hg * wg, ws * ws, C_).cuda()
            # out_[mask == 1.0, ...] = out
            # out = out_

            out = out_.scatter(0, index[:, :, None].expand(-1, ws*ws, C_), out)

        # reverse
        mask = mask.reshape(-1, 1, 1).expand(-1, ws*ws, -1)
        out = window_reverse(out, ws, H, W).flatten(1, 2)
        mask = window_reverse(mask, ws, H, W).flatten(1, 2)
        x = out * mask + res * (1.0 - mask)
        return x

    def forward_global_broadcast(self, q, k, v):
        """
        q: image tokens
        k: global tokens
        v: global tokens
        """
        B, num_heads, N, _ = q.shape
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        return x

    def forward_global_feature(self, x, attn_mask=None):
        B, N, C = x.shape
        # NT = self.query_num
        #
        # # qkv
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)  # B, h, N, c
        #
        # # split img tokens & global tokens
        # q_img, k_img, v_img = q[:, :, NT:], k[:, :, NT:], v[:, :, NT:]
        # q_glb, _, _ = q[:, :, :NT], k[:, :, :NT], v[:, :, :NT]
        #
        # # global aggregation
        # x_glb, t_img = self.forward_global_aggregation(q_glb, k_img, v_img, attn_mask)
        x_glb = self.getKV(x, attn_mask) # [B, n_hash, n_query, C]
        x_glb = torch.sum(nn.functional.softmax(x_glb, dim=1) * x_glb, dim=1)
        k_glb, v_glb = self.kv_global(x_glb).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        x_img = self.forward_global_broadcast(q, k_glb, v_glb)
        return x_img

    def forward_global(self, x, mask=None):
        B, L, C = x.shape
        if mask.sum() == 0.0: return x

        # query = self.prototype.unsqueeze(1).expand((-1, B, -1, -1)).flatten(0, 1)  # 2B, q_n, dim
        # query = self.norm1(query)
        # query = self.getKV(x, mask) # [B, n_hash, L, query, C]
        if self.training:
            # out = torch.cat([query, x], dim=1)

            x = x * mask.unsqueeze(-1)
            out = self.forward_global_feature(x, attn_mask=mask)
        else:
            # x = x.flatten(0, 1)  # (N, dim)
            # mask = mask.flatten(0, 1).squeeze()
            # select_x = x[mask == 1.0, :].unsqueeze(0)
            # select_x = torch.cat([query, select_x], dim=1)
            # select_x = self.forward_global_feature(select_x)
            # out = torch.zeros_like(x)
            # out[mask == 1.0, :] = select_x
            # out = out.reshape(B, L, C)
            out = x.clone().flatten(0, 1)
            index = torch.nonzero(mask.flatten(0, 1))
            select_x = x.flatten(0, 1)
            select_x = select_x.gather(dim=0, index=index.expand(-1, C)).unsqueeze(0)
            # select_x = torch.cat([query, select_x], dim=1)
            select_x = self.forward_global_feature(select_x).flatten(0, 1)

            out.scatter_(0, index.expand(-1, C), select_x)
            out = out.reshape(B, L, C)


        x_global = out * mask.unsqueeze(-1) + x * (1.0 - mask).unsqueeze(-1)
        return x_global

    def forward(self, x, x_size, global_mask=None, local_mask=None):
        H, W = x_size
        # local
        x_local = self.forward_local(x, (H, W), local_mask.squeeze(-1))
        # global
        global_mask = rearrange(global_mask, 'b c h w -> b (h w) c').squeeze(-1)
        x_global = self.forward_global(x, global_mask)

        return  x_local + x_global #x_global #x_local # x_local + x_global
#=====================================

# ======== Query Refinement ==========
class LightViTAttentionRefiment(nn.Module):
    def __init__(self, dim, query_class=1, query_num=256, num_heads=8, window_size=8,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.query_num = query_num
        self.query_class = query_class
        self.window_size = window_size
        self.attn_area = window_size * window_size
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_local = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.qkv_refinement = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.querynorm = nn.Sequential(
            nn.BatchNorm2d(dim)
        )
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.prototype = nn.Parameter(torch.randn((self.query_class, query_num, dim)))

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.prototype, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, H, N, HW = attn.size()
        attn_policy = policy.reshape(B, 1, 1, HW)

        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward_global_aggregation(self, q, k, v, attn_mask=None):
        """
        q: global tokens
        k: image tokens
        v: image tokens
        """
        B, _, N, _ = q.shape
        _, h, HW, c = k.shape
        q = q * self.scale
        if attn_mask is not None:
            attn = (q @ k.transpose(-2, -1)) * attn_mask.reshape(B, 1, 1, attn_mask.shape[1]).expand(-1, h, N, -1)
        else:
            attn = (q @ k.transpose(-2, -1))

        if attn_mask is not None:
            m = attn_mask.reshape(B, 1, 1, attn_mask.shape[1])
            attn = self.softmax_with_policy(attn, attn_mask) * m
        else:
            attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        t_attn = x
        return x, t_attn

    def forward_local(self, x, x_size, mask):
        """
        q: image tokens
        k: image tokens
        v: image tokens
        """
        if mask.sum() == 0.0: return x
        res = x
        B, N, C_ = x.shape
        ws = self.window_size
        H, W = x_size
        h = self.num_heads
        hg, wg = H // ws, W // ws
        x = rearrange(x, 'b (h hs w ws) c -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws)  # N, ws*ws, c
        if self.training:
            x = x * mask.reshape(-1, 1, 1)
        else:
            # x = x * mask.reshape(-1, 1, 1)
            out_ = x
            # x = x[mask==1.0, ...].contiguous()
            # out_ = x
            index = torch.nonzero(mask)
            x = torch.gather(x, dim=0, index=index[:, :, None].expand(-1, ws * ws, C_))

        qkv = self.qkv_local(x)
        q, k, v = rearrange(qkv, 'n w (d h c) -> d n h w c', d=3, h=h).unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        pos_bias = self._get_relative_positional_bias()
        attn = (attn + pos_bias).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws * ws, -1)

        if not self.training:
            # out_ = torch.zeros(B * hg * wg, ws * ws, C_).cuda()
            # out_[mask == 1.0, ...] = out
            # out = out_

            out = out_.scatter(0, index[:, :, None].expand(-1, ws * ws, C_), out)

        # reverse
        mask = mask.reshape(-1, 1, 1).expand(-1, ws * ws, -1)
        out = window_reverse(out, ws, H, W).flatten(1, 2)
        mask = window_reverse(mask, ws, H, W).flatten(1, 2)
        x = out * mask + res * (1.0 - mask)
        return x

    def forward_global_broadcast(self, q, k, v):
        """
        q: image tokens
        k: global tokens
        v: global tokens
        """
        B, num_heads, N, _ = q.shape
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        return x

    def forward_global_feature(self, x, attn_mask=None):
        B, N, C = x.shape
        NT = self.query_num

        # qkv
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)  # B, h, N, c

        # split img tokens & global tokens
        q_img, k_img, v_img = q[:, :, NT:], k[:, :, NT:], v[:, :, NT:]
        q_glb, _, _ = q[:, :, :NT], k[:, :, :NT], v[:, :, :NT]

        # global aggregation
        x_glb, t_img = self.forward_global_aggregation(q_glb, k_img, v_img, attn_mask)

        # refinement
        q_glb, v_glb, k_glb = self.qkv_refinement(x_glb).view(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        x_glb = self.forward_global_broadcast(q_glb, v_glb, k_glb)

        #
        k_glb, v_glb = self.kv_global(x_glb).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        x_img = self.forward_global_broadcast(q_img, k_glb, v_glb)
        return x_img

    def forward_global(self, x, mask=None):
        B, L, C = x.shape
        if mask.sum() == 0.0: return x
        query = self.prototype.unsqueeze(1).expand((-1, B, -1, -1)).flatten(0, 1)  # 2B, q_n, dim
        query = self.norm1(query)
        if self.training:
            out = torch.cat([query, x], dim=1)
            out = self.forward_global_feature(out, attn_mask=mask)
        else:
            # x = x.flatten(0, 1)  # (N, dim)
            # mask = mask.flatten(0, 1).squeeze()
            # select_x = x[mask == 1.0, :].unsqueeze(0)
            # select_x = torch.cat([query, select_x], dim=1)
            # select_x = self.forward_global_feature(select_x)
            # out = torch.zeros_like(x)
            # out[mask == 1.0, :] = select_x
            # out = out.reshape(B, L, C)

            out = x.clone().flatten(0, 1)
            index = torch.nonzero(mask.flatten(0, 1))
            select_x = x.flatten(0, 1)
            select_x = select_x.gather(dim=0, index=index.expand(-1, C)).unsqueeze(0)
            select_x = torch.cat([query, select_x], dim=1)
            select_x = self.forward_global_feature(select_x).flatten(0, 1)

            out.scatter_(0, index.expand(-1, C), select_x)
            out = out.reshape(B, L, C)

        x_global = out * mask.unsqueeze(-1) + x * (1.0 - mask).unsqueeze(-1)
        return x_global

    def forward(self, x, x_size, global_mask=None, local_mask=None):
        H, W = x_size
        # local
        x_local = self.forward_local(x, (H, W), local_mask.squeeze(-1))
        # global
        global_mask = rearrange(global_mask, 'b c h w -> b (h w) c').squeeze(-1)
        x_global = self.forward_global(x, global_mask)

        return x_local + x_global  # x_global #x_local # x_local + x_global

    def flops(self, x_size, local_mask, global_mask):
        flops = 0
        H, W = x_size
        ws = self.window_size

        ## 1. local
        hg, wg = H // ws, W // ws
        flops += (H * W * self.dim * self.dim * 3) # qkv

        local_mask = rearrange(local_mask, 'b c (h hs) (w ws) -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws).squeeze(-1)
        local_mask = local_mask.max(dim=-1).values

        flops += (local_mask.sum() * ws * ws * ws * ws * self.dim) * 2

        ## 2. global
        # W-MSA/SW-MSA
        flops += self.dim * self.query_num # norm
        flops += (H * W * self.dim * self.dim * 3) # qkv
        flops += (global_mask.sum() * self.query_num * self.dim // self.num_heads * self.num_heads) * 2
        flops += (H * W * self.dim * self.dim * 3)
        flops += (self.query_num * self.query_num * self.dim // self.num_heads * self.num_heads) * 2
        flops += (H * W * self.dim * self.dim * 2)
        flops += (global_mask.sum() * self.query_num * self.dim // self.num_heads * self.num_heads) * 2

        return flops
class LightViTAttentionRefimentV2(nn.Module):
    def __init__(self, dim, query_class=1, query_num=256, num_heads=8, window_size=8,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm, shift=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        self.query_num = query_num
        self.query_class = query_class
        self.window_size = window_size
        self.attn_area = window_size * window_size
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv_refinement = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_local = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.prototype = nn.Parameter(torch.randn((self.query_class, query_num, dim)))

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.prototype, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, H, N, HW = attn.size()
        attn_policy = policy.reshape(B, 1, 1, HW)

        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward_global_aggregation(self, q, k, v, attn_mask=None):
        """
        q: global tokens
        k: image tokens
        v: image tokens
        """
        B, _, N, _ = q.shape
        _, h, HW, c = k.shape
        q = q * self.scale
        if attn_mask is not None:
            attn = (q @ k.transpose(-2, -1)) * attn_mask.reshape(B, 1, 1, attn_mask.shape[1]).expand(-1, h, N, -1)
            m = attn_mask.reshape(B, 1, 1, attn_mask.shape[1])
            attn = self.softmax_with_policy(attn, attn_mask) * m
        else:
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x

    def forward_local(self, x, x_size, mask):
        """
        q: image tokens
        k: image tokens
        v: image tokens
        """
        if mask.sum() == 0.0: return x
        res = x
        B, N, C_ = x.shape
        ws = self.window_size
        H, W = x_size
        h = self.num_heads
        hg, wg = H // ws, W // ws
        x = rearrange(x, 'b (h hs w ws) c -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws)  # N, ws*ws, c
        if self.training:
            x = x * mask.reshape(-1, 1, 1)
        else:
            # x = x * mask.reshape(-1, 1, 1)
            out_ = x.clone()
            # x = x[mask==1.0, ...].contiguous()
            # out_ = x
            index = torch.nonzero(mask)
            x = torch.gather(x, dim=0, index=index[:, :, None].expand(-1, ws * ws, C_))

        qkv = self.qkv_local(x)
        q, k, v = rearrange(qkv, 'n w (d h c) -> d n h w c', d=3, h=h).unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        pos_bias = self._get_relative_positional_bias()
        attn = (attn + pos_bias).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws * ws, -1)

        if not self.training:
            # out_ = torch.zeros(B * hg * wg, ws * ws, C_).cuda()
            # out_[mask == 1.0, ...] = out
            # out = out_

            out = out_.scatter(0, index[:, :, None].expand(-1, ws * ws, C_), out)

        # reverse
        mask = mask.reshape(-1, 1, 1).expand(-1, ws * ws, -1)
        out = window_reverse(out, ws, H, W).flatten(1, 2)
        mask = window_reverse(mask, ws, H, W).flatten(1, 2)
        x = out * mask + res * (1.0 - mask)
        return x

    def forward_global_broadcast(self, q, k, v):
        """
        q: image tokens
        k: global tokens
        v: global tokens
        """
        B, num_heads, N, _ = q.shape
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        return x

    def forward_global_feature(self, x, attn_mask=None):
        B, N, C = x.shape
        NT = self.query_num

        # qkv
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)  # B, h, N, c

        # split img tokens & global tokens
        q_img, k_img, v_img = q[:, :, NT:], k[:, :, NT:], v[:, :, NT:]
        q_glb, _, _ = q[:, :, :NT], k[:, :, :NT], v[:, :, :NT]

        # global aggregation
        x_glb = self.forward_global_aggregation(q_glb, k_img, v_img, attn_mask)
        q_glb, v_glb, k_glb = self.qkv_refinement(x_glb).view(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        x_glb = self.forward_global_broadcast(q_glb, v_glb, k_glb)

        k_glb, v_glb = self.kv_global(x_glb).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        x_img = self.forward_global_broadcast(q_img, k_glb, v_glb)
        return x_img

    def forward_global(self, x, x_size=None, mask=None):
        B, L, C = x.shape
        res = x
        if mask.sum() == 0.0: return x
        query = self.prototype.unsqueeze(1).expand((-1, B, -1, -1)).flatten(0, 1)  # 2B, q_n, dim
        query = self.norm1(query)
        if self.training:
            out = torch.cat([query, x], dim=1)
            out = self.forward_global_feature(out, attn_mask=mask)
        else:
            # x = x.flatten(0, 1)  # (N, dim)
            # mask = mask.flatten(0, 1).squeeze()
            # select_x = x[mask == 1.0, :].unsqueeze(0)
            # select_x = torch.cat([query, select_x], dim=1)
            # select_x = self.forward_global_feature(select_x)
            # out = torch.zeros_like(x)
            # out[mask == 1.0, :] = select_x
            # out = out.reshape(B, L, C)

            out = x.clone().flatten(0, 1)
            index = torch.nonzero(mask.flatten(0, 1))
            select_x = x.flatten(0, 1)
            select_x = select_x.gather(dim=0, index=index.expand(-1, C)).unsqueeze(0)
            select_x = torch.cat([query, select_x], dim=1)
            select_x = self.forward_global_feature(select_x).flatten(0, 1)

            out.scatter_(0, index.expand(-1, C), select_x)
            out = out.reshape(B, L, C)

            # H, W = x_size
            # split = min(2 ** round(math.log2(min(H * 1.0, W * 1.0) / 48)), 8)
            # chunk_size = 48
            # # print(split)
            # # hw = H // split
            # # ww = W // split
            # x = rearrange(x, 'b (h x w y) c -> (b h w) (x y) c', h=split, x=H // split, w=split, y=W // split)
            # reshape_mask = rearrange(mask, 'b (h x w y) -> (b h w) (x y)', h=split, x=H // split, w=split, y=W // split)
            # out = x.clone().flatten(0, 1)
            # index = torch.nonzero(reshape_mask.flatten(0, 1))
            # select_x = x.flatten(0, 1)
            # select_x = select_x.gather(dim=0, index=index.expand(-1, C))
            # num_select = select_x.size(0)
            # padding = (split * split) - num_select % (split * split) if num_select % (split * split) != 0 else 0
            # # padding = chunk_size - num_select % chunk_size if num_select % chunk_size != 0 else 0
            #
            # if padding and select_x.size(0) >= padding:
            #     pad_x = select_x[-padding:, :].clone()
            #     # pad_x = torch.ones((padding, select_x.size(-1)), device=select_x.device) * -1000.0
            #     select_x = torch.cat([select_x, pad_x], dim=0)
            #
            # # group = select_x.size(0) // chunk_size
            # group = max(select_x.size(0) // (split * split), 1)
            # select_x = select_x.reshape(B * group, -1, C)  # [(B h w hw ww), C]
            # query = query.expand(B * group, -1, -1)
            #
            # select_x = torch.cat([query, select_x], dim=1)
            # select_x = self.forward_global_feature(select_x).flatten(0, 1)  # [(B h w) (hw ww), C]
            # select_x = select_x[:num_select]
            # out.scatter_(0, index.expand(-1, C), select_x)  # [(B h w hw ww), C]
            # out = rearrange(out, '(b h w x y) c -> b (h x w y) c', b=B, h=split, w=split, x=H // split, y=W // split)
            # print(out.shape, mask.shape)

        x_global = out * mask.unsqueeze(-1) + res * (1.0 - mask).unsqueeze(-1)
        return x_global

    def forward(self, x, x_size, global_mask=None, local_mask=None):
        H, W = x_size
        # local
        x_local = self.forward_local(x, (H, W), local_mask.squeeze(-1))
        # global
        global_mask = rearrange(global_mask, 'b c h w -> b (h w) c').squeeze(-1)
        x_global = self.forward_global(x, x_size, global_mask)

        return x_local + x_global  # x_global #x_local # x_local + x_global

    def flops(self, x_size, local_mask, global_mask):
        flops = 0
        H, W = x_size
        ws = self.window_size

        ## 1. local
        hg, wg = H // ws, W // ws
        flops += (H * W * self.dim * self.dim * 3) # qkv

        local_mask = rearrange(local_mask, 'b c (h hs) (w ws) -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws).squeeze(-1)
        local_mask = local_mask.max(dim=-1).values

        flops += (local_mask.sum() * ws * ws * ws * ws * self.dim) * 2

        ## 2. global
        # W-MSA/SW-MSA
        flops += self.dim * self.query_num # norm
        flops += (H * W * self.dim * self.dim * 3) # qkv

        flops += (global_mask.sum() * self.query_num * self.dim // self.num_heads * self.num_heads) * 2
        flops += (H * W * self.dim * self.dim * 2)
        flops += (global_mask.sum() * self.query_num * self.dim // self.num_heads * self.num_heads) * 2

        return flops

class LightViTAttentionInferenceQueryGroup(nn.Module):
    def __init__(self, dim, query_class=1, query_num=256, num_heads=8, window_size=8,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        self.query_num = query_num
        self.query_class = query_class
        self.window_size = window_size
        self.attn_area = window_size * window_size
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_local = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.querynorm = nn.Sequential(
            nn.BatchNorm2d(dim)
        )
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.prototype = nn.Parameter(torch.randn((self.query_class, query_num, dim)))

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.prototype, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, H, N, HW = attn.size()
        attn_policy = policy.reshape(B, 1, 1, HW)

        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward_global_aggregation(self, q, k, v, attn_mask=None):
        """
        q: global tokens
        k: image tokens
        v: image tokens
        """
        B, _, N, _ = q.shape
        _, h, HW, c = k.shape
        q = q * self.scale
        if attn_mask is not None:
            attn = (q @ k.transpose(-2, -1)) * attn_mask.reshape(B, 1, 1, attn_mask.shape[1]).expand(-1, h, N, -1)
        else:
            attn = (q @ k.transpose(-2, -1))

        if attn_mask is not None:
            m = attn_mask.reshape(B, 1, 1, attn_mask.shape[1])
            attn = self.softmax_with_policy(attn, attn_mask) * m
        else:
            attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        t_attn = x
        return x, t_attn

    def forward_local(self, x, x_size, mask):
        """
        q: image tokens
        k: image tokens
        v: image tokens
        """
        if mask.sum() == 0.0: return x
        res = x
        B, N, C_ = x.shape
        ws = self.window_size
        H, W = x_size
        h = self.num_heads
        hg, wg = H // ws, W // ws
        x = rearrange(x, 'b (h hs w ws) c -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws)  # N, ws*ws, c
        if self.training:
            x = x * mask.reshape(-1, 1, 1)
        else:
            # x = x * mask.reshape(-1, 1, 1)
            out_ = x
            # x = x[mask==1.0, ...].contiguous()
            # out_ = x
            index = torch.nonzero(mask)
            x = torch.gather(x, dim=0, index=index[:, :, None].expand(-1, ws * ws, C_))

        qkv = self.qkv_local(x)
        q, k, v = rearrange(qkv, 'n w (d h c) -> d n h w c', d=3, h=h).unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        pos_bias = self._get_relative_positional_bias()
        attn = (attn + pos_bias).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws * ws, -1)

        if not self.training:
            # out_ = torch.zeros(B * hg * wg, ws * ws, C_).cuda()
            # out_[mask == 1.0, ...] = out
            # out = out_

            out = out_.scatter(0, index[:, :, None].expand(-1, ws * ws, C_), out)

        # reverse
        mask = mask.reshape(-1, 1, 1).expand(-1, ws * ws, -1)
        out = window_reverse(out, ws, H, W).flatten(1, 2)
        mask = window_reverse(mask, ws, H, W).flatten(1, 2)
        x = out * mask + res * (1.0 - mask)
        return x

    def forward_global_broadcast(self, q, k, v):
        """
        q: image tokens
        k: global tokens
        v: global tokens
        """
        B, num_heads, N, _ = q.shape
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        return x

    def forward_global_feature(self, x, attn_mask=None):
        B, N, C = x.shape
        NT = self.query_num

        # qkv
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)  # B, h, N, c

        # split img tokens & global tokens
        q_img, k_img, v_img = q[:, :, NT:], k[:, :, NT:], v[:, :, NT:]
        q_glb, _, _ = q[:, :, :NT], k[:, :, :NT], v[:, :, :NT]

        # global aggregation
        x_glb, t_img = self.forward_global_aggregation(q_glb, k_img, v_img, attn_mask)
        k_glb, v_glb = self.kv_global(x_glb).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                         4).unbind(0)
        x_img = self.forward_global_broadcast(q_img, k_glb, v_glb)
        return x_img

    def forward_global(self, x, x_size=None, mask=None):
        B, L, C = x.shape
        if mask.sum() == 0.0: return x
        res = x
        query = self.prototype  # 1, q_n, dim
        query = self.norm1(query)
        if self.training:
            query = query.expand(B, -1, -1)
            out = torch.cat([query, x], dim=1)
            out = self.forward_global_feature(out, attn_mask=mask)
        else:
            # x = x.flatten(0, 1)  # (N, dim)
            # mask = mask.flatten(0, 1).squeeze()
            # select_x = x[mask == 1.0, :].unsqueeze(0)
            # select_x = torch.cat([query, select_x], dim=1)
            # select_x = self.forward_global_feature(select_x)
            # out = torch.zeros_like(x)
            # out[mask == 1.0, :] = select_x
            # out = out.reshape(B, L, C)


            H, W = x_size
            split = 4
            # hw = H // split
            # ww = W // split
            x = rearrange(x, 'b (h x w y) c -> (b h w) (x y) c', h = split, x = H // split, w = split, y = W // split)
            reshape_mask = rearrange(mask, 'b (h x w y) -> (b h w) (x y)', h = split, x = H // split, w = split, y = W // split)
            out = x.clone().flatten(0, 1)
            index = torch.nonzero(reshape_mask.flatten(0, 1))
            select_x = x.flatten(0, 1)
            select_x = select_x.gather(dim=0, index=index.expand(-1, C))
            num_select = select_x.size(0)
            padding = (split * split) - num_select % (split * split) if num_select % (split * split) != 0 else 0

            if padding:
                pad_x = select_x[-padding:, :].clone()
                select_x = torch.cat([select_x, pad_x], dim=0)

            select_x = select_x.reshape(B * split * split, -1, C) # [(B h w hw ww), C]
            query = query.expand(split * split, -1, -1)
            select_x = torch.cat([query, select_x], dim=1)
            select_x = self.forward_global_feature(select_x).flatten(0, 1) # [(B h w) (hw ww), C]
            select_x = select_x[:num_select]
            out.scatter_(0, index.expand(-1, C), select_x) # [(B h w hw ww), C]
            out =  rearrange(out, '(b h w x y) c -> b (h x w y) c', b = B, h = split, w = split, x = H // split, y = W // split)

        x_global = out * mask.unsqueeze(-1) + res * (1.0 - mask).unsqueeze(-1)
        return x_global

    def forward(self, x, x_size, global_mask=None, local_mask=None):
        H, W = x_size
        # local
        x_local = self.forward_local(x, (H, W), local_mask.squeeze(-1))
        # global
        global_mask = rearrange(global_mask, 'b c h w -> b (h w) c').squeeze(-1)
        x_global = self.forward_global(x, (H, W), global_mask)

        return x_local + x_global  # x_global #x_local # x_local + x_global

    def flops(self, x_size, local_mask, global_mask):
        flops = 0
        H, W = x_size
        ws = self.window_size

        ## 1. local
        hg, wg = H // ws, W // ws
        flops += (H * W * self.dim * self.dim * 3) # qkv

        local_mask = rearrange(local_mask, 'b c (h hs) (w ws) -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws).squeeze(-1)
        local_mask = local_mask.max(dim=-1).values

        flops += (local_mask.sum() * ws * ws * ws * ws * self.dim) * 2

        ## 2. global
        # W-MSA/SW-MSA
        flops += self.dim * self.query_num # norm
        flops += (H * W * self.dim * self.dim * 3) # qkv

        flops += (global_mask.sum() * self.query_num * self.dim // self.num_heads * self.num_heads) * 2
        flops += (H * W * self.dim * self.dim * 2)
        flops += (global_mask.sum() * self.query_num * self.dim // self.num_heads * self.num_heads) * 2

        return flops

class LightViTAttentionInferenceLocalQuery(nn.Module):
    def __init__(self, dim, query_class=1, query_num=256, num_heads=8, window_size=8,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        self.query_num = query_num
        self.query_class = query_class
        self.window_size = window_size
        self.attn_area = window_size * window_size
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_local = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.querynorm = nn.Sequential(
            nn.BatchNorm2d(dim)
        )
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.prototype = nn.Parameter(torch.randn((self.query_class, query_num, dim)))
        self.local_prototype = nn.Parameter(torch.randn((self.query_class, query_num, dim)))
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.prototype, std=.02)
        trunc_normal_(self.local_prototype, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, H, N, HW = attn.size()
        attn_policy = policy.reshape(B, 1, 1, HW)

        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward_global_aggregation(self, q, k, v, attn_mask=None):
        """
        q: global tokens
        k: image tokens
        v: image tokens
        """
        B, _, N, _ = q.shape
        _, h, HW, c = k.shape
        q = q * self.scale
        if attn_mask is not None:
            attn = (q @ k.transpose(-2, -1)) * attn_mask.reshape(B, 1, 1, attn_mask.shape[1]).expand(-1, h, N, -1)
        else:
            attn = (q @ k.transpose(-2, -1))

        if attn_mask is not None:
            m = attn_mask.reshape(B, 1, 1, attn_mask.shape[1])
            attn = self.softmax_with_policy(attn, attn_mask) * m
        else:
            attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        t_attn = x
        return x, t_attn

    def forward_local(self, x, x_size, mask):
        """
        q: image tokens
        k: image tokens
        v: image tokens
        """
        if mask.sum() == 0.0: return x
        res = x
        B, N, C_ = x.shape
        ws = self.window_size
        H, W = x_size
        h = self.num_heads
        hg, wg = H // ws, W // ws
        x = rearrange(x, 'b (h hs w ws) c -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws)  # N, ws*ws, c
        if self.training:
            x = x * mask.reshape(-1, 1, 1)
        else:
            # x = x * mask.reshape(-1, 1, 1)
            out_ = x
            # x = x[mask==1.0, ...].contiguous()
            # out_ = x
            index = torch.nonzero(mask)
            x = torch.gather(x, dim=0, index=index[:, :, None].expand(-1, ws * ws, C_))

        qkv = self.qkv_local(x)
        q, k, v = rearrange(qkv, 'n w (d h c) -> d n h w c', d=3, h=h).unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        pos_bias = self._get_relative_positional_bias()
        attn = (attn + pos_bias).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws * ws, -1)

        if not self.training:
            # out_ = torch.zeros(B * hg * wg, ws * ws, C_).cuda()
            # out_[mask == 1.0, ...] = out
            # out = out_

            out = out_.scatter(0, index[:, :, None].expand(-1, ws * ws, C_), out)

        # reverse
        mask = mask.reshape(-1, 1, 1).expand(-1, ws * ws, -1)
        out = window_reverse(out, ws, H, W).flatten(1, 2)
        mask = window_reverse(mask, ws, H, W).flatten(1, 2)
        x = out * mask + res * (1.0 - mask)
        return x

    def pad_feature_size(self, x, ws):
        _, _, h, w = x.size()
        mod_pad_h = (ws - h % ws) % ws
        mod_pad_w = (ws - w % ws) % ws
        x = torch.nn.functional.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_local_query(self, x, x_size, mask):
        B, L, C = x.shape
        if mask.sum() == 0.0: return x
        res = x
        H, W = x_size
        ws = 24 if not self.training else 48#self.window_size

        query = self.prototype  # 1, q_n, dim
        query = self.norm1(query)
        x = rearrange(x, 'b (h w) c -> b c h w ', h = H, w = W)
        # print(mask.shape)
        reshape_mask = rearrange(mask, 'b (h w) -> b h w ', h=H, w=W).unsqueeze(1)
        x = self.pad_feature_size(x, ws)
        reshape_mask = self.pad_feature_size(reshape_mask, ws)
        HH, WW = x.shape[2:]
        nh = HH // ws
        nw = WW // ws

        x = rearrange(x, 'b c (h x) (w y) -> (b h w) (x y) c', h=nh, x=ws, w=nw, y=ws)
        reshape_mask = rearrange(reshape_mask, 'b c (h x) (w y) -> (b h w) (x y) c', h=nh, x=ws, w=nw, y=ws).squeeze(-1)
        # x = rearrange(x, 'b (h x w y) c -> (b h w) (x y) c', h = nh, x= ws, w = nw, y = ws)
        # reshape_mask = rearrange(mask, 'b (h x w y) -> (b h w) (x y)', h = nh, x= ws, w = nw, y = ws)
        nb, nl, _ = x.shape

        if self.training:

            query = query.expand(nb, -1, -1)
            out = torch.cat([query, x], dim=1)
            out = self.forward_global_feature(out, attn_mask=reshape_mask)
            out = rearrange(out, '(b h w) (x y) c -> b c (h x) (w y)', b=B, h=nh, w=nw, x=ws, y=ws)
            out = out[:, :, :H, :W]
            out = rearrange(out, 'b c h w -> b (h w) c')

        else:
            out = x.clone().flatten(0, 1)
            index = torch.nonzero(reshape_mask.flatten(0, 1))
            select_x = x.flatten(0, 1)
            select_x = select_x.gather(dim=0, index=index.expand(-1, C))
            num_select = select_x.size(0)
            hws = ws // 2
            padding = (hws * hws) - num_select % (hws * hws) if num_select % (hws * hws) != 0 else 0

            if padding:
                pad_x = select_x[-padding:, :].clone()
                select_x = torch.cat([select_x, pad_x], dim=0)

            select_x = select_x.reshape(-1, hws * hws, C) # [(B h w hw ww), C]
            nq = (num_select + padding) // (hws * hws)
            query = query.expand(nq, -1, -1)
            select_x = torch.cat([query, select_x], dim=1)
            select_x = self.forward_global_feature(select_x).flatten(0, 1) # [(B h w) (hw ww), C]
            select_x = select_x[:num_select]
            out.scatter_(0, index.expand(-1, C), select_x) # [(B h w hw ww), C]
            out = rearrange(out, '(b h w x y) c -> b c (h x) (w y)', b=B, h=nh, w=nw, x=ws, y=ws)
            out = out[:, :, :H, :W]
            out = rearrange(out, 'b c h w -> b (h w) c')
            # out = rearrange(out, '(b h w x y) c -> b (h x w y) c', b = B, h = nh, w = nw, x = ws, y = ws)

        x_global = out * mask.unsqueeze(-1) + res * (1.0 - mask).unsqueeze(-1)
        return x_global

    def forward_global_broadcast(self, q, k, v):
        """
        q: image tokens
        k: global tokens
        v: global tokens
        """
        B, num_heads, N, _ = q.shape
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        return x

    def forward_global_feature(self, x, attn_mask=None):
        B, N, C = x.shape
        NT = self.query_num

        # qkv
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)  # B, h, N, c

        # split img tokens & global tokens
        q_img, k_img, v_img = q[:, :, NT:], k[:, :, NT:], v[:, :, NT:]
        q_glb, _, _ = q[:, :, :NT], k[:, :, :NT], v[:, :, :NT]

        # global aggregation
        x_glb, t_img = self.forward_global_aggregation(q_glb, k_img, v_img, attn_mask)
        k_glb, v_glb = self.kv_global(x_glb).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                         4).unbind(0)
        x_img = self.forward_global_broadcast(q_img, k_glb, v_glb)
        return x_img

    def forward_global(self, x, x_size=None, mask=None):
        B, L, C = x.shape
        if mask.sum() == 0.0: return x
        res = x
        query = self.prototype  # 1, q_n, dim
        query = self.norm1(query)
        if self.training:
            query = query.expand(B, -1, -1)
            out = torch.cat([query, x], dim=1)
            out = self.forward_global_feature(out, attn_mask=mask)
        else:
            x = x.flatten(0, 1)  # (N, dim)
            mask = mask.flatten(0, 1).squeeze()
            select_x = x[mask == 1.0, :].unsqueeze(0)
            select_x = torch.cat([query, select_x], dim=1)
            select_x = self.forward_global_feature(select_x)
            out = torch.zeros_like(x)
            out[mask == 1.0, :] = select_x
            out = out.reshape(B, L, C)


            # H, W = x_size
            # split = 4
            # # hw = H // split
            # # ww = W // split
            # x = rearrange(x, 'b (h x w y) c -> (b h w) (x y) c', h = split, x = H // split, w = split, y = W // split)
            # reshape_mask = rearrange(mask, 'b (h x w y) -> (b h w) (x y)', h = split, x = H // split, w = split, y = W // split)
            # out = x.clone().flatten(0, 1)
            # index = torch.nonzero(reshape_mask.flatten(0, 1))
            # select_x = x.flatten(0, 1)
            # select_x = select_x.gather(dim=0, index=index.expand(-1, C))
            # num_select = select_x.size(0)
            # padding = (split * split) - num_select % (split * split) if num_select % (split * split) != 0 else 0
            #
            # if padding:
            #     pad_x = select_x[-padding:, :].clone()
            #     select_x = torch.cat([select_x, pad_x], dim=0)
            #
            # select_x = select_x.reshape(B * split * split, -1, C) # [(B h w hw ww), C]
            # query = query.expand(split * split, -1, -1)
            # select_x = torch.cat([query, select_x], dim=1)
            # select_x = self.forward_global_feature(select_x).flatten(0, 1) # [(B h w) (hw ww), C]
            # select_x = select_x[:num_select]
            # out.scatter_(0, index.expand(-1, C), select_x) # [(B h w hw ww), C]
            # out =  rearrange(out, '(b h w x y) c -> b (h x w y) c', b = B, h = split, w = split, x = H // split, y = W // split)

        x_global = out * mask.unsqueeze(-1) + res * (1.0 - mask).unsqueeze(-1)
        return x_global

    def forward(self, x, x_size, global_mask=None, local_mask=None):
        H, W = x_size
        # local
        x_local = self.forward_local(x, (H, W), local_mask.squeeze(-1))

        # global
        # global query
        global_mask = rearrange(global_mask, 'b c h w -> b (h w) c').squeeze(-1)
        x_local_query = self.forward_local_query(x, (H, W), global_mask)
        # x_global = self.forward_global(x, (H, W), global_mask)

        return x_local + x_local_query #x_local_query + x_global  # x_global #x_local # x_local + x_global

    def flops(self, x_size, local_mask, global_mask):
        flops = 0
        H, W = x_size
        ws = self.window_size

        ## 1. local
        hg, wg = H // ws, W // ws
        flops += (H * W * self.dim * self.dim * 3) # qkv

        local_mask = rearrange(local_mask, 'b c (h hs) (w ws) -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws).squeeze(-1)
        local_mask = local_mask.max(dim=-1).values

        flops += (local_mask.sum() * ws * ws * ws * ws * self.dim) * 2

        ## 2. global
        # W-MSA/SW-MSA
        flops += self.dim * self.query_num # norm
        flops += (H * W * self.dim * self.dim * 3) # qkv

        flops += (global_mask.sum() * self.query_num * self.dim // self.num_heads * self.num_heads) * 2
        flops += (H * W * self.dim * self.dim * 2)
        flops += (global_mask.sum() * self.query_num * self.dim // self.num_heads * self.num_heads) * 2

        return flops

class LightViTAttentionExclude(nn.Module):
    def __init__(self, dim, query_class=1, query_num=256, num_heads=8, window_size=8,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        self.query_num = query_num
        self.query_class = query_class
        self.window_size = window_size
        self.attn_area = window_size * window_size
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_local = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)

        self.prototype = nn.Parameter(torch.randn((self.query_class, query_num, dim)))

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.prototype, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, H, N, HW = attn.size()
        attn_policy = policy.reshape(B, 1, 1, HW)

        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward_global_aggregation(self, q, k, v, attn_mask=None):
        """
        q: global tokens
        k: image tokens
        v: image tokens
        """
        B, _, N, _ = q.shape
        _, h, HW, c = k.shape
        q = q * self.scale
        if attn_mask is not None:
            attn = (q @ k.transpose(-2, -1)) * attn_mask.reshape(B, 1, 1, attn_mask.shape[1]).expand(-1, h, N, -1)
            m = attn_mask.reshape(B, 1, 1, attn_mask.shape[1])
            attn = self.softmax_with_policy(attn, attn_mask) * m
        else:
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x

    def forward_local(self, x, x_size, mask):
        """
        q: image tokens
        k: image tokens
        v: image tokens
        """
        if mask.sum() == 0.0: return x
        res = x
        B, N, C_ = x.shape
        ws = self.window_size
        H, W = x_size
        h = self.num_heads
        hg, wg = H // ws, W // ws
        x = rearrange(x, 'b (h hs w ws) c -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws)  # N, ws*ws, c
        if self.training:
            x = x * mask.reshape(-1, 1, 1)
        else:
            # x = x * mask.reshape(-1, 1, 1)
            out_ = x
            # x = x[mask==1.0, ...].contiguous()
            # out_ = x
            index = torch.nonzero(mask)
            x = torch.gather(x, dim=0, index=index[:, :, None].expand(-1, ws * ws, C_))

        qkv = self.qkv_local(x)
        q, k, v = rearrange(qkv, 'n w (d h c) -> d n h w c', d=3, h=h).unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        pos_bias = self._get_relative_positional_bias()
        attn = (attn + pos_bias).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws * ws, -1)

        if not self.training:
            # out_ = torch.zeros(B * hg * wg, ws * ws, C_).cuda()
            # out_[mask == 1.0, ...] = out
            # out = out_

            out = out_.scatter(0, index[:, :, None].expand(-1, ws * ws, C_), out)

        # reverse
        mask = mask.reshape(-1, 1, 1).expand(-1, ws * ws, -1)
        out = window_reverse(out, ws, H, W).flatten(1, 2)
        mask = window_reverse(mask, ws, H, W).flatten(1, 2)
        x = out * mask #+ res * (1.0 - mask)
        return x

    def forward_global_broadcast(self, q, k, v):
        """
        q: image tokens
        k: global tokens
        v: global tokens
        """
        B, num_heads, N, _ = q.shape
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        return x

    def forward_global_feature(self, x, attn_mask=None):
        B, N, C = x.shape
        NT = self.query_num

        # qkv
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)  # B, h, N, c

        # split img tokens & global tokens
        q_img, k_img, v_img = q[:, :, NT:], k[:, :, NT:], v[:, :, NT:]
        q_glb, _, _ = q[:, :, :NT], k[:, :, :NT], v[:, :, :NT]

        # global aggregation
        x_glb = self.forward_global_aggregation(q_glb, k_img, v_img, attn_mask)
        k_glb, v_glb = self.kv_global(x_glb).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        x_img = self.forward_global_broadcast(q_img, k_glb, v_glb)
        return x_img

    def forward_global(self, x, x_size=None, mask=None):
        B, L, C = x.shape
        res = x
        if mask.sum() == 0.0: return x
        query = self.prototype.unsqueeze(1).expand((-1, B, -1, -1)).flatten(0, 1)  # 2B, q_n, dim
        query = self.norm1(query)
        if self.training:
            out = torch.cat([query, x], dim=1)
            out = self.forward_global_feature(out, attn_mask=mask)
        else:
            H, W = x_size
            split = min(2 ** round(math.log2(min(H * 1.0, W * 1.0) / 48)), 8)
            chunk_size = 48
            # print(split)
            # hw = H // split
            # ww = W // split
            x = rearrange(x, 'b (h x w y) c -> (b h w) (x y) c', h=split, x=H // split, w=split, y=W // split)
            reshape_mask = rearrange(mask, 'b (h x w y) -> (b h w) (x y)', h=split, x=H // split, w=split, y=W // split)
            out = x.clone().flatten(0, 1)
            index = torch.nonzero(reshape_mask.flatten(0, 1))
            select_x = x.flatten(0, 1)
            select_x = select_x.gather(dim=0, index=index.expand(-1, C))
            num_select = select_x.size(0)
            padding = (split * split) - num_select % (split * split) if num_select % (split * split) != 0 else 0
            # padding = chunk_size - num_select % chunk_size if num_select % chunk_size != 0 else 0

            if padding and select_x.size(0) >= padding:
                pad_x = select_x[-padding:, :].clone()
                # pad_x = torch.ones((padding, select_x.size(-1)), device=select_x.device) * -1000.0
                select_x = torch.cat([select_x, pad_x], dim=0)
                # group = select_x.size(0) // chunk_size
                group = select_x.size(0) // (split * split)

            else:
                group = 1

            select_x = select_x.reshape(B * group, -1, C)  # [(B h w hw ww), C]
            query = query.expand(B * group, -1, -1)

            select_x = torch.cat([query, select_x], dim=1)
            select_x = self.forward_global_feature(select_x).flatten(0, 1)  # [(B h w) (hw ww), C]
            select_x = select_x[:num_select]
            out.scatter_(0, index.expand(-1, C), select_x)  # [(B h w hw ww), C]
            out = rearrange(out, '(b h w x y) c -> b (h x w y) c', b=B, h=split, w=split, x=H // split, y=W // split)
            # print(out.shape, mask.shape)

        x_global = out * mask.unsqueeze(-1) # + res * (1.0 - mask).unsqueeze(-1)
        return x_global

    def forward(self, x, x_size, global_mask=None, local_mask=None):
        H, W = x_size
        # local
        x_local = self.forward_local(x, (H, W), local_mask.squeeze(-1))
        # global
        global_mask = rearrange(global_mask, 'b c h w -> b (h w) c').squeeze(-1)
        x_global = self.forward_global(x, x_size, global_mask)
        x = self.proj(x_local + x_global)
        return x  # x_global #x_local # x_local + x_global

    def flops(self, x_size, local_mask, global_mask):
        flops = 0
        H, W = x_size
        ws = self.window_size

        ## 1. local
        hg, wg = H // ws, W // ws
        flops += (H * W * self.dim * self.dim * 3) # qkv

        local_mask = rearrange(local_mask, 'b c (h hs) (w ws) -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws).squeeze(-1)
        local_mask = local_mask.max(dim=-1).values

        flops += (local_mask.sum() * ws * ws * ws * ws * self.dim) * 2

        ## 2. global
        # W-MSA/SW-MSA
        flops += self.dim * self.query_num # norm
        flops += (H * W * self.dim * self.dim * 3) # qkv

        flops += (global_mask.sum() * self.query_num * self.dim // self.num_heads * self.num_heads) * 2
        flops += (H * W * self.dim * self.dim * 2)
        flops += (global_mask.sum() * self.query_num * self.dim // self.num_heads * self.num_heads) * 2

        return flops

class LightViTAttentionLocalRoll(nn.Module):
    def __init__(self, dim, query_class=1, query_num=256, num_heads=8, window_size=8,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm, shift=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        self.query_num = query_num
        self.query_class = query_class
        self.window_size = window_size
        self.attn_area = window_size * window_size
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_local = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.shifts = shift
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.prototype = nn.Parameter(torch.randn((self.query_class, query_num, dim)))

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.prototype, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, H, N, HW = attn.size()
        attn_policy = policy.reshape(B, 1, 1, HW)

        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward_global_aggregation(self, q, k, v, attn_mask=None):
        """
        q: global tokens
        k: image tokens
        v: image tokens
        """
        B, _, N, _ = q.shape
        _, h, HW, c = k.shape
        q = q * self.scale
        if attn_mask is not None:
            attn = (q @ k.transpose(-2, -1)) * attn_mask.reshape(B, 1, 1, attn_mask.shape[1]).expand(-1, h, N, -1)
            m = attn_mask.reshape(B, 1, 1, attn_mask.shape[1])
            attn = self.softmax_with_policy(attn, attn_mask) * m
        else:
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x

    def forward_local(self, x, x_size, mask):
        """
        q: image tokens
        k: image tokens
        v: image tokens
        """
        H, W = x_size
        B, N, C_ = x.shape
        ws = self.window_size

        h = self.num_heads
        hg, wg = H // ws, W // ws

        ## get mask
        if self.shifts > 0:
            mask = torch.roll(mask, shifts=(-ws // 2, -ws // 2), dims=(2, 3)) # (b 1 h w)
        mask = rearrange(mask, 'b c (h hs) (w ws) -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws)
        # mask = (mask.mean(1).squeeze() >= 0.5).float()
        mask = (mask.mean(1).squeeze() >= 0.999).float()
        if mask.sum() == 0.0: return x

        ## x
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        res = x
        if self.shifts > 0:
            x = torch.roll(x, shifts=(-ws // 2, -ws // 2), dims=(2, 3))
            # mask = torch.roll(mask, shifts=(-ws // 2, -ws // 2), dims=(2, 3)) # (b 1 h w)
        x = rearrange(x, 'b c (h hs) (w ws) -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws)  # N, ws*ws, c
        if self.training:
            x = x * mask.reshape(-1, 1, 1)
        else:
            out_ = x.clone()
            index = torch.nonzero(mask)
            x = torch.gather(x, dim=0, index=index[:, :, None].expand(-1, ws * ws, C_))

        qkv = self.qkv_local(x)
        q, k, v = rearrange(qkv, 'n w (d h c) -> d n h w c', d=3, h=h).unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        pos_bias = self._get_relative_positional_bias()
        attn = (attn + pos_bias).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws * ws, -1)

        if not self.training:
            out = out_.scatter(0, index[:, :, None].expand(-1, ws * ws, C_), out)
        # reverse
        mask = mask.reshape(-1, 1, 1).expand(-1, ws * ws, -1)
        out = rearrange(out, '(b h w) (hs ws) c -> b c (h hs) (w ws)', h=hg, hs=ws, w=wg, ws=ws)
        mask = rearrange(mask, '(b h w) (hs ws) c -> b c (h hs) (w ws)', h=hg, hs=ws, w=wg, ws=ws)
        if self.shifts > 0:
            out = torch.roll(out, shifts=(ws // 2, ws // 2), dims=(2, 3))
            mask = torch.roll(mask, shifts=(ws // 2, ws // 2), dims=(2, 3))
        x = out * mask + res * (1.0 - mask)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

    def forward_global_broadcast(self, q, k, v):
        """
        q: image tokens
        k: global tokens
        v: global tokens
        """
        B, num_heads, N, _ = q.shape
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        return x

    def forward_global_feature(self, x, attn_mask=None):
        B, N, C = x.shape
        NT = self.query_num

        # qkv
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)  # B, h, N, c

        # split img tokens & global tokens
        q_img, k_img, v_img = q[:, :, NT:], k[:, :, NT:], v[:, :, NT:]
        q_glb, _, _ = q[:, :, :NT], k[:, :, :NT], v[:, :, :NT]

        # global aggregation
        x_glb = self.forward_global_aggregation(q_glb, k_img, v_img, attn_mask)
        k_glb, v_glb = self.kv_global(x_glb).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        x_img = self.forward_global_broadcast(q_img, k_glb, v_glb)
        return x_img

    def forward_global(self, x, x_size=None, mask=None):
        B, L, C = x.shape
        res = x
        if mask.sum() == 0.0: return x
        query = self.prototype.unsqueeze(1).expand((-1, B, -1, -1)).flatten(0, 1)  # 2B, q_n, dim
        query = self.norm1(query)
        if self.training:
            out = torch.cat([query, x], dim=1)
            out = self.forward_global_feature(out, attn_mask=mask)
        else:
            # x = x.flatten(0, 1)  # (N, dim)
            # mask = mask.flatten(0, 1).squeeze()
            # select_x = x[mask == 1.0, :].unsqueeze(0)
            # select_x = torch.cat([query, select_x], dim=1)
            # select_x = self.forward_global_feature(select_x)
            # out = torch.zeros_like(x)
            # out[mask == 1.0, :] = select_x
            # out = out.reshape(B, L, C)

            # out = x.clone().flatten(0, 1)
            # index = torch.nonzero(mask.flatten(0, 1))
            # select_x = x.flatten(0, 1)
            # select_x = select_x.gather(dim=0, index=index.expand(-1, C)).unsqueeze(0)
            # select_x = torch.cat([query, select_x], dim=1)
            # select_x = self.forward_global_feature(select_x).flatten(0, 1)
            #
            # out.scatter_(0, index.expand(-1, C), select_x)
            # out = out.reshape(B, L, C)

            H, W = x_size
            split = min(2 ** round(math.log2(min(H * 1.0, W * 1.0) / 48)), 8)
            chunk_size = 48
            # print(split)
            # hw = H // split
            # ww = W // split
            x = rearrange(x, 'b (h x w y) c -> (b h w) (x y) c', h=split, x=H // split, w=split, y=W // split)
            reshape_mask = rearrange(mask, 'b (h x w y) -> (b h w) (x y)', h=split, x=H // split, w=split, y=W // split)
            out = x.clone().flatten(0, 1)
            index = torch.nonzero(reshape_mask.flatten(0, 1))
            select_x = x.flatten(0, 1)
            select_x = select_x.gather(dim=0, index=index.expand(-1, C))
            num_select = select_x.size(0)
            padding = (split * split) - num_select % (split * split) if num_select % (split * split) != 0 else 0
            # padding = chunk_size - num_select % chunk_size if num_select % chunk_size != 0 else 0

            if padding and select_x.size(0) >= padding:
                pad_x = select_x[-padding:, :].clone()
                # pad_x = torch.ones((padding, select_x.size(-1)), device=select_x.device) * -1000.0
                select_x = torch.cat([select_x, pad_x], dim=0)
                # group = select_x.size(0) // chunk_size
                group = select_x.size(0) // (split * split)

            else:
                group = 1

            select_x = select_x.reshape(B * group, -1, C)  # [(B h w hw ww), C]
            query = query.expand(B*group, -1, -1)

            select_x = torch.cat([query, select_x], dim=1)
            select_x = self.forward_global_feature(select_x).flatten(0, 1)  # [(B h w) (hw ww), C]
            select_x = select_x[:num_select]
            out.scatter_(0, index.expand(-1, C), select_x)  # [(B h w hw ww), C]
            out = rearrange(out, '(b h w x y) c -> b (h x w y) c', b=B, h=split, w=split, x=H // split, y=W // split)
            # print(out.shape, mask.shape)

        x_global = out * mask.unsqueeze(-1) + res * (1.0 - mask).unsqueeze(-1)
        return x_global

    def forward(self, x, x_size, global_mask=None, local_mask=None):
        H, W = x_size
        # local
        # x_local = self.forward_local(x, (H, W), local_mask.squeeze(-1))
        x_local = self.forward_local(x, (H, W), local_mask)
        # global
        global_mask = rearrange(global_mask, 'b c h w -> b (h w) c').squeeze(-1)
        x_global = self.forward_global(x, x_size, global_mask)

        return 0.5 * x_local + 0.5 * x_global  # x_global #x_local # x_local + x_global

    def flops(self, x_size, local_mask, global_mask):
        flops = 0
        H, W = x_size
        ws = self.window_size

        ## 1. local
        hg, wg = H // ws, W // ws
        flops += (H * W * self.dim * self.dim * 3) # qkv

        local_mask = rearrange(local_mask, 'b c (h hs) (w ws) -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws).squeeze(-1)
        local_mask = local_mask.max(dim=-1).values

        flops += (local_mask.sum() * ws * ws * ws * ws * self.dim) * 2

        ## 2. global
        # W-MSA/SW-MSA
        flops += self.dim * self.query_num # norm
        flops += (H * W * self.dim * self.dim * 3) # qkv

        flops += (global_mask.sum() * self.query_num * self.dim // self.num_heads * self.num_heads) * 2
        flops += (H * W * self.dim * self.dim * 2)
        flops += (global_mask.sum() * self.query_num * self.dim // self.num_heads * self.num_heads) * 2

        return flops

class LightViTAttentionOnlyLocal(nn.Module):
    def __init__(self, dim, query_class=1, query_num=256, num_heads=8, window_size=8,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm, shift=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        self.query_num = query_num
        self.query_class = query_class
        self.window_size = window_size
        self.attn_area = window_size * window_size
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_local = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.prototype = nn.Parameter(torch.randn((self.query_class, query_num, dim)))

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.prototype, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, H, N, HW = attn.size()
        attn_policy = policy.reshape(B, 1, 1, HW)

        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward_global_aggregation(self, q, k, v, attn_mask=None):
        """
        q: global tokens
        k: image tokens
        v: image tokens
        """
        B, _, N, _ = q.shape
        _, h, HW, c = k.shape
        q = q * self.scale
        if attn_mask is not None:
            attn = (q @ k.transpose(-2, -1)) * attn_mask.reshape(B, 1, 1, attn_mask.shape[1]).expand(-1, h, N, -1)
            m = attn_mask.reshape(B, 1, 1, attn_mask.shape[1])
            attn = self.softmax_with_policy(attn, attn_mask) * m
        else:
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x

    def forward_local(self, x, x_size, mask):
        """
        q: image tokens
        k: image tokens
        v: image tokens
        """
        if mask.sum() == 0.0: return x
        res = x
        B, N, C_ = x.shape
        ws = self.window_size
        H, W = x_size
        h = self.num_heads
        hg, wg = H // ws, W // ws
        x = rearrange(x, 'b (h hs w ws) c -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws)  # N, ws*ws, c
        if self.training:
            x = x * mask.reshape(-1, 1, 1)
        else:
            # x = x * mask.reshape(-1, 1, 1)
            out_ = x.clone()
            # x = x[mask==1.0, ...].contiguous()
            # out_ = x
            index = torch.nonzero(mask)
            x = torch.gather(x, dim=0, index=index[:, :, None].expand(-1, ws * ws, C_))

        qkv = self.qkv_local(x)
        q, k, v = rearrange(qkv, 'n w (d h c) -> d n h w c', d=3, h=h).unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        pos_bias = self._get_relative_positional_bias()
        attn = (attn + pos_bias).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws * ws, -1)

        if not self.training:
            # out_ = torch.zeros(B * hg * wg, ws * ws, C_).cuda()
            # out_[mask == 1.0, ...] = out
            # out = out_

            out = out_.scatter(0, index[:, :, None].expand(-1, ws * ws, C_), out)

        # reverse
        mask = mask.reshape(-1, 1, 1).expand(-1, ws * ws, -1)
        out = window_reverse(out, ws, H, W).flatten(1, 2)
        mask = window_reverse(mask, ws, H, W).flatten(1, 2)
        x = out * mask + res * (1.0 - mask)
        return x

    def forward_global_broadcast(self, q, k, v):
        """
        q: image tokens
        k: global tokens
        v: global tokens
        """
        B, num_heads, N, _ = q.shape
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        return x

    def forward_global_feature(self, x, attn_mask=None):
        B, N, C = x.shape
        NT = self.query_num

        # qkv
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)  # B, h, N, c

        # split img tokens & global tokens
        q_img, k_img, v_img = q[:, :, NT:], k[:, :, NT:], v[:, :, NT:]
        q_glb, _, _ = q[:, :, :NT], k[:, :, :NT], v[:, :, :NT]

        # global aggregation
        x_glb = self.forward_global_aggregation(q_glb, k_img, v_img, attn_mask)
        k_glb, v_glb = self.kv_global(x_glb).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        x_img = self.forward_global_broadcast(q_img, k_glb, v_glb)
        return x_img

    def forward_global(self, x, x_size=None, mask=None):
        B, L, C = x.shape
        res = x
        if mask.sum() == 0.0: return x
        query = self.prototype.unsqueeze(1).expand((-1, B, -1, -1)).flatten(0, 1)  # 2B, q_n, dim
        query = self.norm1(query)
        if self.training:
            out = torch.cat([query, x], dim=1)
            out = self.forward_global_feature(out, attn_mask=mask)
        else:
            # x = x.flatten(0, 1)  # (N, dim)
            # mask = mask.flatten(0, 1).squeeze()
            # select_x = x[mask == 1.0, :].unsqueeze(0)
            # select_x = torch.cat([query, select_x], dim=1)
            # select_x = self.forward_global_feature(select_x)
            # out = torch.zeros_like(x)
            # out[mask == 1.0, :] = select_x
            # out = out.reshape(B, L, C)

            out = x.clone().flatten(0, 1)
            index = torch.nonzero(mask.flatten(0, 1))
            select_x = x.flatten(0, 1)
            select_x = select_x.gather(dim=0, index=index.expand(-1, C)).unsqueeze(0)
            select_x = torch.cat([query, select_x], dim=1)
            select_x = self.forward_global_feature(select_x).flatten(0, 1)

            out.scatter_(0, index.expand(-1, C), select_x)
            out = out.reshape(B, L, C)

            # H, W = x_size
            # split = min(2 ** round(math.log2(min(H * 1.0, W * 1.0) / 48)), 8)
            # chunk_size = 48
            # # print(split)
            # # hw = H // split
            # # ww = W // split
            # x = rearrange(x, 'b (h x w y) c -> (b h w) (x y) c', h=split, x=H // split, w=split, y=W // split)
            # reshape_mask = rearrange(mask, 'b (h x w y) -> (b h w) (x y)', h=split, x=H // split, w=split, y=W // split)
            # out = x.clone().flatten(0, 1)
            # index = torch.nonzero(reshape_mask.flatten(0, 1))
            # select_x = x.flatten(0, 1)
            # select_x = select_x.gather(dim=0, index=index.expand(-1, C))
            # num_select = select_x.size(0)
            # padding = (split * split) - num_select % (split * split) if num_select % (split * split) != 0 else 0
            # # padding = chunk_size - num_select % chunk_size if num_select % chunk_size != 0 else 0
            #
            # if padding and select_x.size(0) >= padding:
            #     pad_x = select_x[-padding:, :].clone()
            #     # pad_x = torch.ones((padding, select_x.size(-1)), device=select_x.device) * -1000.0
            #     select_x = torch.cat([select_x, pad_x], dim=0)
            #
            # # group = select_x.size(0) // chunk_size
            # group = max(select_x.size(0) // (split * split), 1)
            # select_x = select_x.reshape(B * group, -1, C)  # [(B h w hw ww), C]
            # query = query.expand(B * group, -1, -1)
            #
            # select_x = torch.cat([query, select_x], dim=1)
            # select_x = self.forward_global_feature(select_x).flatten(0, 1)  # [(B h w) (hw ww), C]
            # select_x = select_x[:num_select]
            # out.scatter_(0, index.expand(-1, C), select_x)  # [(B h w hw ww), C]
            # out = rearrange(out, '(b h w x y) c -> b (h x w y) c', b=B, h=split, w=split, x=H // split, y=W // split)
            # print(out.shape, mask.shape)

        x_global = out * mask.unsqueeze(-1) + res * (1.0 - mask).unsqueeze(-1)
        return x_global

    def forward(self, x, x_size, global_mask=None, local_mask=None):
        H, W = x_size
        # local
        x_local = self.forward_local(x, (H, W), local_mask.squeeze(-1))
        # global
        # global_mask = rearrange(global_mask, 'b c h w -> b (h w) c').squeeze(-1)
        # x_global = self.forward_global(x, x_size, global_mask)

        return x_local #+ x_global  # x_global #x_local # x_local + x_global

    def flops(self, x_size, local_mask, global_mask):
        flops = 0
        H, W = x_size
        ws = self.window_size

        ## 1. local
        hg, wg = H // ws, W // ws
        flops += (H * W * self.dim * self.dim * 3) # qkv

        local_mask = rearrange(local_mask, 'b c (h hs) (w ws) -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws).squeeze(-1)
        local_mask = local_mask.max(dim=-1).values

        flops += (local_mask.sum() * ws * ws * ws * ws * self.dim) * 2

        ## 2. global
        # W-MSA/SW-MSA
        flops += self.dim * self.query_num # norm
        flops += (H * W * self.dim * self.dim * 3) # qkv

        flops += (global_mask.sum() * self.query_num * self.dim // self.num_heads * self.num_heads) * 2
        flops += (H * W * self.dim * self.dim * 2)
        flops += (global_mask.sum() * self.query_num * self.dim // self.num_heads * self.num_heads) * 2

        return flops

class LightViTAttentionOnlyGlobal(nn.Module):
    def __init__(self, dim, query_class=1, query_num=256, num_heads=8, window_size=8,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm, shift=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        self.query_num = query_num
        self.query_class = query_class
        self.window_size = window_size
        self.attn_area = window_size * window_size
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_local = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.prototype = nn.Parameter(torch.randn((self.query_class, query_num, dim)))

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size,
                                                                                    window_size).view(-1))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.prototype, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.
        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, H, N, HW = attn.size()
        attn_policy = policy.reshape(B, 1, 1, HW)

        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward_global_aggregation(self, q, k, v, attn_mask=None):
        """
        q: global tokens
        k: image tokens
        v: image tokens
        """
        B, _, N, _ = q.shape
        _, h, HW, c = k.shape
        q = q * self.scale
        if attn_mask is not None:
            attn = (q @ k.transpose(-2, -1)) * attn_mask.reshape(B, 1, 1, attn_mask.shape[1]).expand(-1, h, N, -1)
            m = attn_mask.reshape(B, 1, 1, attn_mask.shape[1])
            attn = self.softmax_with_policy(attn, attn_mask) * m
        else:
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return x

    def forward_local(self, x, x_size, mask):
        """
        q: image tokens
        k: image tokens
        v: image tokens
        """
        if mask.sum() == 0.0: return x
        res = x
        B, N, C_ = x.shape
        ws = self.window_size
        H, W = x_size
        h = self.num_heads
        hg, wg = H // ws, W // ws
        x = rearrange(x, 'b (h hs w ws) c -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws)  # N, ws*ws, c
        if self.training:
            x = x * mask.reshape(-1, 1, 1)
        else:
            # x = x * mask.reshape(-1, 1, 1)
            out_ = x.clone()
            # x = x[mask==1.0, ...].contiguous()
            # out_ = x
            index = torch.nonzero(mask)
            x = torch.gather(x, dim=0, index=index[:, :, None].expand(-1, ws * ws, C_))

        qkv = self.qkv_local(x)
        q, k, v = rearrange(qkv, 'n w (d h c) -> d n h w c', d=3, h=h).unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        pos_bias = self._get_relative_positional_bias()
        attn = (attn + pos_bias).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(v.shape[0], ws * ws, -1)

        if not self.training:
            # out_ = torch.zeros(B * hg * wg, ws * ws, C_).cuda()
            # out_[mask == 1.0, ...] = out
            # out = out_

            out = out_.scatter(0, index[:, :, None].expand(-1, ws * ws, C_), out)

        # reverse
        mask = mask.reshape(-1, 1, 1).expand(-1, ws * ws, -1)
        out = window_reverse(out, ws, H, W).flatten(1, 2)
        mask = window_reverse(mask, ws, H, W).flatten(1, 2)
        x = out * mask + res * (1.0 - mask)
        return x

    def forward_global_broadcast(self, q, k, v):
        """
        q: image tokens
        k: global tokens
        v: global tokens
        """
        B, num_heads, N, _ = q.shape
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        return x

    def forward_global_feature(self, x, attn_mask=None):
        B, N, C = x.shape
        NT = self.query_num

        # qkv
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)  # B, h, N, c

        # split img tokens & global tokens
        q_img, k_img, v_img = q[:, :, NT:], k[:, :, NT:], v[:, :, NT:]
        q_glb, _, _ = q[:, :, :NT], k[:, :, :NT], v[:, :, :NT]

        # global aggregation
        x_glb = self.forward_global_aggregation(q_glb, k_img, v_img, attn_mask)
        k_glb, v_glb = self.kv_global(x_glb).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        x_img = self.forward_global_broadcast(q_img, k_glb, v_glb)
        return x_img

    def forward_global(self, x, x_size=None, mask=None):
        B, L, C = x.shape
        res = x
        if mask.sum() == 0.0: return x
        query = self.prototype.unsqueeze(1).expand((-1, B, -1, -1)).flatten(0, 1)  # 2B, q_n, dim
        query = self.norm1(query)
        if self.training:
            out = torch.cat([query, x], dim=1)
            out = self.forward_global_feature(out, attn_mask=mask)
        else:
            # x = x.flatten(0, 1)  # (N, dim)
            # mask = mask.flatten(0, 1).squeeze()
            # select_x = x[mask == 1.0, :].unsqueeze(0)
            # select_x = torch.cat([query, select_x], dim=1)
            # select_x = self.forward_global_feature(select_x)
            # out = torch.zeros_like(x)
            # out[mask == 1.0, :] = select_x
            # out = out.reshape(B, L, C)

            # =========================================
            out = x.clone().flatten(0, 1)
            index = torch.nonzero(mask.flatten(0, 1))
            select_x = x.flatten(0, 1)
            select_x = select_x.gather(dim=0, index=index.expand(-1, C)).unsqueeze(0)
            select_x = torch.cat([query, select_x], dim=1)
            select_x = self.forward_global_feature(select_x).flatten(0, 1)

            out.scatter_(0, index.expand(-1, C), select_x)
            out = out.reshape(B, L, C)

            # ========================================
            # H, W = x_size
            # split = min(2 ** round(math.log2(min(H * 1.0, W * 1.0) / 48)), 8)
            # chunk_size = 48
            # # print(split)
            # # hw = H // split
            # # ww = W // split
            # x = rearrange(x, 'b (h x w y) c -> (b h w) (x y) c', h=split, x=H // split, w=split, y=W // split)
            # reshape_mask = rearrange(mask, 'b (h x w y) -> (b h w) (x y)', h=split, x=H // split, w=split, y=W // split)
            # out = x.clone().flatten(0, 1)
            # index = torch.nonzero(reshape_mask.flatten(0, 1))
            # select_x = x.flatten(0, 1)
            # select_x = select_x.gather(dim=0, index=index.expand(-1, C))
            # num_select = select_x.size(0)
            # padding = (split * split) - num_select % (split * split) if num_select % (split * split) != 0 else 0
            # # padding = chunk_size - num_select % chunk_size if num_select % chunk_size != 0 else 0
            #
            # if padding and select_x.size(0) >= padding:
            #     pad_x = select_x[-padding:, :].clone()
            #     # pad_x = torch.ones((padding, select_x.size(-1)), device=select_x.device) * -1000.0
            #     select_x = torch.cat([select_x, pad_x], dim=0)
            #
            # # group = select_x.size(0) // chunk_size
            # group = max(select_x.size(0) // (split * split), 1)
            # select_x = select_x.reshape(B * group, -1, C)  # [(B h w hw ww), C]
            # query = query.expand(B * group, -1, -1)
            #
            # select_x = torch.cat([query, select_x], dim=1)
            # select_x = self.forward_global_feature(select_x).flatten(0, 1)  # [(B h w) (hw ww), C]
            # select_x = select_x[:num_select]
            # out.scatter_(0, index.expand(-1, C), select_x)  # [(B h w hw ww), C]
            # out = rearrange(out, '(b h w x y) c -> b (h x w y) c', b=B, h=split, w=split, x=H // split, y=W // split)
            # print(out.shape, mask.shape)

        x_global = out * mask.unsqueeze(-1) + res * (1.0 - mask).unsqueeze(-1)
        return x_global

    def forward(self, x, x_size, global_mask=None, local_mask=None):
        H, W = x_size
        # local
        # x_local = self.forward_local(x, (H, W), local_mask.squeeze(-1))
        # global
        global_mask = rearrange(global_mask, 'b c h w -> b (h w) c').squeeze(-1)
        x_global = self.forward_global(x, x_size, global_mask)

        return  x_global #+ x_global  # x_global #x_local # x_local + x_global

    def flops(self, x_size, local_mask, global_mask):
        flops = 0
        H, W = x_size
        ws = self.window_size

        ## 1. local
        hg, wg = H // ws, W // ws
        flops += (H * W * self.dim * self.dim * 3) # qkv

        local_mask = rearrange(local_mask, 'b c (h hs) (w ws) -> (b h w) (hs ws) c', h=hg, hs=ws, w=wg, ws=ws).squeeze(-1)
        local_mask = local_mask.max(dim=-1).values

        flops += (local_mask.sum() * ws * ws * ws * ws * self.dim) * 2

        ## 2. global
        # W-MSA/SW-MSA
        flops += self.dim * self.query_num # norm
        flops += (H * W * self.dim * self.dim * 3) # qkv

        flops += (global_mask.sum() * self.query_num * self.dim // self.num_heads * self.num_heads) * 2
        flops += (H * W * self.dim * self.dim * 2)
        flops += (global_mask.sum() * self.query_num * self.dim // self.num_heads * self.num_heads) * 2

        return flops
