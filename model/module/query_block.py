import torch.nn as nn
import torch
from model.refinetor.util import Mlp, window_reverse
class LightViTAttention(nn.Module):
    def __init__(self, dim,  query_class=2, query_num=256, num_heads=8,  qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.query_num = query_num
        self.query_class = query_class
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv_global = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        # print("policy", policy.shape)
        B, HW, _ = policy.size()
        B, H, N, HW = attn.size()
        attn_policy = policy.reshape(B, 1, 1, HW)  # * policy.reshape(B, 1, N, 1)
        # eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        # attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # attn = attn.exp_() * attn_policy
        # return attn / attn.sum(dim=-1, keepdim=True)

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

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

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
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        return x


    def forward_global(self, x, attn_mask=None):
        B, N, C = x.shape
        NT = self.query_num
        # qkv
        qkv = self.qkv(x)
        # t_qkv = qkv
        q, k, v = qkv.view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0) # B, h, N, c

        # split img tokens & global tokens
        x_glb = x[:, :, :NT]
        q_img, k_img, v_img = q[:, :, NT:], k[:, :, NT:], v[:, :, NT:]
        q_glb, _, _ = q[:, :, :NT], k[:, :, :NT], v[:, :, :NT]

        # global aggregation
        x_glb = self.forward_global_aggregation(q_glb, k_img, v_img, attn_mask)
        k_glb, v_glb = self.kv_global(x_glb).view(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)

        x_img = self.forward_global_broadcast(q_img, k_glb, v_glb)
        x = self.proj(x_img)
        return x

    def forward(self, x, H=None, W=None, attn_mask=None):
        return self.forward_global(x, attn_mask=attn_mask)

class QueryTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., query_num=256,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.query_class = 1
        self.query_num = query_num
        self.prototype = nn.Parameter(torch.randn((self.query_class, query_num, dim)))
        self.attn = LightViTAttention(dim=dim, query_class=self.query_class, query_num=query_num, num_heads=num_heads,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, var=None):
        B, L, C = x.shape
        shortcut = x

        if self.training:
            query = self.prototype.unsqueeze(1).expand((-1, B, -1, -1)).flatten(0, 1)  # 2B, q_n, dim
            mask = var
            x = torch.cat([query, x], dim=1)
            x = self.attn(self.norm1(x), attn_mask=mask)
            x = x * mask
        else:
            query = self.prototype.unsqueeze(1).expand((-1, B, -1, -1)).flatten(0, 1)
            x = x.flatten(0, 1)  # (N, dim)
            mask = var.flatten(0, 1).squeeze()
            select_x = x[mask==1.0, :].unsqueeze(0)
            if min(select_x.shape) != 0:
                select_x = torch.cat([query, select_x], dim=1)
                select_x = self.attn(self.norm1(select_x))
            out = torch.zeros_like(x)
            out[mask==1.0, :] = select_x
            x = out.reshape(B, L, C)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops