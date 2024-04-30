# -*- coding: utf-8 -*-
"""
@author: csmliu
@e-mail: csmliu@outlook.com
"""
from random import random

import numpy as np

def Conv(in_shape, inc, outc, ks, stride=1, padding=None,
         groups=1, bias=True, mask=None):
    if padding is None:
        padding = ks//2
    if groups != 1:
        assert inc % groups == 0 and outc % groups == 0
        inc = inc // groups
        outc = outc // groups

    _per_pos = ks * ks * inc * outc * groups
    if mask is not None:
        assert all(in_shape == mask.shape)
        n_pos = (mask > 0).sum()
    else:
        n_pos = np.array(in_shape).prod()
    _sum = _per_pos * n_pos
    if bias:
        return _sum + n_pos * outc
    return _sum

def BN(in_shape, inc):
    return np.array(in_shape).prod() * inc * 2 # affine

def ReLU(in_shape, inc):
    return np.array(in_shape).prod() * inc

def pixsf(in_shape, inc, scale):
    _sum_conv = Conv(in_shape, inc, inc*scale**2, 3)
    return np.array(in_shape).prod() * inc + _sum_conv

def pool(in_shape, inc):
    return np.array(in_shape).prod() * inc

def linear(inc, outc, bias=True):
    _sum = inc * outc
    if bias:
        return _sum + outc
    return _sum

def upsample(in_shape, inc, scale=2):
    return (np.array(in_shape)*scale).prod() * inc

def ResBlock(in_shape, inc, mode='CRC', mask=None):
    _sum = 0
    for m in mode:
        if m == 'C':
            _sum += Conv(in_shape, inc, inc, ks=3, mask=mask)
        elif m in 'RPL':
            _sum += ReLU(in_shape, inc)
        elif m == 'B':
            _sum += BN(in_shape, inc)
        else:
            print('mode %s is not supported in ResBlock.'%m)
    return _sum + np.array(in_shape).prod() * inc

def CA(in_shape, inc):
    _sum = np.array(in_shape).prod() * inc # AvgPool
    _sum += linear(inc, inc//16) # 1st conv
    _sum += inc // 16 # ReLU
    _sum += linear(inc//16, inc) # 2nd conv
    _sum += inc // 16 # Sigmoid
    _sum += np.array(in_shape).prod() * inc
    return _sum

def clip(x, layer):
    return np.clip(x, layer, layer+1) - layer



class FLOPs():

    @staticmethod
    def SWIM_WHOLE(in_shape, scale):
        flops = 0
        H, W = in_shape
        embed_dim = 180
        # window_size = 8
        window_size = 8
        ws = window_size
        mod_pad_h = (ws - H % ws) % ws
        mod_pad_w = (ws - W % ws) % ws
        H += mod_pad_h
        W += mod_pad_w

        num_heads = 6
        depths = [6, 6, 6, 6, 6, 6]
        # depths = [2]
        # num_heads = [6, 6, 6, 6, 6, 6],
        mlp_ratio = 2

        flops += H * W * 3 * embed_dim * 9 # conv_first
        # flops += H * W * embed_dim * embed_dim * 9
        # patch embeding
        flops += H * W * embed_dim

        def attn_flops(N):
            # calculate flops for 1 window with token length of N
            flops = 0
            # qkv = self.qkv(x)
            flops += N * embed_dim * 3 * embed_dim
            # attn = (q @ k.transpose(-2, -1))
            flops += num_heads * N * (embed_dim // num_heads) * N
            #  x = (attn @ v)
            flops += num_heads * N * N * (embed_dim // num_heads)
            # x = self.proj(x)
            flops += N * embed_dim * embed_dim
            return flops

        def block_flops():
            flops = 0
            # norm1
            flops += embed_dim * H * W
            # W-MSA/SW-MSA
            nW = H * W / (window_size * window_size)
            flops += nW * attn_flops(window_size * window_size)
            # mlp
            flops += 2 * H * W * embed_dim * embed_dim * mlp_ratio
            # norm2
            flops += embed_dim * H * W
            return flops

        # basic layer
        def basic_flops(deep):
            flops = 0
            flops += (block_flops() * deep)
            return flops

        # layer
        def RSTB_flops(deep):
            flops = 0
            flops += basic_flops(deep)
            flops += H * W * embed_dim * embed_dim * 9
            flops += H * W * embed_dim
            return flops

        for i, deep in enumerate(depths):
            # print(deep)
            flops += RSTB_flops(deep)

        # flops += H * W * 9 * embed_dim * embed_dim
        # flops += H * W * 9 * embed_dim * embed_dim
        # flops += ReLU(in_shape, embed_dim)
        # flops += H * W * 9 * embed_dim * embed_dim

        # upsample flops
        # if scale == 3:
        #     flops += pixsf(in_shape, embed_dim, 3)
        #     in_shape *= 3
        # else:
        #     assert scale in (2, 4)
        #     for i in range(1, scale, 2):
        #         flops += pixsf(in_shape, embed_dim, 2)
        #         in_shape *= 2
        # flops += Conv(in_shape, embed_dim, 3, 3)

        # flops += self.upsample.flops()
        # H, W = self.input_resolution
        flops += H * W * embed_dim * embed_dim * 9
        flops += H * W * embed_dim * (3 * scale * scale) * 9

        return flops

    @staticmethod
    def ETB(in_shape, scale):
        flops = 0
        H, W = in_shape
        embed_dim = 180
        # window_size = 8
        window_size = 8
        ws = window_size
        mod_pad_h = (ws - H % ws) % ws
        mod_pad_w = (ws - W % ws) % ws
        H += mod_pad_h
        W += mod_pad_w

        num_heads = 6
        depths = [6, 6, 6, 6, 6, 6]
        # depths = [2]
        # num_heads = [6, 6, 6, 6, 6, 6],
        mlp_ratio = 2
        num_feat = 32
        noise_down = 2

        ws_0 = 6
        ws_1 = 24

        # head
        flops += H * W * 3 * num_feat * 9  # conv_first
        for i in range(noise_down):
            flops += H * W * num_feat * 2 ** i * num_feat * 2 ** (i) * 9
            flops += H * W * num_feat * 2 ** i * num_feat * 2 ** (i + 1) * 9
            flops += H * W * num_feat * 2 ** i * num_feat * 2 ** (i + 1) * 1
            flops += H * W * embed_dim * 2

        flops += H * W * num_feat * noise_down ** 2 * embed_dim * 9

        # body

        def attn_flops():
            # calculate flops for 1 window with token length of N
            flops = 0
            # flops += H * W * embed_dim // 2 * 9
            N = H // ws_0 * W // ws_1
            ws = ws_0 * ws_1
            flops += num_heads * N * (embed_dim // 2 // num_heads) * ws * ws
            flops += num_heads * N * (embed_dim // 2 // num_heads) * ws * ws
            flops += H * W * embed_dim // 2 * embed_dim // 2 * 9
            return flops

        def block_flops():
            flops = 0
            # norm1
            flops += embed_dim * H * W
            # W-MSA/SW-MSA
            flops += 2 * attn_flops()
            # mlp
            flops += (2 * H * W * embed_dim * embed_dim * mlp_ratio + H * W * embed_dim * 25)
            # norm2
            flops += embed_dim * H * W
            # qkv = self.qkv(x)
            flops += H * W * embed_dim * 3 * embed_dim
            # proj
            flops += H * W * embed_dim * embed_dim

            return flops

        # basic layer
        def basic_flops(deep):
            flops = 0
            flops += (block_flops() * deep)
            return flops

        # layer
        def RSTB_flops(deep):
            flops = 0
            flops += basic_flops(deep)
            flops += H * W * embed_dim * embed_dim * 9
            flops += H * W * embed_dim
            return flops

        # flops += H * W * embed_dim
        for i, deep in enumerate(depths):
            # print(deep)
            flops += RSTB_flops(deep)

        flops += H * W * embed_dim
        flops += H * W * embed_dim * embed_dim * 9



        # tail
        for i in reversed(range(noise_down)):
            in_chl = embed_dim if i == noise_down - 1 else num_feat * 2 ** (i + 2)
            out_chl = num_feat * 2 ** (i + 1)

            flops += H * W * in_chl * out_chl * 9
            flops += H * W * out_chl * 2 * out_chl * 9
            flops += H * W * out_chl * out_chl * 9
            flops += H * W * out_chl * 2 * out_chl * 1
            flops += H * W * embed_dim * 2

        flops += H * W * embed_dim * (3 * scale * scale) * 9
        flops += H * scale * W * scale * num_feat*2**noise_down * 3 * 9
        flops += H * W * num_feat * 2 * 3 * 9

        return flops

    @staticmethod
    def HAT(in_shape, scale):
        flops = 0
        H, W = in_shape
        embed_dim = 180
        # window_size = 8
        window_size = 16
        ws = window_size
        mod_pad_h = (ws - H % ws) % ws
        mod_pad_w = (ws - W % ws) % ws
        H += mod_pad_h
        W += mod_pad_w

        num_heads = 6
        depths = [6, 6, 6, 6, 6, 6]
        # depths = [2]
        # num_heads = [6, 6, 6, 6, 6, 6],
        mlp_ratio = 2
        compress_ratio = 3
        squeeze_ratio = 30

        flops += H * W * 3 * embed_dim * 9  # conv_first
        # flops += H * W * embed_dim * embed_dim * 9
        # patch embeding
        flops += H * W * embed_dim

        def attn_flops(N):
            # calculate flops for 1 window with token length of N
            flops = 0
            # qkv = self.qkv(x)
            flops += N * embed_dim * 3 * embed_dim
            # attn = (q @ k.transpose(-2, -1))
            flops += num_heads * N * (embed_dim // num_heads) * N
            #  x = (attn @ v)
            flops += num_heads * N * N * (embed_dim // num_heads)
            # x = self.proj(x)
            flops += N * embed_dim * embed_dim
            return flops

        def CAB_Flops():
            flops = 0
            flops += 3 * H * W * embed_dim * (embed_dim // compress_ratio) * 2
            flops += H * W * embed_dim
            flops += 1 * H * W * embed_dim * (embed_dim // squeeze_ratio) * 2
            flops += H * W * embed_dim # relu
            return flops


        def block_flops():
            flops = 0
            # norm1
            flops += embed_dim * H * W
            flops += CAB_Flops()
            # W-MSA/SW-MSA
            nW = H * W / (window_size * window_size)
            flops += nW * attn_flops(window_size * window_size)
            # mlp
            flops += 2 * H * W * embed_dim * embed_dim * mlp_ratio
            # norm2
            flops += embed_dim * H * W
            return flops

        def over_attn_flops(N, M=32):
            flops = 0
            flops += embed_dim * H * W
            flops += embed_dim * H * W
            # qkv = self.qkv(x)
            flops += N * embed_dim * 3 * embed_dim
            # attn = (q @ k.transpose(-2, -1))
            flops += num_heads * N * (embed_dim // num_heads) * M
            #  x = (attn @ v)
            flops += num_heads * N * M * (embed_dim // num_heads)
            # x = self.proj(x)
            flops += N * embed_dim * embed_dim
            return flops

        # basic layer
        def basic_flops(deep):
            flops = 0
            flops += (block_flops() * deep)
            nW = H * W / (window_size * window_size)
            flops += nW * over_attn_flops(window_size * window_size, (window_size + window_size * 0.5) * (window_size + window_size * 0.5)) * deep
            return flops

        # layer
        def RHAG_flops(deep):
            flops = 0
            flops += basic_flops(deep)
            flops += H * W * embed_dim * embed_dim * 9
            flops += H * W * embed_dim
            return flops

        for i, deep in enumerate(depths):
            # print(deep)
            flops += RHAG_flops(deep)

        # flops += H * W * 9 * embed_dim * embed_dim
        # flops += H * W * 9 * embed_dim * embed_dim
        # flops += ReLU(in_shape, embed_dim)
        # flops += H * W * 9 * embed_dim * embed_dim

        # upsample flops
        # if scale == 3:
        #     flops += pixsf(in_shape, embed_dim, 3)
        #     in_shape *= 3
        # else:
        #     assert scale in (2, 4)
        #     for i in range(1, scale, 2):
        #         flops += pixsf(in_shape, embed_dim, 2)
        #         in_shape *= 2
        # flops += Conv(in_shape, embed_dim, 3, 3)

        # flops += self.upsample.flops()
        # H, W = self.input_resolution
        flops += H * W * embed_dim * embed_dim * 9
        flops += H * W * embed_dim * (3 * scale * scale) * 9
        flops += H * W * embed_dim * 3 * 9

        return flops

    @staticmethod
    def ELAN(in_shape, scale):
        flops = 0
        H, W = in_shape
        ws = 8
        mod_pad_h = (ws - H % ws) % ws
        mod_pad_w = (ws - W % ws) % ws
        H += mod_pad_h
        W += mod_pad_w

        embed_dim = 60
        m_elan = 24
        window_sizes=[4, 8, 16]
        r_expand = 2
        n_share = 1

        # head
        flops += H * W * 3 * embed_dim * 9

        # body
        def elanblock():
            b_f = 0

            # lfe
            b_lfe = 0
            b_lfe += H * W * 1 * embed_dim * 9 * embed_dim // embed_dim  # group conv
            b_lfe += H * W * embed_dim * embed_dim * r_expand * 1
            # print(H * W * embed_dim * embed_dim * r_expand * 1)
            # b_lfe += ReLU(in_shape, embed_dim * r_expand)
            b_lfe += H * W * 1 * embed_dim * r_expand * 9 * embed_dim // (embed_dim * r_expand)
            b_lfe += H * W * embed_dim * r_expand * embed_dim * 1

            # b_lfe += H * W * 1 * embed_dim * 9 * embed_dim   # group conv
            # b_lfe += ReLU(in_shape, embed_dim * r_expand)
            # b_lfe += H * W * 1 * embed_dim * r_expand * 9 * embed_dim


            # gmsa
            gmsa_attn = 0
            gmsa_attn += H * W * embed_dim * embed_dim * 2 * 1
            gmsa_attn += BN(in_shape, embed_dim * 2)
            split_c = embed_dim * 2 // 3
            for xs in window_sizes:
                nW = (H * W) // xs // xs
                win_s = xs * xs
                # gmsa_attn += nW * win_s * split_c * 2 * split_c
                gmsa_attn += nW * win_s * split_c * win_s # q*k
                gmsa_attn += nW * win_s * win_s * split_c # attn * v
            gmsa_attn += H * W * embed_dim * embed_dim * 1

            gmsa_w_attn = 0
            gmsa_w_attn += H * W * embed_dim * embed_dim * 2 * 1
            gmsa_attn += BN(in_shape, embed_dim * 2)
            split_c = embed_dim // 3
            for xs in window_sizes:
                nW = H * W // xs // xs
                win_s = xs * xs
                # gmsa_w_attn += nW * win_s * split_c * 2 * split_c
                # gmsa_w_attn += nW * win_s * split_c * win_s  # qk
                gmsa_w_attn += nW * win_s * win_s * split_c  # attn * v
            gmsa_w_attn += H * W * embed_dim * embed_dim * 1

            b_f += (2 * b_lfe + gmsa_attn + gmsa_w_attn)
            return b_f, gmsa_attn + gmsa_w_attn

        total_flops, transformer_flops = elanblock()
        flops += total_flops * (m_elan // 2)
        # tail
        flops += H * W * embed_dim * (3 * scale * scale) * 9
        return flops, transformer_flops * (m_elan // 2)

    @staticmethod
    def SWINIR(in_shape, scale):
        H, W = in_shape

    @staticmethod
    def EDSR(in_shape, scale, mask=None, nb=32, nf=256):
        _sum = 0
        _sum += Conv(in_shape, 3, nf, 3)
        if mask is None:
            _sum += ResBlock(in_shape, nf) * nb
        else:
            for i in range(nb):
                _sum += ResBlock(in_shape, nf, mask=clip(mask, i))
        _sum += Conv(in_shape, nf, nf, 3) + in_shape.prod() * nf
        if scale == 3:
            _sum += pixsf(in_shape, nf, 3)
            in_shape *= 3
        else:
            assert scale in (2, 4)
            for i in range(1, scale, 2):
                _sum += pixsf(in_shape, nf, 2)
                in_shape *= 2
        _sum += Conv(in_shape, nf, 3, 3)
        return _sum

    @staticmethod
    def NLSN(in_shape, scale,  nb=16, nf=64):
        flop = 0
        flop += FLOPs.EDSR(in_shape, scale, nb=nb, nf=nf)
        n_hash = in_shape.prod() // 144 + 1
        block_flops = 0
        block_flops += Conv(in_shape, nf, nf // 4, 3)
        block_flops += Conv(in_shape, nf, nf, 1)
        # hash backet
        block_flops += in_shape.prod() * (nf // 4) * 4 * 7 # torch.einsum('btf,bfhi->bhti', x, random_rotations) #[N, n_hashes, H*W, hash_buckets//2] n_hashes=4, hash_bucket=15

        # unormalized attention score
        block_flops += 4*n_hash*144*144*3*(nf // 4) # q*k
        block_flops += 4*n_hash*144*3*144*nf # k*v

        return flop + block_flops * (nb // 8 + 1)

    @staticmethod
    def ENLCN(in_shape, scale, nb=16, nf=64, m=128):
        flop = 0
        # flop += FLOPs.EDSR(in_shape, scale)
        reduction = 4
        block_flops = 0
        block_flops += Conv(in_shape, nf, nf // reduction, 1) * 2
        block_flops += Conv(in_shape, nf, nf, 1)

        # create kernel  q, k -> (1, 1, H*W, 128)
        block_flops += in_shape.prod()*m*(nf // reduction) * 2

        # q, k , v -> (
        block_flops += in_shape.prod()*m
        block_flops += m*nf*in_shape.prod()
        block_flops += in_shape.prod()*m*nf
        block_flops += in_shape.prod()*nf
        return flop + block_flops * (nb // 8 + 1)

    @staticmethod
    def NL_EDSR(in_shape, scale, nb=16, nf=64):
        flop = 0
        # flop += FLOPs.EDSR(in_shape, scale, nb=nb, nf=nf)

        block_flops = 0
        block_flops += Conv(in_shape, nf, nf // 2, 1) * 2
        block_flops += Conv(in_shape, nf, nf, 1)
        block_flops += ReLU(in_shape, nf) * 3
        # hash backet
        block_flops += in_shape.prod() * in_shape.prod() * (nf // 2)
        block_flops += in_shape.prod() * in_shape.prod() * nf
        return flop + block_flops * (nb // 8 + 1)


    @staticmethod
    def AdaRCAN(in_shape, scale, mask=None):
        return Conv(in_shape, 64, 64, 3) * 4 + ReLU(in_shape, 128) * 4 + \
               Conv(in_shape, 64, 10, 3)

    @staticmethod
    def EF_EDSR(in_shape, scale, m_nums, block_num, group=False, embed_dim=128, nb=16, nf=64, window_size=256):
        flop = 0
        flop += FLOPs.EDSR(in_shape, scale, nb=nb, nf=nf)
        # SCPS
        flop += (Conv(in_shape, 64, 64, 3) * 2 + Conv(in_shape, 64, 2, 3)) * 3

        # Refinetor
        # encoder
        # patch_embeding
        # pe_flops = (Conv(in_shape, 16, embed_dim//4, 1) + Conv(in_shape, 16, embed_dim//4, 3) * (1 + 2 + 3)) # multi
        pe_flops = Conv(in_shape, nf, embed_dim, 3)  # multi

        # pe_flops = (Conv(in_shape, 64, 128, 1))
        # block
        # if group:
        #     if m_nums <= 512:
        #         attn_flops = m_nums * m_nums * 128 * 2
        #     else:
        #         if m_nums % 512 == 0:
        #             attn_flops = (m_nums // 512) * (512 * 512) * 128 * 2
        #         else:
        #             attn_flops = (m_nums // 512 + 1) * (512 * 512) * 128 * 2
        #             # print('group:', attn_flops)
        # else:
        #     attn_flops = m_nums * m_nums * 128 * 2
        #     # print('ungroup:', attn_flops)

        # attn_flops = (embed_dim * embed_dim) * m_nums * 2
        attn_flops = (m_nums // window_size) * (window_size * window_size) * embed_dim * 2
        mlp_flops = 2 * linear(embed_dim, 2 * embed_dim) + ReLU(in_shape, embed_dim)
        block_flops = (attn_flops + mlp_flops) * block_num
        #
        # # decoder
        decoder_flops = linear(64, 128)
        flop += (pe_flops + block_flops + decoder_flops) * 3
        return flop


    @staticmethod
    def SRCNN(in_shape, scale, mask=None):
        _sum = 0
        _sum += Conv(in_shape, 1, 64, 9) + ReLU(in_shape, 64)
        _sum += Conv(in_shape, 64, 32, 5) + ReLU(in_shape, 32)
        _sum += Conv(in_shape, 32, 1, 5)
        return _sum



def find(name):
    for func in FLOPs.__dict__.keys():
        if func.lower() == name.lower():
            return func
    raise ValueError('No function named %s is found'%name)

# def cvt(num):
#     units = ['', 'K', 'M', 'G', 'T', 'P', 'Z']
#     cur = 0
#     while num > 1024:
#         cur += 1
#         num /= 1024
#     return '%.3f %s FLOPs' % (num, units[cur])

def cvt(num, binary=True):
    step = 1024 if binary else 1000
    return '%.2f GFLOPs' %(num / step**3)

# def chop(input_shape, shave=10, min_size=160000):
#     h, w = input_shape
#     h_half, w_half = h//2, w//2
#     h_size, w_size = h_half+shave, w_half+shave
#     if h_size * w_size < min_size:
#         return np.array([np.array([h_size, w_size])]*4)
#     else:
#         ret = np.array([chop(np.array([h_size, w_size]))]*4)
#         return ret

# def chop_pred(pred, shave=10, min_size=160000):
#     if pred is None: return None
#     h, w = pred.shape
#     h_half, w_half = h//2, w//2
#     h_size, w_size = h_half+shave, w_half+shave
#     if h_size * w_size < min_size:
#         return np.array([
#                 pred[0:h_size, 0:w_size],
#                 pred[0:h_size, (w-w_size):w],
#                 pred[(h-h_size):h, 0:w_size],
#                 pred[(h-h_size):h, (w-w_size):w]
#             ])
#     else:
#         return np.array([
#                 chop_pred(pred[0:h_size, 0:w_size]),
#                 chop_pred(pred[0:h_size, (w-w_size):w]),
#                 chop_pred(pred[(h-h_size):h, 0:w_size]),
#                 chop_pred(pred[(h-h_size):h, (w-w_size):w])
#             ])
#
# def chop(input_shape, m_nums, shave=10, min_size=160000):
#     h, w = input_shape
#     h_half, w_half = h//2, w//2
#     m_nums_half = m_nums // 4
#     h_size, w_size = h_half+shave, w_half+shave
#     if h_size * w_size < min_size:
#         return np.array([np.array([h_size, w_size])]*4), [m_nums_half] * 4
#     else:
#         size_array, num_array = chop(np.array([h_size, w_size]))
#         ret = np.array([size_array]*4), [num_array]*4
#         return ret

def chop_EF_EDSR(input_shape, m_nums, scale, block_num, nb=16, nf=64, shave=10, min_size=160000):
    h, w = input_shape
    h_half, w_half = h//2, w//2
    m_nums_half = m_nums // 4
    h_size, w_size = h_half+shave, w_half+shave
    flops = 0
    if h_size * w_size < min_size:
        # return np.array([np.array([h_size, w_size])]*4), [m_nums_half] * 4
        flops += FLOPs.EF_EDSR(np.array([h_size, w_size]), scale, m_nums_half, block_num, nb=nb, nf=nf) * 4
        return flops
    else:
        flops += chop_EF_EDSR(np.array([h_size, w_size]), scale, m_nums_half, block_num, nb=nb, nf=nf) * 4
        # ret = np.array([size_array]*4), [num_array]*4
        return flops

def chop_pred(pred, shave=10, min_size=160000):
    if pred is None: return None
    h, w = pred.shape
    h_half, w_half = h//2, w//2
    h_size, w_size = h_half+shave, w_half+shave
    if h_size * w_size < min_size:
        return np.array([
                pred[0:h_size, 0:w_size],
                pred[0:h_size, (w-w_size):w],
                pred[(h-h_size):h, 0:w_size],
                pred[(h-h_size):h, (w-w_size):w]
            ])
    else:
        return np.array([
                chop_pred(pred[0:h_size, 0:w_size]),
                chop_pred(pred[0:h_size, (w-w_size):w]),
                chop_pred(pred[(h-h_size):h, 0:w_size]),
                chop_pred(pred[(h-h_size):h, (w-w_size):w])
            ])


methods = {
    'hr': ['srcnn', 'vdsr'],
    'lr': ['edsr', 'adaedsr', 'adaedsr_fixd', 'rdn', 'rcan', 'san', 'adarcan'],
}
methods = {i:j for j in methods.keys() for i in methods[j]}

from option import args
import data
from tqdm import tqdm
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
def prepare(*args):
    device = torch.device('cuda')
    def _prepare(tensor):
        # if self.args.precision == 'half': tensor = tensor.half()
        return tensor.to(device)

    return [_prepare(a) for a in args]



if __name__ == '__main__':
    args.test_only = True
    args.scale = [4]
    args.data_test  = ['Set5','Set14','B100','Urban100','Manga109']
    # args.data_test  = ['test2k','test4k','test8k']
    loader = data.Data(args)
    loader_test = loader.loader_test
    height = int((1024 // 4 // 8 + 1) * 8)
    width = int((720 // 4 // 8 + 1) * 8)
    # height = 1024
    # width = 720
    print(height, width)
    in_shape = torch.tensor([height, width], dtype=torch.float)
    # # flops = FLOPs.SWIM_WHOLE(in_shape, 4)
    # flops = FLOPs.ELAN(in_shape, 4)
    # print(flops / 1e9)
    # #
    # flops = FLOPs.SWIM_WHOLE(in_shape, 4)
    # print(flops / 1e9)
    # flops = FLOPs.EDSR(in_shape, 4, nb=16, nf=64)
    # H, W = height, width
    # n_feats = 64
    # flops = 0
    # flops += (H * W * n_feats * n_feats * 9) * 2 * 16
    # flops += (H * W * n_feats) * 16  # relu
    # flops = ResBlock(in_shape, 64) * 16!
    # print(flops / 1e9)

    # m_nums = in_shape.prod() * 0.45
    # flops = FLOPs.EF_EDSR(in_shape, 4, m_nums, 4, embed_dim=128, window_size=512, nb=32, nf=128)
    # flops = FLOPs.HAT(in_shape, 4)
    flops = FLOPs.ETB(in_shape, 4)
    print(flops / 1e9)
    # for idx_data, d in enumerate(loader_test):
    #     for idx_scale, scale in enumerate(args.scale):
    #         d.dataset.set_scale(idx_scale)
    #         psnr_sr_chop = 0
    #         flops_total = 0
    #         flops_transformer = 0
    #         flops_total_group = 0
    #         for lr, hr, filename, _ in tqdm(d, ncols=80):
    #             lr, hr = prepare(lr, hr)
    #             in_shape = np.array(lr.shape[2:])
    #             # print(in_shape)
    #             m_nums = in_shape.prod()*0.45
    #             # m_nums = torch.sum(var_list[1] == 1.0) - int(in_shape.prod()*0.05)
    #             # m_nums = torch.sum(var_list[1] == 1.0).cpu().numpy()
    #             # sparity.append(m_nums / in_shape.prod())
    #             # print('m_nums', m_nums)
    #             # flops = FLOPs.SWIM_WHOLE(in_shape, scale) # 15.79
    #             # print(flops / 1000000000)
    #             # print(flops / 1000000000)
    #             # flops, transformer_flops = FLOPs.ELAN(in_shape, scale) # 15.66
    #             # flops = FLOPs.EF_EDSR(in_shape, scale, m_nums, 1, embed_dim=128, window_size=1024, nb=16, nf=64)
    #             # flops = FLOPs.EF_EDSR(in_shape, scale, m_nums, 4, embed_dim=128, window_size=512, nb=32, nf=128)
    #             # flops = FLOPs.SWIM(in_shape, scale)
    #             # flops = FLOPs.EF_RCAN(in_shape, scale, m_nums, 4, embed_dim=192, n_group=5, n_block=10)
    #             # flops = FLOPs.ENLCN(in_shape, scale, m=1024, nf=64)  # 1024 7.37
    #             flops_total += (flops / 1e9)
    #             # flops_transformer += (transformer_flops / 1e9)
    #         flops_total /= len(d)
    #         flops_transformer /= len(d)
    #         print("total flops:", flops_total)
            # print("flops_transformer:", flops_transformer)
