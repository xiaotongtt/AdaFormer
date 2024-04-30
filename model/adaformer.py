import torch.nn as nn
import torch.nn.functional as F
from model import common
from model.adaformer_block import PSPL
import model.adaformer_block as ab
import model.attention as mattn

def make_model(args):
    return AdaFormer(args)

class AdaFormer(nn.Module):
    def __init__(self, args, in_chans=3,
                 pspl_depths=6, num_heads=4, query_num=8,
                 window_size=8, mlp_ratio=2., atten_patch_size=1, conv=common.default_conv, **kwargs):

        super(AdaFormer, self).__init__()
        n_feats = args.n_feats
        pspl_depths = args.pspl_depths
        embed_dim = args.embed_dim
        n_resblock = args.n_resblocks // 2
        query_num = args.query_num
        scale = args.scale[0]
        pspl_module = args.pspl_module
        attn_module = args.attn_module
        kernel_size = 3
        act = nn.ReLU(True)

        self.scale = scale
        self.window_size = window_size
        self.n_feats = n_feats
        self.n_resblock = args.n_resblocks
        self.t_eps = args.threshold_eps

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)

        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, sign=1)

        m_head = [conv(in_chans, n_feats, 3)]
        m_shallow_feature_extraction = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblock) # for _ in range(n_resblock)
        ]

        # PSPL = import_module('model.aspformer_block.'+ pspl_module)
        PSPL = getattr(ab, pspl_module)
        Attn = getattr(mattn, attn_module)
        # from model.aspformer_block import pspl_module as PSPL
        m_body = PSPL(
                    dim=embed_dim,
                    n_feats=n_feats,
                    num_heads=num_heads,
                    m_attn=Attn,
                    window_size=window_size,
                    depths=pspl_depths,
                    mlp_ratio=mlp_ratio,
                    query_num=query_num,
                    att_patch_size=atten_patch_size,
                    t_eps=self.t_eps,
                 )

        m_deep_feature_extraction = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblock)
        ]

        m_tail = [
            nn.Conv2d(n_feats, in_chans * scale * scale, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(self.scale)
        ]

        self.head = nn.Sequential(*m_head)
        self.shollow_feature_extraction = nn.Sequential(*m_shallow_feature_extraction)
        self.body = m_body
        self.deep_feature_extraction = nn.Sequential(*m_deep_feature_extraction)
        self.tail = nn.Sequential(*m_tail)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        # _, _, h, w = x.size()
        # ws = 48
        # mod_pad_h = (ws - h % ws) % ws
        # mod_pad_w = (ws - w % ws) % ws
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.shollow_feature_extraction(x)
        res, halting_mask, global_mask_list, local_mask_list = self.body(res)
        res = self.deep_feature_extraction(res)
        res = res + x
        x = self.tail(res)
        x = self.add_mean(x)

        return x[:, :, 0:H * self.scale, 0:W * self.scale], halting_mask, global_mask_list, local_mask_list

    def flops(self, x_size, scale, dict):
        H, W = x_size
        flops = 0
        flops += H * W * 3 * self.n_feats * 9
        ## shallow + deep
        flops += (H * W * self.n_feats * self.n_feats * 9) * 2 * self.n_resblock  ## 和elan在conv上差6g
        # flops += (H * W * self.n_feats) * self.n_resblock # relu

        ## pspl
        body_flops = self.body.flops(x_size, dict)
        flops += body_flops


        # flops
        flops += H * W * self.n_feats * (3 * scale * scale) * 9
        # print("Transformer FLops:", body_flops / 1e9)
        # print("Total FLops:", flops / 1e9)
        flops_dict = {}
        flops_dict['flops'] = flops
        flops_dict['transformer_flops'] = body_flops
        return flops_dict
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))



