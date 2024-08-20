import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from typing import Tuple

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from models_v2 import vit_models, Layer_scale_init_Block, Attention

try:
    from timm.models.vision_transformer import HybridEmbed
except ImportError:
    # for higher version of timm
    from timm.models.vision_transformer_hybrid import HybridEmbed

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import pdb


# 将相对位置编码当成绝对位置编码的表征版本 cope 2d
class CoPE_unit(nn.Module):
    def __init__(self, npos_max, head_dim):
        super(CoPE_2d_unit, self).__init__()
        self.npos_max = npos_max
        self.pos_emb = nn.Parameter(torch.zeros(1, head_dim, npos_max))

    def forward(self, attn_logits):
        # query: (batch_size, heads, seq_len, head_dim)
        # attn_logits: (batch_size, heads, seq_len, seq_len)
        
        # compute positions
        gates = torch.sigmoid(attn_logits)
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.npos_max - 1)
        # interpolate from integer positions
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        logits_int = torch.matmul(query, self.pos_emb)  # (batch_size, heads, seq_len, npos_max)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor
        return logits_ceil * w + logits_floor * (1 - w)

class CoPE2d(nn.Module):
    def __init__(self, npos_max, head_dim, input_dim=196*2, hidd_dim=128, output_dim=196):
        super(CoPE2d, self).__init__()
        self.npos_max = npos_max
        self.pos_emb = nn.Parameter(torch.zeros(1, head_dim, npos_max))
        self.cope_x = CoPE_unit(npos_max, head_dim)
        self.cope_y = CoPE_unit(npos_max, head_dim)
        self.wx = nn.Parameter(torch.tensor(0.5))
        # self.mlp = nn.Sequential(nn.Linear(input_dim, hidd_dim), nn.Dropout(0.1), nn.Linear(hidd_dim, output_dim), nn.ReLU())
    
    def get_pe(self, w_x, w_y):
        B, H, heads, W, _ = w_x.shape
        # print(w_x.shape)
        
        # 使用 reshape 和 permute 重构张量
        # 对 w_x 和 w_y 进行操作，改变它们的形状和维度排列
        pe_x = w_x.permute(0, 1, 3, 2, 4).reshape(B, H*W, heads, W).to(torch.cuda.current_device())
        pe_y = w_y.permute(0, 3, 1, 2, 4).reshape(B, H*W, heads, H).to(torch.cuda.current_device())
        
        # print(pe_x[0, 46, 0, 12])
        
        # pe_x = torch.zeros((B, H*W, heads, W)).to(torch.cuda.current_device())
        # pe_y = torch.zeros((B, H*W, heads, H)).to(torch.cuda.current_device())

        # print(w_x.shape)
        # for i in range(H):
        #     for j in range(W):
        #         pe_x[:, i*W+j] = w_x[:, i, :, j]
        #         pe_y[:, i*W+j] = w_y[:, j, :, i]
        # print(pe_x[0, 46, 0, 12])
        

        # 假设 pe_x 和 pe_y 已经定义并具有形状 (B, H*W, heads, W)
        # 我们需要转换这些张量以适应距离计算
        # 将 heads 维度合并到 batch 维度中，并确保每个 head 的每个点对都能计算距离
        pe_x_flat = pe_x.transpose(1, 2).reshape(B * heads, H * W, W)
        pe_y_flat = pe_y.transpose(1, 2).reshape(B * heads, H * W, H)

        # 使用 torch.cdist 计算点对之间的欧氏距离
        pe_res_x = torch.cdist(pe_x_flat, pe_x_flat, p=2).reshape(B, heads, H*W, H*W).to(torch.cuda.current_device())
        pe_res_y = torch.cdist(pe_y_flat, pe_y_flat, p=2).reshape(B, heads, H*W, H*W).to(torch.cuda.current_device())

        
        # pdist = nn.PairwiseDistance(p=2)
        # pe_res_x = torch.zeros((B, heads, H*W, H*W)).to(torch.cuda.current_device())
        # pe_res_y = torch.zeros((B, heads, H*W, H*W)).to(torch.cuda.current_device())
        # for i in range(H*W):
        #     for j in range(H*W):
        #         pe_res_x[:, :, i, j] = pdist(pe_x[:, i], pe_x[:, j])
        #         pe_res_y[:, :, i, j] = pdist(pe_y[:, i], pe_y[:, j])
        # print(pe_res_x.shape, pe_res_y.shape)
        # for i in range(H):
        #     for j in range(W):
        #         for i1 in range(H):
        #             for j1 in range(W):
        #                 # print(pdist(pe_x[:, i*W+j], pe_x[:, i1*W+j1]).shape)
        #                 pe_res_x[:, :, i*W+j, i1*W+j1] = pdist(pe_x[:, i*W+j], pe_x[:, i1*W+j1])
        #                 pe_res_y[:, :, i*W+j, i1*W+j1] = pdist(pe_y[:, i*W+j], pe_y[:, i1*W+j1])
                
        return pe_res_x, pe_res_y
        
    def forward(self, query, attn_logits):
        # query: (batch_size, heads, seq_len, head_dim)
        # attn_logits: (batch_size, heads, seq_len, seq_len)
        
        # 选出对应的行和列，维度归到 batch 上
        B, heads, seq_len, head_dim = query.shape
        import math
        W = H = int(math.sqrt(seq_len))
        query = query.reshape(B, heads, H, W, head_dim)
        q_x = query.permute(0, 2, 1, 3, 4).reshape(B*H, heads, W, head_dim)
        q_y = query.permute(0, 3, 1, 2, 4).reshape(B*W, heads, H, head_dim)

        attn_logits_x = torch.zeros((B, heads, H, W, W)).to(torch.cuda.current_device())
        for i in range(H):
            for j in range(W):
                attn_logits_x[:, :, i, j] = attn_logits[:, :, W*i+j, W*i:W*i + W]
        
        
        attn_logits_y = torch.zeros((B, heads, W, H, H)).to(torch.cuda.current_device())

        for i in range(W):
            idx = torch.arange(start=i, end=W*H-(W-i) + 1, step=W)
            for j in range(H):
                attn_logits_y[:, :, i, j] = attn_logits[:, :, idx[j], idx]
        
        attn_logits_x = attn_logits_x.reshape(B*H, heads, W, W)
        attn_logits_y = attn_logits_y.reshape(B*W, heads, H, H)
        # print(q_x.shape, attn_logits_x.shape)
        # print(query.shape, attn_logits.shape)
        
        w_x = self.cope_x(q_x, attn_logits_x).reshape(B, H, heads, W, W)
        w_y = self.cope_y(q_y, attn_logits_y).reshape(B, W, heads, H, H)

        # print(w_x.shape, w_y.shape)
        # print(w_x[7, 1, 2, 10, :5])
        
        # calculate positional embedding
        pe_res_x, pe_res_y = self.get_pe(w_x, w_y)
        
        # print(pe_res_x[7, 1, 27, :10])
        
        # combine cope x and y
        # solution 0: add directly
        # w_all = pe_res_x + pe_res_y
        
        # solution 1: use parameter
        w_all = self.wx * pe_res_x + (1  - self.wx) * pe_res_y

        # solution 2: use mlp, concat on the last dim
        # w_all = self.mlp(torch.concat([pe_res_x, pe_res_y], dim=-1))
        
        return w_all


class CoPE2dAttention(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., npos_max=10):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.cope = CoPE2d(npos_max, head_dim)

    def forward(self, x, mask=None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if mask is not None:
            attn.masked_fill_(mask == 0, float('-inf'))

        attn += self.cope(q, attn)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CoPE_2d_Block(Layer_scale_init_Block):
    def __init__(self, *args, **kwargs):
        kwargs["Attention_block"] = CoPE2dAttention
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class cope_vit_models(vit_models):
    """ Vision Transformer with support for patch or hybrid CNN input stage
                           and image relative position encoding
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, self.embed_dim))  # 这里去 override 使得 num_patch + 1
        trunc_normal_(self.pos_embed, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]


@register_model
def cope_2d_mixed_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=CoPE_2d_Block, Attention_block=CoPE2dAttention,
        rope_theta=10.0, rope_mixed=True, **kwargs)
    model.default_cfg = _cfg()
    return model
