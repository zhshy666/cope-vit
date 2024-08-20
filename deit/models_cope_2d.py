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

# 将 query 拆成两部分分别学习 x 和 y 轴上的位置编码
class CoPE_2d_unit(nn.Module):
    def __init__(self, npos_max, head_dim, scale, mode=0):
        super(CoPE_2d_unit, self).__init__()
        self.npos_max = npos_max
        self.pos_emb = nn.Parameter(torch.zeros(1, head_dim, npos_max))
        self.w_k = nn.Parameter(torch.zeros(1, head_dim, head_dim))
        self.mode = mode
        self.scale = scale
        # 初始化 pos_emb 使用正态分布
        trunc_normal_(self.pos_emb, std=.02)
        # nn.init.normal_(self.pos_emb, mean=0.0, std=0.01)

    def forward(self, q, k, v, attn_logits, is_cope_k=1):
        # print(self.w_k.grad)
        # query: (batch_size, heads, seq_len, head_dim)
        # attn_logits: (batch_size, heads, seq_len, seq_len)
        # compute positions
        
        if self.mode == 1:
            k = k @ self.w_k
            attn_logits = (q * self.scale) @ k.transpose(-2, -1)
        elif self.mode == 2:
            attn_logits = (q * self.scale) @ v.transpose(-2, -1)
        gates = torch.sigmoid(attn_logits)
        
        # print(attn_logits.grad)
        # print(gates.shape)  # [1792, 6, 14, 14]
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.npos_max - 1)
        # interpolate from integer positions
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        logits_int = torch.matmul(q if is_cope_k else k, self.pos_emb)  # (batch_size, heads, seq_len, npos_max)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor
        return logits_ceil * w + logits_floor * (1 - w)
    
    
class CoPE2d_v2(nn.Module):
    def __init__(self, npos_max, head_dim, scale):
        super(CoPE2d_v2, self).__init__()
        self.npos_max = npos_max
        self.cope_x = CoPE_2d_unit(npos_max, head_dim, scale)
        self.cope_y = CoPE_2d_unit(npos_max, head_dim, scale)

    def forward(self, query, key, value, attn_logits, is_cope_k):
        # query: (batch_size, heads, seq_len, head_dim)
        
        # 选出对应的行和列，维度归到 batch 上
        B, heads, seq_len, head_dim = query.shape
        import math
        W = H = int(math.sqrt(seq_len))
        query = query.reshape(B, heads, H, W, head_dim)
        key = key.reshape(B, heads, H, W, head_dim)
        value = value.reshape(B, heads, H, W, head_dim)
        # print(query_x.shape)  # [128, 6, 14, 14, 32]
        
        q_x = query.permute(0, 2, 1, 3, 4).reshape(B*H, heads, W, head_dim)
        q_y = query.permute(0, 3, 1, 2, 4).reshape(B*W, heads, H, head_dim)
        
        # query 拆两半用于记录 x 和 y 轴位置信息
        # query_x = q_x[:, :, :, :head_dim // 2]
        # query_y = q_y[:, :, :, head_dim // 2:]
        
        k_x = key.permute(0, 2, 1, 3, 4).reshape(B*H, heads, W, head_dim)
        k_y = key.permute(0, 3, 1, 2, 4).reshape(B*W, heads, H, head_dim)
        
        v_x = value.permute(0, 2, 1, 3, 4).reshape(B*H, heads, W, head_dim)
        v_y = value.permute(0, 3, 1, 2, 4).reshape(B*W, heads, H, head_dim)
        
        # 分割 q k 并计算新的 att
        w_x = self.cope_x(q_x, k_x, v_x, attn_logits, is_cope_k)  # [1792, 6, 14, 14]
        
        w_y = self.cope_y(q_y, k_y, v_y, attn_logits, is_cope_k)
        
        w_x = w_x.reshape(B, H, heads, W, H).permute(0, 2, 1, 3, 4).reshape(B * heads, H * W, H)
        w_y = w_y.reshape(B, W, heads, H, W).permute(0, 2, 3, 1, 4).reshape(B * heads, H * W, H)

        # 使用 torch.cdist 计算点对之间的欧氏距离
        pe_res_x = torch.cdist(w_x, w_x, p=2).reshape(B, heads, H*W, H*W).to(torch.cuda.current_device())
        pe_res_y = torch.cdist(w_y, w_y, p=2).reshape(B, heads, H*W, H*W).to(torch.cuda.current_device())
        
        # return pe_res_x + pe_res_y
        # 调整为 (num_patch + 1)
        cope = torch.zeros(B, heads, W * H + 1, W * H + 1, requires_grad=True).to(torch.cuda.current_device())
        cope[:, :, 1:, 1:] = pe_res_x + pe_res_y
    
        return cope


class CoPE2dAttention_v2(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., npos_max=10, cope_k=1, cope_q=0, cope_v=0):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.cope = CoPE2d_v2(npos_max, head_dim, self.scale)
        self.cope_k = cope_k
        self.cope_q = cope_q
        self.cope_v = cope_v

    def forward(self, x, mask=None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if mask is not None:
            attn.masked_fill_(mask == 0, float('-inf'))

        # attn += self.cope(q, k, v)
        if self.cope_k:
            attn += self.cope(q[:, :, 1:], k[:, :, 1:], v[:, :, 1:], attn[:, :, 1:, 1:])
        if self.cope_q:
            attn += self.cope(q[:, :, 1:], k[:, :, 1:], v[:, :, 1:], attn[:, :, 1:, 1:], is_cope_k=0)
            if self.cope_k:
                pe = self.cope.get_pos_embed()
                attn += pe @ pe

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if self.cope_v:
            # not test here
            attn += attn * self.cope.get_pos_embed()

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CoPE_2d_Block(Layer_scale_init_Block):
    def __init__(self, *args, **kwargs):
        kwargs["Attention_block"] = CoPE2dAttention_v2
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
def cope_2d_v2_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=CoPE_2d_Block, Attention_block=CoPE2dAttention_v2(cope_k=1),
        rope_theta=10.0, rope_mixed=True, **kwargs)
    model.default_cfg = _cfg()
    return model

# swin transformer