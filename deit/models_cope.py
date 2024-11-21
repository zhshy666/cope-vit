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

# mode 0: default, 1: sep-keys, 2: val-gates
# cope_q, cope_k, cope_v

class CoPE(nn.Module):
    def __init__(self, npos_max, head_dim, scale, mode=1):
        super(CoPE, self).__init__()
        self.npos_max = npos_max
        self.pos_emb = nn.Parameter(torch.zeros(1, head_dim, npos_max))
        self.w_k = nn.Parameter(torch.zeros(1, head_dim, head_dim))
        # self.w_q = nn.Parameter(torch.zeros(1, head_dim, head_dim))
        self.mode = mode
        self.scale = scale
    
    def get_pos_embed(self):
        return self.pos_emb

    def forward(self, query, attn_logits, key, value, is_cope_k=1):
        # query: (batch_size, heads, seq_len, head_dim)
        # attn_logits: (batch_size, heads, seq_len, seq_len)
        # key: (batch_size, heads, seq_len, head_dim)

        # compute positions
        if self.mode == 1:
            key = key @ self.w_k
            attn_logits = (query * self.scale) @ key.transpose(-2, -1)
        elif self.mode == 11:
            if is_cope_k:
                key[:, :, 1:] = key[:, :, 1:] @ self.w_k
            else:
                query[:, :, 1:] = query[:, :, 1:] @ self.w_q
            attn_logits = (query * self.scale) @ key.transpose(-2, -1)
        elif self.mode == 2:
            attn_logits = (query * self.scale) @ value.transpose(-2, -1)
            
        gates = torch.sigmoid(attn_logits)
        # gates = torch.ones_like(gates)
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.npos_max - 1)
        # interpolate from integer positions
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        
        logits_int = torch.matmul(query if is_cope_k else key, self.pos_emb)  # (batch_size, heads, seq_len, npos_max)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor
        return logits_ceil * w + logits_floor * (1 - w)


class CoPEAttention(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., npos_max=10, cope_k=1, cope_q=0, cope_v=0, mode=0):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.cope = CoPE(npos_max, head_dim, self.scale, mode)
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

        if self.cope_k:
            attn += self.cope(q, attn, k, v)
        if self.cope_q:
            attn += self.cope(q, attn, k, v, is_cope_k=0)
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

class CoPE_Block(Layer_scale_init_Block):
    def __init__(self, *args, **kwargs):
        kwargs["Attention_block"] = CoPEAttention
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

# default: cope on k

@register_model
def cope_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=CoPE_Block, Attention_block=CoPEAttention(dim=384, mode=0),
        rope_theta=10.0, rope_mixed=True, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def cope_sep_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=CoPE_Block, Attention_block=CoPEAttention(dim=384, mode=1),
        rope_theta=10.0, rope_mixed=True, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def cope_val_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=CoPE_Block, Attention_block=CoPEAttention(dim=384, mode=2),
        rope_theta=10.0, rope_mixed=True, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def cope_deit_small_patch16_LS_q(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=CoPE_Block, Attention_block=CoPEAttention(dim=384, cope_q=1, cope_k=0),
        rope_theta=10.0, rope_mixed=True, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def cope_deit_small_patch16_LS_qk(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=CoPE_Block, Attention_block=CoPEAttention(dim=384, cope_q=1),
        rope_theta=10.0, rope_mixed=True, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def cope_deit_small_patch16_LS_kv(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=CoPE_Block, Attention_block=CoPEAttention(dim=384, cope_v=1),
        rope_theta=10.0, rope_mixed=True, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def cope_deit_small_patch16_LS_qkv(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=CoPE_Block, Attention_block=CoPEAttention(dim=384, cope_q=1, cope_v=1),
        rope_theta=10.0, rope_mixed=True, **kwargs)
    model.default_cfg = _cfg()
    return model