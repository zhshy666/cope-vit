from inspect import isclass
import math
from cmath import nan, sqrt
from functools import partial
from typing import Tuple

import cv2
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import Mlp, PatchEmbed, _cfg

from models_v2 import Attention, Layer_scale_init_Block, vit_models

try:
    from timm.models.vision_transformer import HybridEmbed
except ImportError:
    # for higher version of timm
    from timm.models.vision_transformer_hybrid import HybridEmbed

import pdb

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


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
    
    def get_pos_embed(self):
        return self.pos_emb

    def forward(self, q, k, v, is_cope_k=1):
        # print(self.w_k.grad)
        # query: (batch_size, heads, seq_len, head_dim)
        # attn_logits: (batch_size, heads, seq_len, seq_len)
        # compute positions
        
        if self.mode == 0:
            attn_logits = (q * self.scale) @ k.transpose(-2, -1)
        elif self.mode == 1:
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
        # if is_cope_k == 2:
        #     print(attn_logits.shape, self.pos_emb.shape, q.shape)
        #     # torch.Size([256, 6, 2, 2]) torch.Size([1, 64, 10]) torch.Size([256, 6, 2, 64])
        #     logits_int = torch.matmul(attn_logits, self.pos_emb)
        #     exit()
        # else:
        logits_int = torch.matmul(q if is_cope_k else k, self.pos_emb)  # (batch_size, heads, seq_len, npos_max)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor
        return logits_ceil * w + logits_floor * (1 - w)
    
    
class CoPE2d_v2(nn.Module):
    def __init__(self, npos_max, head_dim, scale, mode, dwt, num_heads, num_patches, img_size):
        super(CoPE2d_v2, self).__init__()
        self.npos_max = npos_max
        self.cope_x = CoPE_2d_unit(npos_max, head_dim, scale, mode)
        self.cope_y = CoPE_2d_unit(npos_max, head_dim, scale, mode)
        self.dwt = dwt
        # print((img_size // 2)**2, num_patches * num_heads * head_dim)
        # self.embed_dwt_x = nn.Sequential(
        #     nn.Linear((img_size // 2)**2, num_patches * num_heads * head_dim),
        #     nn.ReLU()
        # )
        # self.embed_dwt_y = nn.Sequential(
        #     nn.Linear((img_size // 2)**2, num_patches * num_heads * head_dim),
        #     nn.ReLU()
        # )
        
        # conv
        # self.embed_dwt_x = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=num_heads * head_dim, dilation=2, kernel_size=8),
        #     # nn.ReLU(),
        #     ComplexGaborLayer(omega0=30),
        #     nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        # )
        # self.embed_dwt_y = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=num_heads * head_dim, dilation=2, kernel_size=8),
        #     # nn.ReLU(),
        #     ComplexGaborLayer(omega0=30),
        #     nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        # )
        
        self.norm = nn.LayerNorm(head_dim)
    
    def get_pos_embed(self):
        return self.cope_x.get_pos_embed()

    def forward(self, query, key, value, is_cope_k, dwt_x=None, dwt_y=None):
        # query: (batch_size, heads, seq_len, head_dim)
        
        # 选出对应的行和列，维度归到 batch 上
        B, heads, seq_len, head_dim = query.shape
        import math
        W = H = int(math.sqrt(seq_len))
        query = query.reshape(B, heads, H, W, head_dim)
        key = key.reshape(B, heads, H, W, head_dim)
        value = value.reshape(B, heads, H, W, head_dim)
        # [batch, heads, patch_h, patch_w, hidden], [batch, h/2, w/2]
        # print(query.shape, dwt_x.shape)
        
        q_x = query.permute(0, 2, 1, 3, 4).reshape(B*H, heads, W, head_dim)
        q_y = query.permute(0, 3, 1, 2, 4).reshape(B*W, heads, H, head_dim)
        
        # query 拆两半用于记录 x 和 y 轴位置信息
        # query_x = q_x[:, :, :, :head_dim // 2]
        # query_y = q_y[:, :, :, head_dim // 2:]
        
        
        # todo: x 和 y 轴分别小波变换
        # 问题1：能否在 x 进行 patch embedding 之后再进行小波变换，也就是小波变换怎么用的问题
        # 问题2：能否将小波变换后的结果和原本的 key 进行一些操作
        #       （例如 concat，乘积，FN 等）进行维度映射，qkv 的维度需要保持一致
        # 问题3：能否直接对 key 进行小波变换？这样其实没有显式用到原始图像的信息
        
        if self.dwt:
            # k_x = self.embed_dwt_x(dwt_x.reshape(B, -1)).reshape(B, heads, H, W, head_dim)
            # k_y = self.embed_dwt_y(dwt_y.reshape(B, -1)).reshape(B, heads, H, W, head_dim)
            
            # conv
            k_x = self.embed_dwt_x(dwt_x.unsqueeze(1)).reshape(B, heads, head_dim, H, W).permute(0, 1, 3, 4, 2)
            # print(k_x.shape, heads, head_dim, H, W)
            k_y = self.embed_dwt_y(dwt_y.unsqueeze(1)).reshape(B, heads, head_dim, H, W).permute(0, 1, 3, 4, 2)
            
            # use directly
            # k_x = k_x.permute(0, 2, 1, 3, 4).reshape(B*H, heads, W, head_dim)
            # k_y = k_y.permute(0, 3, 1, 2, 4).reshape(B*W, heads, H, head_dim)
            
            # add
            k_x = (self.norm(key) + self.norm(k_x)).permute(0, 2, 1, 3, 4).reshape(B*H, heads, W, head_dim)
            k_y = (self.norm(key) + self.norm(k_y)).permute(0, 3, 1, 2, 4).reshape(B*W, heads, H, head_dim)
            
            # no norm
            # k_x = (key + k_x).permute(0, 2, 1, 3, 4).reshape(B*H, heads, W, head_dim)
            # k_y = (key + k_y).permute(0, 3, 1, 2, 4).reshape(B*W, heads, H, head_dim)
            
        else:
            k_x = key.permute(0, 2, 1, 3, 4).reshape(B*H, heads, W, head_dim)
            k_y = key.permute(0, 3, 1, 2, 4).reshape(B*W, heads, H, head_dim)
        
        v_x = value.permute(0, 2, 1, 3, 4).reshape(B*H, heads, W, head_dim)
        v_y = value.permute(0, 3, 1, 2, 4).reshape(B*W, heads, H, head_dim)
        
        # 分割 q k 并计算新的 att
        w_x = self.cope_x(q_x, k_x, v_x, is_cope_k)  # [1792, 6, 14, 14]
        w_y = self.cope_y(q_y, k_y, v_y, is_cope_k)
        
        w_x = w_x.reshape(B, H, heads, W, H).permute(0, 2, 1, 3, 4).reshape(B * heads, H * W, H)
        w_y = w_y.reshape(B, W, heads, H, W).permute(0, 2, 3, 1, 4).reshape(B * heads, H * W, W)

        # 使用 torch.cdist 计算点对之间的欧氏距离
        pe_res_x = torch.cdist(w_x, w_x, p=2).reshape(B, heads, H*W, H*W).to(torch.cuda.current_device())
        pe_res_y = torch.cdist(w_y, w_y, p=2).reshape(B, heads, H*W, H*W).to(torch.cuda.current_device())
        
        # return pe_res_x + pe_res_y
        # 调整为 (num_patch + 1)
        cope = torch.zeros(B, heads, W * H + 1, W * H + 1, requires_grad=True).to(torch.cuda.current_device())
        cope[:, :, 1:, 1:] = pe_res_x + pe_res_y
        # print(cope.shape)
    
        return cope


class CoPE2dAttention_v2(Attention):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 npos_max=10, cope_k=1, cope_q=0, cope_v=0, mode=0, dwt=0, num_patches=0, img_size=224):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, 
                         npos_max, cope_k, cope_q, cope_v, mode, dwt, num_patches, img_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.cope = CoPE2d_v2(npos_max, head_dim, self.scale, mode, dwt, num_heads, num_patches, img_size)
        self.cope_k = cope_k
        self.cope_q = cope_q
        self.cope_v = cope_v
        # self.act = ComplexGaborLayer(omega0=30)

    def forward(self, x, mask=None, dwt_x=None, dwt_y=None, blk_num=0, bs=0):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if mask is not None:
            attn.masked_fill_(mask == 0, float('-inf'))

        # attn += self.cope(q, k, v)
        temp = 0
        if self.cope_k:
            temp = self.cope(q[:, :, 1:], k[:, :, 1:], v[:, :, 1:], dwt_x=dwt_x, dwt_y=dwt_y, is_cope_k=0)
            attn += temp
        if self.cope_q:
            temp = self.cope(q[:, :, 1:], k[:, :, 1:], v[:, :, 1:], dwt_x=dwt_x, dwt_y=dwt_y, is_cope_k=0)
            attn += temp
            if self.cope_k:
                # pe = self.cope.get_pos_embed()
                attn += temp @ temp

        # 加小波激活函数
        # print(attn.shape)
        # norm = nn.LayerNorm(N).to(torch.cuda.current_device())
        # attn = norm(attn)
        
        attn = attn.softmax(dim=-1)
        # attn = self.act(
        #         attn.reshape(B, self.num_heads, -1, 1)
        #     ).reshape(B, self.num_heads, N, N).float()
        attn = self.attn_drop(attn)

        if self.cope_v:
            # not test here
            temp = self.cope(q[:, :, 1:], k[:, :, 1:], v[:, :, 1:], dwt_x=dwt_x, dwt_y=dwt_y, is_cope_k=0)
            # print(temp.shape, q.shape, attn.shape)
            # attn += torch.matmul(q, temp)
            temp = q.permute(0, 1, 3, 2) @ temp
            # print(temp.shape)
            x = (attn @ v + temp.permute(0, 1, 3, 2)).transpose(1, 2).reshape(B, N, C)
            # print(x.shape)
            # exit()
        else:
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # np.save('../draw/batch' + str(bs) + '+layer' + str(blk_num) + '.npy', attn.cpu().numpy())
        # np.save('../draw-bias/batch' + str(bs) + '+layer' + str(blk_num) + '.npy', temp.cpu().numpy())
        
        # attn = self.act(
        #         attn.reshape(B, N, C, 1).permute(0, 2, 1, 3)
        #     ).reshape(B, C, N).permute(0, 2, 1).float()
        # print(x.shape)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CoPE_2d_Block(Layer_scale_init_Block):
    def __init__(self, *args, **kwargs):
        kwargs["Attention_block"] = CoPE2dAttention_v2
        super().__init__(*args, **kwargs)

    def forward(self, x, dwt_x=None, dwt_y=None, blk_num=0, bs=0):
        x = x + self.drop_path(self.attn(self.norm1(x), dwt_x=dwt_x, dwt_y=dwt_y, 
                                         blk_num=blk_num, bs=bs))
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

    def forward_features(self, x, bs=0):
        B, _, H, W = x.shape
        
        # here: 在图像编码到隐空间前进行小波变换
        images = x[:, 0].reshape(B, H, W).cpu().numpy()
        coeffs = pywt.dwt2(images, 'haar')
        # 从结果中获取近似子带和细节子带
        cA, (cH, cV, cD) = coeffs
        # cV = torch.tensor(cV).to(torch.cuda.current_device())
        # cD = torch.tensor(cD).to(torch.cuda.current_device())
        # 小波变换代码结束
        
        x = self.patch_embed(x)  # [batch, patch, hidden]
        # print(x.shape)
        
        # here: 在图像编码到隐空间后进行小波变换
        # images = x[:, 0].reshape(B, H, W).cpu().numpy()

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)

        i = 0
        for blk in self.blocks:
            x = blk(x, 
                    dwt_x=torch.tensor(cV).to(torch.cuda.current_device()), 
                    dwt_y=torch.tensor(cH).to(torch.cuda.current_device()),
                    blk_num=i,
                    bs=bs,
                )
            i += 1

        x = self.norm(x)
        return x[:, 0]

# from: https://github.com/294coder/Efficient-MIF/blob/main-release/model/module/fe_block.py
class ComplexGaborLayer(nn.Module):
    '''
        Complex Gabor nonlinearity 

        Inputs:
            input: Input features
            omega0: Frequency of Gabor sinusoid term
            sigma0: Scaling of Gabor Gaussian term
            trainable: If True, omega and sigma are trainable parameters
    '''

    def __init__(self, omega0=30.0, sigma0=10.0, trainable=True):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0

        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

    def forward(self, input):
        input = input.permute(0, -2, -1, 1)

        omega = self.omega_0 * input
        scale = self.scale_0 * input
        # return torch.exp(1j * omega - scale.abs().square())
        return (torch.exp(1j * omega - scale.abs().square())).permute(0, -1, 1, 2).float()

# mode = 0
@register_model
def cope_2d_v2_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=CoPE_2d_Block, 
        Attention_block=CoPE2dAttention_v2,
        rope_theta=10.0, rope_mixed=True,
        cope_k=1, mode=0, num_patches=img_size//16,
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def cope_2d_v2_deit_small_patch4_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cope_vit_models(
        img_size = img_size, patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=CoPE_2d_Block, 
        Attention_block=CoPE2dAttention_v2,
        rope_theta=10.0, rope_mixed=True,
        cope_k=1, mode=0, num_patches=img_size//4,
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def cope_2d_v2_deit_small_patch4_LS_q(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cope_vit_models(
        img_size = img_size, patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=CoPE_2d_Block, 
        Attention_block=CoPE2dAttention_v2,
        rope_theta=10.0, rope_mixed=True, 
        cope_k=0,cope_q=1, mode=0, num_patches=img_size//4,
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def cope_2d_v2_deit_small_patch4_LS_v(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cope_vit_models(
        img_size = img_size, patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=CoPE_2d_Block, 
        Attention_block=CoPE2dAttention_v2,
        rope_theta=10.0, rope_mixed=True, 
        cope_k=0,cope_v=1, mode=0, num_patches=img_size//4,
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def cope_2d_v2_deit_small_patch4_LS_qk(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cope_vit_models(
        img_size = img_size, patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=CoPE_2d_Block, 
        Attention_block=CoPE2dAttention_v2,
        rope_theta=10.0, rope_mixed=True, 
        cope_k=1,cope_q=1, mode=0, num_patches=img_size//4,
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def cope_2d_v2_deit_small_patch4_LS_qv(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cope_vit_models(
        img_size = img_size, patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=CoPE_2d_Block, 
        Attention_block=CoPE2dAttention_v2,
        rope_theta=10.0, rope_mixed=True, 
        cope_k=0,cope_v=1, cope_q=1, mode=0, num_patches=img_size//4,
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def cope_2d_v2_deit_small_patch4_LS_kv(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cope_vit_models(
        img_size = img_size, patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=CoPE_2d_Block, 
        Attention_block=CoPE2dAttention_v2,
        rope_theta=10.0, rope_mixed=True, 
        cope_k=1,cope_v=1, mode=0, num_patches=img_size//4,
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def cope_2d_v2_deit_small_patch4_LS_qkv(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cope_vit_models(
        img_size = img_size, patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=CoPE_2d_Block, 
        Attention_block=CoPE2dAttention_v2,
        rope_theta=10.0, rope_mixed=True, 
        cope_k=1,cope_v=1, cope_q=1, mode=0, num_patches=img_size//4,
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def cope_2d_v2_deit_dwt_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=CoPE_2d_Block, 
        Attention_block=CoPE2dAttention_v2,
        rope_theta=10.0, rope_mixed=True,
        cope_k=1, mode=0, dwt=1, num_patches=img_size//16,
        **kwargs)
    model.default_cfg = _cfg()
    return model

# mode = 1
@register_model
def cope_2d_v2_sep_keys_deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=CoPE_2d_Block, 
        Attention_block=CoPE2dAttention_v2,
        rope_theta=10.0, rope_mixed=True, 
        cope_k=1, mode=1, num_patches=img_size//16,
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def cope_2d_v2_sep_keys_deit_small_patch4_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cope_vit_models(
        img_size = img_size, patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=CoPE_2d_Block, 
        Attention_block=CoPE2dAttention_v2,
        rope_theta=10.0, rope_mixed=True, 
        cope_k=1, mode=1, num_patches=img_size//4,
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def cope_2d_v2_sep_keys_deit_small_patch4_LS_q(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cope_vit_models(
        img_size = img_size, patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=CoPE_2d_Block, 
        Attention_block=CoPE2dAttention_v2,
        rope_theta=10.0, rope_mixed=True, 
        cope_k=0, cope_q=1, mode=1, num_patches=img_size//4,
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def cope_2d_v2_sep_keys_deit_dwt_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = cope_vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), block_layers=CoPE_2d_Block, 
        Attention_block=CoPE2dAttention_v2,
        rope_theta=10.0, rope_mixed=True, 
        cope_k=1, mode=1, dwt=1, num_patches=img_size//16,
        **kwargs)
    model.default_cfg = _cfg()
    return model

# swin transformer