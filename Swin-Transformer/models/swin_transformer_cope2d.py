"""
This code was originally obtained from:
https://github.com/microsoft/Swin-Transformer
and
https://github.com/meta-llama/codellama/blob/main/llama/model.py
"""
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from typing import Any, Optional, Tuple
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .swin_transformer import SwinTransformer, Mlp, SwinTransformerBlock, WindowAttention, BasicLayer
from .swin_transformer import window_partition, window_reverse, PatchMerging
from .swin_transformer import WindowProcess, WindowProcessReverse

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

    def forward(self, query, key, value, attn_logits, is_cope_k=1):
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
        
        return pe_res_x + pe_res_y


class CoPE2dWindowAttention(WindowAttention):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., npos_max=10, cope_k=1, cope_q=0, cope_v=0):

        super().__init__(dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.cope = CoPE2d_v2(npos_max, head_dim, self.scale)
        self.cope_k = cope_k
        self.cope_q = cope_q
        self.cope_v = cope_v
        
    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if mask is not None:
            attn.masked_fill_(mask == 0, float('-inf'))

        if self.cope_k:
            attn += self.cope(q, k, v, attn)
        if self.cope_q:
            attn += self.cope(q, k, v, attn, is_cope_k=0)
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


class CoPE2dSwinTransformerBlock(SwinTransformerBlock):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False,
                 npos_max=10, cope_k=1, cope_q=0, cope_v=0):
        super().__init__(
            dim, input_resolution, num_heads, window_size=window_size, shift_size=shift_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
            attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer,
            fused_window_process=fused_window_process
        )

        self.attn = CoPE2dWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            npos_max=npos_max, cope_k=cope_k, cope_q=cope_q, cope_v=cope_v
        )


class CoPE2dBasicLayer(BasicLayer):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False,
                 npos_max=10, cope_k=1, cope_q=0, cope_v=0):

        super().__init__(
            dim=dim, input_resolution=input_resolution, depth=depth, num_heads=num_heads,
            window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer,
            downsample=downsample, use_checkpoint=use_checkpoint, fused_window_process=fused_window_process
        )

        # build blocks
        self.blocks = nn.ModuleList([
            CoPE2dSwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                fused_window_process=fused_window_process,
                npos_max=npos_max, cope_k=cope_k, cope_q=cope_q, cope_v=cope_v
            )
            for i in range(depth)])


class CoPE2dSwinTransformer(SwinTransformer):
    
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False,
                 npos_max=10, cope_k=1, cope_q=0, cope_v=0,
                 **kwargs):

        super().__init__(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads,
            window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer, ape=ape,
            patch_norm=patch_norm, use_checkpoint=use_checkpoint, **kwargs
        )

        # absolute position embedding
        self.ape = False
        self.absolute_pos_embed = None

        patches_resolution = self.patch_embed.patches_resolution

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = CoPE2dBasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                fused_window_process=fused_window_process,
                npos_max=npos_max, cope_k=cope_k, cope_q=cope_q, cope_v=cope_v
            )
            self.layers.append(layer)

        self.apply(self._init_weights)