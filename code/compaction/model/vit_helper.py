# 这个文件里面实现不同的 transformer attention

from torch._C import has_cudnn
from einops import rearrange, repeat
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
from torch import einsum
from functools import partial
from timm.models.layers import DropPath

def qkv_attn(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out

class BasicAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3,
            self.num_heads,
            C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn

class TrajectoryAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, seq_len=196, num_frames=8):
        B, N, C = x.shape
        P = seq_len  # seq_len 是指 spatial 上的 token 数目
        F = num_frames
        h = self.num_heads

        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Reshape: 'b n (h d) -> (b h) n d'
        # 拆成不同的 heads, n 就是 token 的数目
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # remove CLS token from q, k, v
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(
            lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let CLS token attend to key / values of all patches across time and space
        cls_out = qkv_attn(cls_q * self.scale, k, v) # (b*h, 1, d) att. (b*h, N, d)
        cls_out = rearrange(cls_out, f'(b h) f d -> b f (h d)', f=1, h=h)

        # Using full attention
        q_dot_k = q_ @ k_.transpose(-2, -1) # (b*h, N, N)
        q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # N 拆分成 temporal 和 spatil, 最后一维就是 spatial
        space_attn = (self.scale * q_dot_k).softmax(dim=-1)
        attn = self.attn_drop(space_attn)
        v_ = rearrange(v_, 'b (f n) d -> b f n d', f=F, n=P) # spatial, temporal 拆分
        x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_) # (b*h, N, T, d)

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # (b, N, T, h*d)
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # (b, T, S, T, d)
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # (b, S, d, T) # y_stt
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # (b, N, d)
        q2 = self.proj_q(x_diag)
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # (b, N, T, h*d)
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h) # (b, h, N, d)
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2)) # (b, h, N, T, d)
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)
        attn = attn.softmax(dim=-1)

        x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)') # (b, N, d)

        # concat back the cls token
        x = torch.cat((cls_out, x), dim=1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class CrossTrajectoryAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, num_frames=8, key_mask=None):
        # xq: b, l, d
        # xk: b, m, d
        B, M, _ = xk.shape
        F = num_frames
        h = self.num_heads
        ks = M // F

        q = self.q(xq)
        k,v = self.kv(xk).chunk(2, dim=-1)

        # 拆成不同的 heads, n 就是 token 的数目
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        q_dot_k = q @ k.transpose(-2, -1) # (b*h, l, m)
        q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # 把 m 个 token 拆成 temporal 和 spatial
        space_attn = (self.scale * q_dot_k).softmax(dim=-1) # b, l, T, m
        attn = self.attn_drop(space_attn)
        v = rearrange(v, 'b (f n) d -> b f n d', f=F, n=ks) # 将 v 的 spatial, temporal 拆分，b, T, m, d
        # b, l, T, m @ b, T, m, d  -> b, l, T, d
        x = torch.einsum('b q f n, b f n d -> b q f d', attn, v) # (b*h, l, T, d)

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # (b, N, T, h*d)
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # (b, T, S, T, d)
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # (b, S, d, T) # y_stt
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # (b, N, d)
        q2 = self.proj_q(x_diag)
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # (b, N, T, h*d)
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h) # (b, h, N, d)
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2)) # (b, h, N, T, d)
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)
        attn = attn.softmax(dim=-1)

        x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)') # (b, N, d)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class CrossTrajectoryAttentionMask(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, num_frames=8, key_mask=None):
        # xq: b, l, d
        # xk: b, m, d
        # key_mask: b, t*m
        B, M, _ = xk.shape
        F = num_frames
        h = self.num_heads
        ks = M // F

        q = self.q(xq)
        k,v = self.kv(xk).chunk(2, dim=-1)

        # 拆成不同的 heads, n 就是 token 的数目
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        q_dot_k = q @ k.transpose(-2, -1) # (b*h, l, m)

        key_mask = key_mask.expand(self.num_heads, -1) # (b*h, m)
        q_dot_k = q_dot_k.masked_fill(key_mask.unsqueeze(1), float('-inf'))

        q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)  # 把 m 个 token 拆成 temporal 和 spatial
        space_attn = (self.scale * q_dot_k).softmax(dim=-1) # b, l, T, m
        attn = self.attn_drop(space_attn)
        v = rearrange(v, 'b (f n) d -> b f n d', f=F, n=ks) # 将 v 的 spatial, temporal 拆分，b, T, m, d
        # b, l, T, m @ b, T, m, d  -> b, l, T, d
        x = torch.einsum('b q f n, b f n d -> b q f d', attn, v) # (b*h, l, T, d)

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B) # (b, N, T, h*d)
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F) # (b, T, S, T, d)
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2) # (b, S, d, T) # y_stt
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F) # (b, N, d)
        q2 = self.proj_q(x_diag)
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1) # (b, N, T, h*d)
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h) # (b, h, N, d)
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2)) # (b, h, N, T, d)
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)
        attn = attn.softmax(dim=-1)

        x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)') # (b, N, d)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

def get_attention_module(
    dim=768, num_heads=12, qkv_bias=False,
    attn_drop=0., proj_drop=0.,
):
    attn = TrajectoryAttention(
        dim, num_heads=num_heads, qkv_bias=qkv_bias,
        attn_drop=attn_drop, proj_drop=proj_drop,)
    return attn

class Block(nn.Module):

    def __init__(
            self, dim=768, num_heads=12, attn_type='trajectory',
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = get_attention_module(
            dim=dim, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, seq_len=196, num_frames=8):
        x = x + self.drop_path(
            self.attn(
                self.norm1(x),
                seq_len=seq_len,
                num_frames=num_frames,
            )[0]
        )
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

###############################################################################
#                                  add by lxs                                 #
###############################################################################

class HOReasonBlock(nn.Module):
    def __init__(
            self, dim=768, num_heads=12,
            mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            attn_type='traj'
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if attn_type == 'traj':
            self.attn = CrossTrajectoryAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        elif attn_type == 'traj_mask':
            self.attn = CrossTrajectoryAttentionMask(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        else:
            raise NotImplementedError("No such attn type for HO reason")
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, xq, xk, num_frames=8, key_mask=None):
        xq = xq + self.drop_path(
            self.attn(
                self.norm1(xq), self.norm1(xk),
                num_frames=num_frames,
                key_mask=key_mask
            )[0]
        )
        xq = xq + self.drop_path(self.mlp(self.norm2(xq)))
        return xq

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim=768, num_heads=12,
                 mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = BasicAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(
            self.attn(
                self.norm1(x)))[0]
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None,
        out_features=None, act_layer=nn.GELU, drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# TODO: 添加 concat 的版本
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, attach='sum'):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.attach = attach

    def forward(self, x):
        b, d, t, h, w = x.shape
        # mask: (b, h, w), with 1 on padded pixels
        not_mask = torch.ones(b, t, h, w, device=x.device)
        t_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize: # 2*pi*w/W
            eps = 1e-6
            t_embed = t_embed / (t_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device) # [1,...,512]
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats) # 10000 ^ (2 * [0, 0, 1, 1, 2, 2, ... 255, 255])

        pos_t = t_embed[:, :, :, :, None] / dim_t # b, t, h, w, d, 有一个广播
        pos_x = x_embed[:, :, :, :, None] / dim_t # b, t, h, w, d
        pos_y = y_embed[:, :, :, :, None] / dim_t # b, t, h, w, d

        pos_t = torch.stack((pos_t[:, :, :, :, 0::2].sin(), pos_t[:, :, :, :, 1::2].cos()), dim=5).flatten(4) # 拼接到第5维，然后从第 4 维 flatten, 最终只有 5 个维度
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos = torch.cat((pos_t, pos_y, pos_x), dim=4).permute(0, 4, 1, 2, 3) # b, c, t, h, w

        # if self.attach == 'sum':
            # x = x + pos
        # else:
            # x = torch.cat([x, pos], dim=1)
        
        return pos

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256, attach='sum'):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.temp_embed = nn.Embedding(20, num_pos_feats)

        self.pe_attach = attach
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        nn.init.uniform_(self.temp_embed.weight)

    def forward(self, x):
        # x: b, c, t, h, w
        b, d, t, h, w = x.shape
        i = torch.arange(w, device=x.device) # w, c
        j = torch.arange(h, device=x.device) # h, c
        k = torch.arange(t, device=x.device) # t, c
        x_emb = self.col_embed(i).unsqueeze(1).repeat(1, w, 1).unsqueeze(0).repeat(t, 1, 1, 1).unsqueeze(0).repeat(b, 1, 1, 1, 1) # b, c, t, h, w
        x_emb = x_emb.permute(0, 4, 1, 2, 3).contiguous()
        y_emb = self.row_embed(j).unsqueeze(0).repeat(h, 1, 1).unsqueeze(0).repeat(t, 1, 1, 1).unsqueeze(0).repeat(b, 1, 1, 1, 1) # b, c, t, h, w
        y_emb = y_emb.permute(0, 4, 1, 2, 3).contiguous()
        t_emb = self.temp_embed(k).unsqueeze(1).repeat(1, w, 1).unsqueeze(1).repeat(1, h, 1, 1).unsqueeze(0).repeat(b, 1, 1, 1, 1) # b, c, t, h, w
        pos = torch.cat([x_emb, y_emb, t_emb], dim=1)
        return pos
