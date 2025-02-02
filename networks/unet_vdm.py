# Source: https://github.com/addtt/variational-diffusion-models
#
# MIT License
#
# Copyright (c) 2022 Andrea Dittadi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Modifications:
# - Added data_adapters to UNetVDM to preprocess the inputs and postprocess the outputs
# - Replaced `timesteps` argument of `UNetModel.forward()` with time `t`, which is used to compute the `timesteps`
# - Added 1/1000 to t before computing timesteps embeddings so t isn't 0
# - Added concatenation of input and output of the network before the final projection

import numpy as np

import torch
from torch import einsum, nn, pi, softmax

from utils_model import sandwich


@torch.no_grad()
def zero_init(module: nn.Module) -> nn.Module:
    """Sets to zero all the parameters of a module, and returns the module."""
    
    for p in module.parameters():
        nn.init.zeros_(p.data)
        
    return module


class UNetVDM(nn.Module):
    def __init__(
        self,
        data_adapters,
        embedding_dim: int = 128,
        n_blocks: int = 32,
        n_attention_heads: int = 1,
        dropout_prob: float = 0.1,
        norm_groups: int = 32,
        input_channels: int = 3,
        use_fourier_features: bool = True,
        attention_everywhere: bool = False,
        image_size: int = 32,
    ):
        super().__init__()

        # 对输入进行前置处理, 比如加入位置编码.
        self.input_adapter = data_adapters["input_adapter"]
        # 将输出转换为目标 形式, 通常是将维度数 project 到指定数.
        self.output_adapter = data_adapters["output_adapter"]
        
        attention_params = dict(
            n_heads=n_attention_heads,
            n_channels=embedding_dim,
            norm_groups=norm_groups,
        )
        
        resnet_params = dict(
            ch_in=embedding_dim,
            ch_out=embedding_dim,
            condition_dim=4 * embedding_dim,
            dropout_prob=dropout_prob,
            norm_groups=norm_groups,
        )
        
        self.embed_conditioning = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim * 4),
            nn.SiLU(),
        )
        
        total_input_ch = input_channels
        if use_fourier_features:
            self.fourier_features = FourierFeatures()
            # C = (2F + 1)C, 其中 2F 代表傅里叶特征数(sin & cos 各占 F).
            # 经过傅里叶特征变换所输出的通道数为 2FC, 而这部分特征会和原特征拼接起来,
            # 于是通道数总共就为 (2F+1)C.
            total_input_ch *= 1 + self.fourier_features.num_features
            
        self.conv_in = nn.Conv2d(total_input_ch, embedding_dim, 3, padding=1)

        # Down path: n_blocks blocks with a resnet block and maybe attention.
        self.down_blocks = nn.ModuleList(
            # 注意, 实际上并没有下采样, 分辨率保持不变.
            UpDownBlock(
                resnet_block=ResnetBlock(**resnet_params),
                attention_block=AttentionBlock(**attention_params) if attention_everywhere else None,
            )
            for _ in range(n_blocks)
        )

        self.mid_resnet_block_1 = ResnetBlock(**resnet_params)
        self.mid_attn_block = AttentionBlock(**attention_params)
        self.mid_resnet_block_2 = ResnetBlock(**resnet_params)

        # Up path: n_blocks+1 blocks with a resnet block and maybe attention.
        resnet_params["ch_in"] *= 2  # double input channels due to skip connections
        self.up_blocks = nn.ModuleList(
            # 注意, 实际上并没有上采样, 分辨率保持不变.
            UpDownBlock(
                resnet_block=ResnetBlock(**resnet_params),
                attention_block=AttentionBlock(**attention_params) if attention_everywhere else None,
            )
            for _ in range(n_blocks + 1)
        )

        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=embedding_dim),
            nn.SiLU(),
            # 将最后的输出卷积层初始化为全0.
            zero_init(nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1)),
        )
        
        self.embedding_dim = embedding_dim
        self.input_channels = input_channels
        self.image_size = image_size
        self.use_fourier_features = use_fourier_features

    def forward(
        self,
        data: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # (B,H*W,C)
        flat_x = self.input_adapter(data, t)
        # (B,H,W,C)
        x = flat_x.reshape(flat_x.size(0), self.image_size, self.image_size, self.input_channels)

        # (B,) 因为同一个数据样本在各维度上所对应的时间变量一致, 所以只需要取同的样本的其中1个维度即可.
        t = t.float().flatten(start_dim=1)[:, 0]
        # (B,D) 这里 + 0.001 代表小于 0.001 即看作是起始时刻(因此起始时刻不为0), 与 paper 中的描述一致.
        t_embedding = get_timestep_embedding(t + 0.001, self.embedding_dim)
        # We will condition on time embedding.
        # (B,4D)
        cond = self.embed_conditioning(t_embedding)

        # (B,C,H,W)
        x_perm = x.permute(0, 3, 1, 2).contiguous()
        # 若设定了要使用傅里叶特征, 则将傅里叶特征拼接过来.
        # (B,(2F+1)C,H,W), 其中 2FC 是傅里叶特征变换模块输出的通道数.
        h = self.maybe_concat_fourier(x_perm)
        # (B,D,H,W)
        h = self.conv_in(h)
        
        hs = [h]
        for down_block in self.down_blocks:  # n_blocks times
            h = down_block(h, cond)
            hs.append(h)
        
        h = self.mid_resnet_block_1(h, cond)
        h = self.mid_attn_block(h)
        h = self.mid_resnet_block_2(h, cond)
        
        for up_block in self.up_blocks:  # n_blocks+1 times
            h = torch.cat([h, hs.pop()], dim=1)
            h = up_block(h, cond)
        
        # (B,H*W,D)
        # 这个最后的卷积层初始化为全0, 因此在参数更新前这个输出特征不起作用,
        # 于是以下才将网络的输入也一并拼接在一起再输入到最后的 linear projection.
        out = sandwich(self.conv_out(h).permute(0, 2, 3, 1).contiguous())
        # (B,H*W,C+D)
        out = torch.cat([sandwich(x), out], -1)
        # (B,H*W,out_channels,out_height)
        out = self.output_adapter(out)
        
        return out

    def maybe_concat_fourier(self, z):
        if self.use_fourier_features:
            return torch.cat([z, self.fourier_features(z)], dim=1)
        
        return z


class ResnetBlock(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out=None,
        condition_dim=None,
        dropout_prob=0.0,
        norm_groups=32,
    ):
        super().__init__()
        
        ch_out = ch_in if ch_out is None else ch_out
        
        self.ch_out = ch_out
        self.condition_dim = condition_dim
        
        self.net1 = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=ch_in),
            nn.SiLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
        )
        
        if condition_dim is not None:
            self.cond_proj = zero_init(nn.Linear(condition_dim, ch_out, bias=False))
        
        self.net2 = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=ch_out),
            nn.SiLU(),
            nn.Dropout(dropout_prob),
            zero_init(nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)),
        )
        
        if ch_in != ch_out:
            self.skip_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x, condition):
        h = self.net1(x)
        
        if condition is not None:
            assert condition.shape == (x.shape[0], self.condition_dim)
            
            # 这个条件映射层(全连接层)初始化为全0, 因此在参数更新前条件变量不起作用.
            condition = self.cond_proj(condition)
            # (B,D,1,1)
            condition = condition[:, :, None, None]
            h = h + condition
        
        h = self.net2(h)
        
        if x.shape[1] != self.ch_out:
            x = self.skip_conv(x)
        assert x.shape == h.shape
        
        return x + h


def get_timestep_embedding(
    timesteps,
    embedding_dim: int,
    dtype=torch.float32,
    max_timescale=10_000,
    min_timescale=1,
):
    """正弦位置编码, 相当于将时间变量的值看作是位置."""
    
    # Adapted from tensor2tensor and VDM codebase.
    assert timesteps.ndim == 1
    assert embedding_dim % 2 == 0
    
    num_timescales = embedding_dim // 2
    # num_timescales 个等比元素, 由 1/min_timescale 到 1/max_timescale(包含).
    # logspace 的底默认为 10, 其输入的前两个参数代表最小和最大的幂
    inv_timescales = torch.logspace(  # or exp(-linspace(log(min), log(max), n))
        -np.log10(min_timescale),
        -np.log10(max_timescale),
        num_timescales,
        device=timesteps.device,
    )
    
    timesteps *= 1000.0  # In DDPM the time step is in [0, 1000], here [0, 1]
    emb = timesteps.to(dtype)[:, None] * inv_timescales[None, :]  # (T, D/2)
    
    # sin(t * \frac{1}{10000^{i/d}}), cos(t * \frac{1}{10000^{i/d}})
    return torch.cat([emb.sin(), emb.cos()], dim=1)  # (T, D)


# TODO
class FourierFeatures(nn.Module):
    def __init__(self, first=5.0, last=6.0, step=1.0):
        super().__init__()
        self.freqs_exponent = torch.arange(first, last + 1e-8, step)

    @property
    def num_features(self):
        return len(self.freqs_exponent) * 2

    def forward(self, x):
        assert len(x.shape) >= 2

        # Compute (2pi * 2^n) for n in freqs.
        freqs_exponent = self.freqs_exponent.to(dtype=x.dtype, device=x.device)  # (F, )
        freqs = 2.0**freqs_exponent * 2 * pi  # (F, )
        freqs = freqs.view(-1, *([1] * (x.dim() - 1)))  # (F, 1, 1, ...)

        # Compute (2pi * 2^n * x) for n in freqs.
        features = freqs * x.unsqueeze(1)  # (B, F, X1, X2, ...)
        features = features.flatten(1, 2)  # (B, F * C, X1, X2, ...)

        # Output features are cos and sin of above. Shape (B, 2 * F * C, H, W).
        return torch.cat([features.sin(), features.cos()], dim=1)


def attention_inner_heads(qkv, num_heads):
    """Computes attention with heads inside of qkv in the channel dimension.

    Args:
        qkv: Tensor of shape (B, 3*H*C, T) with Qs, Ks, and Vs, where:
            H = number of heads,
            C = number of channels per head.
        num_heads: number of heads.

    Returns:
        Attention output of shape (B, H*C, T).
    """

    bs, width, length = qkv.shape
    ch = width // (3 * num_heads)

    # Split into (q, k, v) of shape (B, H*C, T).
    q, k, v = qkv.chunk(3, dim=1)

    # 对 Q, K 各自缩放 1/d^{1/4} 相当于 Q, K 矩阵相乘后的结果缩放了 1/(\sqrt{d})
    # Rescale q and k. This makes them contiguous in memory.
    scale = ch ** (-1 / 4)  # scale with 4th root = scaling output by sqrt
    q = q * scale
    k = k * scale

    # Reshape qkv to (B*H, C, T).
    new_shape = (bs * num_heads, ch, length)
    q = q.view(*new_shape)
    k = k.view(*new_shape)
    v = v.reshape(*new_shape)

    # Compute attention.
    weight = einsum("bct,bcs->bts", q, k)  # (B*H, T, T)
    weight = softmax(weight.float(), dim=-1).to(weight.dtype)  # (B*H, T, T)
    out = einsum("bts,bcs->bct", weight, v)  # (B*H, C, T)
    
    return out.reshape(bs, num_heads * ch, length)  # (B, H*C, T)


class Attention(nn.Module):
    """Based on https://github.com/openai/guided-diffusion."""

    def __init__(self, n_heads):
        super().__init__()
        
        self.n_heads = n_heads

    def forward(self, qkv):
        assert qkv.dim() >= 3, qkv.dim()
        assert qkv.shape[1] % (3 * self.n_heads) == 0
        
        spatial_dims = qkv.shape[2:]
        qkv = qkv.view(*qkv.shape[:2], -1)  # (B, 3*n_heads*C, T)
        out = attention_inner_heads(qkv, self.n_heads)  # (B, n_heads*C, T)
        
        return out.view(*out.shape[:2], *spatial_dims).contiguous()


class AttentionBlock(nn.Module):
    """Self-attention residual block."""

    def __init__(self, n_heads, n_channels, norm_groups):
        super().__init__()
        
        assert n_channels % n_heads == 0
        
        self.layers = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=n_channels),
            # 之所以将通道数扩展3倍是因为后续要输入到 Attention 模块, 为 Q, K ,V 各分配数量一致的通道数.
            nn.Conv2d(n_channels, 3 * n_channels, kernel_size=1),  # (B, 3 * C, H, W)
            Attention(n_heads),
            # 输出卷积层初始化为全0，因此在参数更新前这部分输出特征相当于不起作用.
            zero_init(nn.Conv2d(n_channels, n_channels, kernel_size=1)),
        )

    def forward(self, x):
        return self.layers(x) + x


class UpDownBlock(nn.Module):
    def __init__(self, resnet_block, attention_block=None):
        super().__init__()
        
        self.resnet_block = resnet_block
        self.attention_block = attention_block

    def forward(self, x, cond):
        x = self.resnet_block(x, cond)
        if self.attention_block is not None:
            x = self.attention_block(x)
            
        return x
