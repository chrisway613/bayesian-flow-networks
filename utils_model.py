# Copyright 2023 NNAISENSE SA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import numpy as np
import torch
from torch import Tensor

CONST_log_range = 20
CONST_log_min = 1e-10
CONST_summary_rescale = 10
CONST_exp_range = 10
CONST_min_std_dev = math.exp(-CONST_exp_range)


def sandwich(x: Tensor):
    return x.reshape(x.size(0), -1, x.size(-1))


def safe_log(data: Tensor):
    return data.clamp(min=CONST_log_min).log()


def safe_exp(data: Tensor):
    return data.clamp(min=-CONST_exp_range, max=CONST_exp_range).exp()


def idx_to_float(idx: np.ndarray, num_bins: int):
    """将离散化区间索引 k 转换为对应的区间中心值 k_c.
    注意, 此处 k 的取值范围与论文中的不同, 论文中 k 的取值范围是 1~K, 而这里:
    k_c = \frac{2k+1}{K} - 1, where k \in [0, K-1]."""
    
    flt_zero_one = (idx + 0.5) / num_bins
    return (2.0 * flt_zero_one) - 1.0


def float_to_idx(flt: np.ndarray, num_bins: int):
    """根据离散化值 k_c 计算出对应的区间索引 k, 是 float_to_idx() 的逆向操作."""
    
    flt_zero_one = (flt / 2.0) + 0.5
    return torch.clamp(torch.floor(flt_zero_one * num_bins), min=0, max=num_bins - 1).long()


def quantize(flt, num_bins: int):
    """将浮点值量化以对应的离散化区间中点 k_c 表示, 因此看作是一个量化的过程."""
    return idx_to_float(float_to_idx(flt, num_bins), num_bins)


def pe_encode(sequence_length: int, embedding_size: int) -> Tensor:
    """Positional encoding as described in original attention is all you need paper"""

    pe = torch.zeros((sequence_length, embedding_size))
    pos = torch.arange(sequence_length).unsqueeze(1)
    pe[:, 0::2] = torch.sin(
        pos / torch.pow(1000, torch.arange(0, embedding_size, 2, dtype=torch.float32) / embedding_size)
    )
    pe[:, 1::2] = torch.cos(
        pos / torch.pow(1000, torch.arange(1, embedding_size, 2, dtype=torch.float32) / embedding_size)
    )

    return pe


def pe_encode_float(x: Tensor, max_freq: float, embedding_size: int) -> Tensor:
    pe = torch.zeros(list(x.shape) + [embedding_size], device=x.device)
    pos = (((x + 1) / 2) * max_freq).unsqueeze(-1)
    pe[..., 0::2] = torch.sin(
        pos
        / torch.pow(10000, torch.arange(0, embedding_size, 2, dtype=torch.float32, device=x.device) / embedding_size)
    )
    pe[..., 1::2] = torch.cos(
        pos
        / torch.pow(10000, torch.arange(1, embedding_size, 2, dtype=torch.float32, device=x.device) / embedding_size)
    )
    return pe
