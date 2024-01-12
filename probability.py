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

import torch
import functools
from abc import abstractmethod

from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical as torch_Categorical
from torch.distributions.bernoulli import Bernoulli as torch_Bernoulli
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.uniform import Uniform

from math import log

from utils_model import (
    safe_exp,
    safe_log,
    idx_to_float,
    float_to_idx,
    quantize, sandwich,
)


class CtsDistribution:
    @abstractmethod
    def log_prob(self, x):
        pass

    @abstractmethod
    def sample(self):
        pass


class DiscreteDistribution:
    @property
    @abstractmethod
    def probs(self):
        pass

    @functools.cached_property
    def log_probs(self):
        return safe_log(self.probs)

    @functools.cached_property
    def mean(self):
        pass

    @functools.cached_property
    def mode(self):
        pass

    @abstractmethod
    def log_prob(self, x):
        pass

    @abstractmethod
    def sample(self):
        pass


class DiscretizedDistribution(DiscreteDistribution):
    def __init__(self, num_bins, device):
        self.num_bins = num_bins
        self.bin_width = 2.0 / num_bins
        self.half_bin_width = self.bin_width / 2.0
        self.device = device

    @functools.cached_property
    def class_centres(self):
        return torch.arange(self.half_bin_width - 1, 1, self.bin_width, device=self.device)

    @functools.cached_property
    def class_boundaries(self):
        return torch.arange(self.bin_width - 1, 1 - self.half_bin_width, self.bin_width, device=self.device)

    @functools.cached_property
    def mean(self):
        return (self.probs * self.class_centres).sum(-1)

    @functools.cached_property
    def mode(self):
        mode_idx = self.probs.argmax(-1).flatten()
        return self.class_centres[mode_idx].reshape(self.probs.shape[:-1])


class DiscretizedCtsDistribution(DiscretizedDistribution):
    """将一个连续型分布离散化."""
    
    def __init__(self, cts_dist, num_bins, device, batch_dims, clip=True, min_prob=1e-5):
        super().__init__(num_bins, device)
        
        self.cts_dist = cts_dist
        self.log_bin_width = log(self.bin_width)
        self.batch_dims = batch_dims
        self.clip = clip
        self.min_prob = min_prob

    @functools.cached_property
    def probs(self):
        bdry_cdfs = self.cts_dist.cdf(self.class_boundaries.reshape([-1] + ([1] * self.batch_dims)))
        bdry_slice = bdry_cdfs[:1]
        
        # 先对 CDF 做截断再计算每个区间的概率
        if self.clip:
            cdf_min = torch.zeros_like(bdry_slice)
            cdf_max = torch.ones_like(bdry_slice)
            bdry_cdfs = torch.cat([cdf_min, bdry_cdfs, cdf_max], 0)
            
            return (bdry_cdfs[1:] - bdry_cdfs[:-1]).moveaxis(0, -1)
        # 直接先计算各区间概率再按照有效取值范围进行缩放
        else:
            cdf_min = self.cts_dist.cdf(torch.zeros_like(bdry_slice) - 1)
            cdf_max = self.cts_dist.cdf(torch.ones_like(bdry_slice))
            bdry_cdfs = torch.cat([cdf_min, bdry_cdfs, cdf_max], 0)
            cdf_range = cdf_max - cdf_min
            cdf_mask = cdf_range < self.min_prob
            cdf_range = torch.where(cdf_mask, (cdf_range * 0) + 1, cdf_range)
            probs = (bdry_cdfs[1:] - bdry_cdfs[:-1]) / cdf_range
            probs = torch.where(cdf_mask, (probs * 0) + (1 / self.num_bins), probs)
            
            return probs.moveaxis(0, -1)

    def prob(self, x):
        # 区间索引 k \in [0, K-1]
        class_idx = float_to_idx(x, self.num_bins)
        # 区间中心 k_c
        centre = idx_to_float(class_idx, self.num_bins)
        # CDF(k_l), 其中 k_l 代表区间左端点.
        cdf_lo = self.cts_dist.cdf(centre - self.half_bin_width)
        # CDF(k_r), 其中 k_r 代表区间右端点.
        cdf_hi = self.cts_dist.cdf(centre + self.half_bin_width)
        
        # 对原来连续分布的 CDF 做截断, 使得:
        # CDF(k <= 0) = 0;
        # CDF(k >= K-1) = 1
        if self.clip:
            cdf_lo = torch.where(class_idx <= 0, torch.zeros_like(centre), cdf_lo)
            cdf_hi = torch.where(class_idx >= (self.num_bins - 1), torch.ones_like(centre), cdf_hi)
            
            return cdf_hi - cdf_lo
        else:
            # 有效的样本值范围为 [-1, 1], 于是计算对应的有效 CDF 范围.
            cdf_min = self.cts_dist.cdf(torch.zeros_like(centre) - 1)
            cdf_max = self.cts_dist.cdf(torch.ones_like(centre))
            cdf_range = cdf_max - cdf_min
            # 若有效 CDF 范围小于预设的最小概率，则设置 mask，并将此范围的概率以1代替，即不进行缩放, 否则会使得计算出来的采样概率非常接近于1.
            # 试想两个非常小的值相除, 因为它们都很小，非常接近，因此商接近于1.
            cdf_mask = cdf_range < self.min_prob
            cdf_range = torch.where(cdf_mask, (cdf_range * 0) + 1, cdf_range)
            prob = (cdf_hi - cdf_lo) / cdf_range
            
            # 将 mask 部分以均匀采样的概率值即 1/K 代替.
            return torch.where(cdf_mask, (prob * 0) + (1 / self.num_bins), prob)

    def log_prob(self, x):
        prob = self.prob(x)
        return torch.where(
            prob < self.min_prob,
            # 将 x 以对应区间的中点 k_c 表示并计算出其在原来连续分布中的对数概率密度: log(p(k_c)).
            # 这里加上 log(2/K) 相当于将 k_c 乘以 2/K 再取对数.
            self.cts_dist.log_prob(quantize(x, self.num_bins)) + self.log_bin_width,
            safe_log(prob),
        )

    def sample(self, sample_shape=torch.Size([])):
        if self.clip:
            # 采取截断的方式, 即直接从原来的连续分布中采样, 然后将其截断至对应的离散化区间.
            return quantize(self.cts_dist.sample(sample_shape), self.num_bins)
        else:
            # 要求连续分布的 CDF 的反函数存在, 即可以根据概率值逆向求出对应的样本.
            assert hasattr(self.cts_dist, "icdf")
            
            # 由于作者将数据归一化至 [-1, 1] 范围内，因此将 CDF 截断至该范围, 
            # 然后随机抽取该范围内的一个概率值并利用 CDF 的反函数逆向求出对应的样本.
            cdf_min = self.cts_dist.cdf(torch.zeros_like(self.cts_dist.mean) - 1)
            cdf_max = self.cts_dist.cdf(torch.ones_like(cdf_min))
            u = Uniform(cdf_min, cdf_max, validate_args=False).sample(sample_shape)
            cts_samp = self.cts_dist.icdf(u)
            
            return quantize(cts_samp, self.num_bins)


class GMM(MixtureSameFamily):
    def __init__(self, mix_wt_logits, means, std_devs):
        mix_wts = torch_Categorical(logits=mix_wt_logits, validate_args=False)
        components = Normal(means, std_devs, validate_args=False)
        super().__init__(mix_wts, components, validate_args=False)


class DiscretizedGMM(DiscretizedCtsDistribution):
    def __init__(self, params, num_bins, clip=False, min_std_dev=1e-3, max_std_dev=10, min_prob=1e-5, log_dev=True):
        assert params.size(-1) % 3 == 0
        if min_std_dev < 0:
            min_std_dev = 1.0 / (num_bins * 5)
        mix_wt_logits, means, std_devs = params.chunk(3, -1)
        if log_dev:
            std_devs = safe_exp(std_devs)
        std_devs = std_devs.clamp(min=min_std_dev, max=max_std_dev)
        super().__init__(
            cts_dist=GMM(mix_wt_logits, means, std_devs),
            num_bins=num_bins,
            device=params.device,
            batch_dims=params.ndim - 1,
            clip=clip,
            min_prob=min_prob,
        )


class DiscretizedNormal(DiscretizedCtsDistribution):
    def __init__(self, params, num_bins, clip=False, min_std_dev=1e-3, max_std_dev=10, min_prob=1e-5, log_dev=True):
        assert params.size(-1) == 2
        if min_std_dev < 0:
            min_std_dev = 1.0 / (num_bins * 5)
        mean, std_dev = params.split(1, -1)[:2]
        if log_dev:
            std_dev = safe_exp(std_dev)
        std_dev = std_dev.clamp(min=min_std_dev, max=max_std_dev)
        super().__init__(
            cts_dist=Normal(mean.squeeze(-1), std_dev.squeeze(-1), validate_args=False),
            num_bins=num_bins,
            device=params.device,
            batch_dims=params.ndim - 1,
            clip=clip,
            min_prob=min_prob,
        )


class Bernoulli(DiscreteDistribution):
    def __init__(self, logits):
        self.bernoulli = torch_Bernoulli(logits=logits, validate_args=False)

    @functools.cached_property
    def probs(self):
        p = self.bernoulli.probs.unsqueeze(-1)
        return torch.cat([1 - p, p], -1)

    @functools.cached_property
    def mode(self):
        return self.bernoulli.mode

    def log_prob(self, x):
        return self.bernoulli.log_prob(x.float())

    def sample(self, sample_shape=torch.Size([])):
        return self.bernoulli.sample(sample_shape)


class DiscretizedBernoulli(DiscretizedDistribution):
    def __init__(self, logits):
        super().__init__(2, logits.device)
        self.bernoulli = torch_Bernoulli(logits=logits, validate_args=False)

    @functools.cached_property
    def probs(self):
        p = self.bernoulli.probs.unsqueeze(-1)
        return torch.cat([1 - p, p], -1)

    @functools.cached_property
    def mode(self):
        return idx_to_float(self.bernoulli.mode, 2)

    def log_prob(self, x):
        return self.bernoulli.log_prob(float_to_idx(x, 2).float())

    def sample(self, sample_shape=torch.Size([])):
        return idx_to_float(self.bernoulli.sample(sample_shape), 2)


class DeltaDistribution(CtsDistribution):
    def __init__(self, mean, clip_range=1.0):
        if clip_range > 0:
            mean = mean.clip(min=-clip_range, max=clip_range)
        self.mean = mean

    def mode(self):
        return self.mean

    def mean(self):
        return self.mean

    def sample(self, sample_shape=torch.Size([])):
        return self.mean


class Categorical(DiscreteDistribution):
    def __init__(self, logits):
        self.categorical = torch_Categorical(logits=logits, validate_args=False)
        self.n_classes = logits.size(-1)

    @functools.cached_property
    def probs(self):
        return self.categorical.probs

    @functools.cached_property
    def mode(self):
        return self.categorical.mode

    def log_prob(self, x):
        return self.categorical.log_prob(x)

    def sample(self, sample_shape=torch.Size([])):
        return self.categorical.sample(sample_shape)


class DiscretizedCategorical(DiscretizedDistribution):
    def __init__(self, logits=None, probs=None):
        assert (logits is not None) or (probs is not None)
        if logits is not None:
            super().__init__(logits.size(-1), logits.device)
            self.categorical = torch_Categorical(logits=logits, validate_args=False)
        else:
            super().__init__(probs.size(-1), probs.device)
            self.categorical = torch_Categorical(probs=probs, validate_args=False)

    @functools.cached_property
    def probs(self):
        return self.categorical.probs

    @functools.cached_property
    def mode(self):
        return idx_to_float(self.categorical.mode, self.num_bins)

    def log_prob(self, x):
        return self.categorical.log_prob(float_to_idx(x, self.num_bins))

    def sample(self, sample_shape=torch.Size([])):
        return idx_to_float(self.categorical.sample(sample_shape), self.num_bins)


class CtsDistributionFactory:
    @abstractmethod
    def get_dist(self, params: torch.Tensor, input_params=None, t=None) -> CtsDistribution:
        """Note: input_params and t are not used but kept here to be consistency with DiscreteDistributionFactory."""
        pass


class GMMFactory(CtsDistributionFactory):
    def __init__(self, min_std_dev=1e-3, max_std_dev=10, log_dev=True):
        self.min_std_dev = min_std_dev
        self.max_std_dev = max_std_dev
        self.log_dev = log_dev

    def get_dist(self, params, input_params=None, t=None):
        mix_wt_logits, means, std_devs = params.chunk(3, -1)
        if self.log_dev:
            std_devs = safe_exp(std_devs)
        std_devs = std_devs.clamp(min=self.min_std_dev, max=self.max_std_dev)
        
        return GMM(mix_wt_logits, means, std_devs)


class NormalFactory(CtsDistributionFactory):
    def __init__(self, min_std_dev=1e-3, max_std_dev=10):
        self.min_std_dev = min_std_dev
        self.max_std_dev = max_std_dev

    def get_dist(self, params, input_params=None, t=None):
        mean, log_std_dev = params.split(1, -1)[:2]
        std_dev = safe_exp(log_std_dev).clamp(min=self.min_std_dev, max=self.max_std_dev)
        
        return Normal(mean.squeeze(-1), std_dev.squeeze(-1), validate_args=False)


class DeltaFactory(CtsDistributionFactory):
    def __init__(self, clip_range=1.0):
        self.clip_range = clip_range

    def get_dist(self, params, input_params=None, t=None):
        return DeltaDistribution(params.squeeze(-1), self.clip_range)


class DiscreteDistributionFactory:
    @abstractmethod
    def get_dist(self, params: torch.Tensor, input_params=None, t=None) -> DiscreteDistribution:
        """Note: input_params and t are only required by PredDistToDataDistFactory."""
        pass


class BernoulliFactory(DiscreteDistributionFactory):
    def get_dist(self, params, input_params=None, t=None):
        return Bernoulli(logits=params.squeeze(-1))


class CategoricalFactory(DiscreteDistributionFactory):
    def get_dist(self, params, input_params=None, t=None):
        return Categorical(logits=params)


class DiscretizedBernoulliFactory(DiscreteDistributionFactory):
    def get_dist(self, params, input_params=None, t=None):
        return DiscretizedBernoulli(logits=params.squeeze(-1))


class DiscretizedCategoricalFactory(DiscreteDistributionFactory):
    def get_dist(self, params, input_params=None, t=None):
        return DiscretizedCategorical(logits=params)


class DiscretizedGMMFactory(DiscreteDistributionFactory):
    def __init__(self, num_bins, clip=True, min_std_dev=1e-3, max_std_dev=10, min_prob=1e-5, log_dev=True):
        self.num_bins = num_bins
        self.clip = clip
        self.min_std_dev = min_std_dev
        self.max_std_dev = max_std_dev
        self.min_prob = min_prob
        self.log_dev = log_dev

    def get_dist(self, params, input_params=None, t=None):
        return DiscretizedGMM(
            params,
            num_bins=self.num_bins,
            clip=self.clip,
            min_std_dev=self.min_std_dev,
            max_std_dev=self.max_std_dev,
            min_prob=self.min_prob,
            log_dev=self.log_dev,
        )


class DiscretizedNormalFactory(DiscreteDistributionFactory):
    def __init__(self, num_bins, clip=True, min_std_dev=1e-3, max_std_dev=10, min_prob=1e-5, log_dev=True):
        self.num_bins = num_bins
        self.clip = clip
        self.min_std_dev = min_std_dev
        self.max_std_dev = max_std_dev
        self.min_prob = min_prob
        self.log_dev = log_dev

    def get_dist(self, params, input_params=None, t=None):
        return DiscretizedNormal(
            params,
            num_bins=self.num_bins,
            clip=self.clip,
            min_std_dev=self.min_std_dev,
            max_std_dev=self.max_std_dev,
            min_prob=self.min_prob,
            log_dev=self.log_dev,
        )


def noise_pred_params_to_data_pred_params(noise_pred_params: torch.Tensor, input_mean: torch.Tensor, t: torch.Tensor, min_variance: float, min_t=1e-6):
    """Convert output parameters that predict the noise added to data, to parameters that predict the data.
    将模型预测的噪声分布的参数转换为数据分布的参数."""
    
    data_shape = list(noise_pred_params.shape)[:-1]
    noise_pred_params = sandwich(noise_pred_params)
    input_mean = input_mean.flatten(start_dim=1)
    
    if torch.is_tensor(t):
        t = t.flatten(start_dim=1)
    else:
        t = (input_mean * 0) + t
        
    alpha_mask = (t < min_t).unsqueeze(-1)
    
    # \sigma_1^{2t}
    posterior_var = torch.pow(min_variance, t.clamp(min=min_t))
    # \gamma(t) = 1 - \sigma_1^{2t}
    gamma = 1 - posterior_var
    
    A = (input_mean / gamma).unsqueeze(-1)
    B = (posterior_var / gamma).sqrt().unsqueeze(-1)
    
    data_pred_params = []
    
    # 对应建模连续数据的场景: 模型预测的是噪声向量.
    if noise_pred_params.size(-1) == 1:
        noise_pred_mean = noise_pred_params
    # 对应建模离散化数据的场景: 模型预测的是噪声分布的均值与对数标准差. 
    elif noise_pred_params.size(-1) == 2:
        noise_pred_mean, noise_pred_log_dev = noise_pred_params.chunk(2, -1)
    else:
        assert noise_pred_params.size(-1) % 3 == 0
        mix_wt_logits, noise_pred_mean, noise_pred_log_dev = noise_pred_params.chunk(3, -1)
        data_pred_params.append(mix_wt_logits)
        
    data_pred_mean = A - (B * noise_pred_mean)
    data_pred_mean = torch.where(alpha_mask, 0 * data_pred_mean, data_pred_mean)
    data_pred_params.append(data_pred_mean)
    
    if noise_pred_params.size(-1) >= 2:
        noise_pred_dev = safe_exp(noise_pred_log_dev)
        data_pred_dev = B * noise_pred_dev
        data_pred_dev = torch.where(alpha_mask, 1 + (0 * data_pred_dev), data_pred_dev)
        data_pred_params.append(data_pred_dev)
        
    data_pred_params = torch.cat(data_pred_params, -1)
    data_pred_params = data_pred_params.reshape(data_shape + [-1])
    
    return data_pred_params


class PredDistToDataDistFactory(DiscreteDistributionFactory):
    def __init__(self, data_dist_factory, min_variance, min_t=1e-6):
        self.data_dist_factory = data_dist_factory
        self.data_dist_factory.log_dev = False
        self.min_variance = min_variance
        self.min_t = min_t

    def get_dist(self, params, input_params, t):
        data_pred_params = noise_pred_params_to_data_pred_params(params, input_params[0], t, self.min_variance, self.min_t)
        return self.data_dist_factory.get_dist(data_pred_params)
