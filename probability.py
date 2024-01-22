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
        # 离散区间数量: K
        self.num_bins = num_bins
        # 原数据取值范围是[-1,1], 如今划分为 K 个区间, 因此每个区间宽度是 2/K.
        self.bin_width = 2.0 / num_bins
        self.half_bin_width = self.bin_width / 2.0

        self.device = device

    @functools.cached_property
    def class_centres(self):
        # 类别中心的取值范围: [-1 + 1/K, 1 - 1/K]
        return torch.arange(self.half_bin_width - 1, 1, self.bin_width, device=self.device)

    @functools.cached_property
    def class_boundaries(self):
        # 各类别之间的边界: [-1 + 2/K, 1 - 2/K], 共 K-1 个.
        return torch.arange(self.bin_width - 1, 1 - self.half_bin_width, self.bin_width, device=self.device)

    @functools.cached_property
    def mean(self):
        # 将各类别中心用它们各自所对应的概率加权求和: \sum_{k=1}^K{p_k * k_c}
        return (self.probs * self.class_centres).sum(-1)

    @functools.cached_property
    def mode(self):
        """概率分布的 mode, 代表众数, 即概率最高处所对应的样本."""

        # 因为 class_centres 是1维的, 所以这里需要将索引展平.
        mode_idx = self.probs.argmax(-1).flatten()
        return self.class_centres[mode_idx].reshape(self.probs.shape[:-1])
        

class DiscretizedCtsDistribution(DiscretizedDistribution):
    """将一个连续型分布离散化."""
    
    def __init__(self, cts_dist, num_bins, device, batch_dims, clip=True, min_prob=1e-5):
        super().__init__(num_bins, device)

        # 原来的连续型分布, 要对其进行离散化处理.
        self.cts_dist = cts_dist
        # log(2/K)
        self.log_bin_width = log(self.bin_width)
        # B
        self.batch_dims = batch_dims
        
        # 是否要对原来连续型分布的 CDF 做截断.
        self.clip = clip
        # 用作概率的极小值
        self.min_prob = min_prob

    @functools.cached_property
    def probs(self):
        """计算数据位于各离散区间的概率."""
        
        # shape: [K-1] + [1] * B
        bdry_cdfs = self.cts_dist.cdf(self.class_boundaries.reshape([-1] + ([1] * self.batch_dims)))
        # shape: [1] + [1] * B
        bdry_slice = bdry_cdfs[:1]
        
        if self.clip:
            '''对原来连续型分布的 CDF 做截断: 小于第一个区间的左端概率置0、小于等于最后一个区间右端的概率置1.'''
            
            cdf_min = torch.zeros_like(bdry_slice)
            cdf_max = torch.ones_like(bdry_slice)
            # shape: [K+1] + [1] * B
            bdry_cdfs = torch.cat([cdf_min, bdry_cdfs, cdf_max], 0)

            # 利用 CDF(k_r) - CDF(k_l) 得到位于各区间的概率.
            # shape: [1] * B + [K]
            return (bdry_cdfs[1:] - bdry_cdfs[:-1]).moveaxis(0, -1)
        else:
            '''以条件概率的思想来计算数据位于各区间的概率，其中的条件就是数据位于 [-1,1] 取值范围内.
            先计算原连续型分布在 1 和 -1 处的 CDF 值，将两者作差从而得到位于 [-1,1] 内的概率，以此作为条件对各区间的概率进行缩放.'''

            # CDF(-1)
            cdf_min = self.cts_dist.cdf(torch.zeros_like(bdry_slice) - 1)
            # CDF(1)
            cdf_max = self.cts_dist.cdf(torch.ones_like(bdry_slice))
            # shape: [K+1] + [1] * B
            bdry_cdfs = torch.cat([cdf_min, bdry_cdfs, cdf_max], 0)

            # p_{-1 < x <= 1}
            cdf_range = cdf_max - cdf_min
            # 若 cdf_range 太小，则设置 mask，并将其以1代替，即不对区间的概率进行缩放, 否则会使得计算出来的采样概率非常接近于1.
            # 两个非常小的值相除, 由于它们都很小、非常接近，因此商接近于1.
            cdf_mask = cdf_range < self.min_prob
            cdf_range = torch.where(cdf_mask, (cdf_range * 0) + 1, cdf_range)

            # shape: [K] + [1] * B
            probs = (bdry_cdfs[1:] - bdry_cdfs[:-1]) / cdf_range
            # 若整个 cdf_range 太小, 说明各区间的概率差异微不足道, 因此干脆将每个区间的概率都用 1/K 即均等的概率代替.
            probs = torch.where(cdf_mask, (probs * 0) + (1 / self.num_bins), probs)

            # shape: [1] * B + [K]
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
        
        if self.clip:
            '''对原来连续型分布的 CDF 做截断, 使得:
            CDF(k <= 0) = 0;
            CDF(k >= K-1) = 1'''
            
            cdf_lo = torch.where(class_idx <= 0, torch.zeros_like(centre), cdf_lo)
            cdf_hi = torch.where(class_idx >= (self.num_bins - 1), torch.ones_like(centre), cdf_hi)
            
            return cdf_hi - cdf_lo
        else:
            '''以条件概率的思想来计算数据位于某个离散区间内的概率，其中的条件就是数据位于 [-1,1] 取值范围内.
            先计算原连续型分布在 1 和 -1 处的 CDF 值，将两者作差从而得到位于 [-1,1] 内的概率，以此作为条件对区间的概率进行缩放.'''
            
            cdf_min = self.cts_dist.cdf(torch.zeros_like(centre) - 1)
            cdf_max = self.cts_dist.cdf(torch.ones_like(centre))
            cdf_range = cdf_max - cdf_min
            
            # 若 cdf_range 太小，则设置 mask，并将其以1代替，即不对区间的概率进行缩放, 否则会使得计算出来的采样概率非常接近于1.
            # 两个非常小的值相除, 由于它们都很小、非常接近，因此商接近于1.
            cdf_mask = cdf_range < self.min_prob
            cdf_range = torch.where(cdf_mask, (cdf_range * 0) + 1, cdf_range)
            prob = (cdf_hi - cdf_lo) / cdf_range
            
            # 若整个 cdf_range 太小, 说明各区间的概率差异微不足道, 因此干脆将区间的概率都用 1/K 即均等的概率代替.
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
            # 直接从原来的连续型分布中采样, 然后将其量化至对应的离散化区间.
            # 此处, clip 的意思是:
            # 若小于第一个区间，则以第一个区间中点表示；
            # 同理，若大于最后一个区间，则以最后一个区间的中点表示.
            return quantize(self.cts_dist.sample(sample_shape), self.num_bins)
        else:
            # 要求原来连续型分布的 CDF 存在反函数, 即可以根据概率值逆向求出对应的样本.
            assert hasattr(self.cts_dist, "icdf")
            
            # 数据的取值范围是 [-1,1], 先根据原来的连续型分布计算出 CDF(-1) 和 CDF(1),
            # 然后利用 CDF 的反函数仅在这个 range 内考虑采样.
            cdf_min = self.cts_dist.cdf(torch.zeros_like(self.cts_dist.mean) - 1)
            cdf_max = self.cts_dist.cdf(torch.ones_like(cdf_min))

            # 由于 CDF 是服从均匀分布的, 因此从均匀分布中采样出 CDF 值并利用反函数求出对应样本就等价于从目标分布中采样.
            u = Uniform(cdf_min, cdf_max, validate_args=False).sample(sample_shape)
            cts_samp = self.cts_dist.icdf(u)

            # 最后将样本量化至对应的离散化区间.
            # 注意, 与前面 clip 的方式不同, 此处在量化前样本已经处于有效的离散化区间内了, 因为采样区间是在[-1,1]内考虑的.
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
            # 若传入的是对数标准差, 那么此处就需要取自然指数进行还原.
            std_dev = safe_exp(std_dev)
        std_dev = std_dev.clamp(min=min_std_dev, max=max_std_dev)
        
        super().__init__(
            cts_dist=Normal(mean.squeeze(-1), std_dev.squeeze(-1), validate_args=False),
            num_bins=num_bins,
            device=params.device,
            # 注意所谓的 batch dims 并非指数据的 batch size,
            # 而是除离散化区间数量以外与分布本身关系不大的其它维度.
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

    # (B,L,D)
    data_shape = list(noise_pred_params.shape)[:-1]
    # (B,L*D,NP), NP: num parameters per data
    noise_pred_params = sandwich(noise_pred_params)
    # (B,L*D)
    input_mean = input_mean.flatten(start_dim=1)
    
    if torch.is_tensor(t):
        t = t.flatten(start_dim=1)
    else:
        t = (input_mean * 0) + t
        
    # (B,L*D,1)
    alpha_mask = (t < min_t).unsqueeze(-1)
    
    # \sigma_1^{2t}
    posterior_var = torch.pow(min_variance, t.clamp(min=min_t))
    # \gamma(t) = 1 - \sigma_1^{2t}
    gamma = 1 - posterior_var

    # \frac{\mu}{\gamma(t)}
    A = (input_mean / gamma).unsqueeze(-1)
    # \sqrt{\frac{1-\gamma(t)}{\gamma(t)}}
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

    # 连续数据: x = \frac{\mu}{\gamma(t)} - \sqrt{\frac{1-\gamma(t)}{\gamma(t)}} \epsilon
    # 离散化数据: \mu_{x} = \frac{\mu}{\gamma(t)} - \sqrt{\frac{1-\gamma(t)}{\gamma(t)}} \mu_{\epsilon}
    data_pred_mean = A - (B * noise_pred_mean)
    # 时间变量的值过小则被认为是起始时刻, 等同于先验形式, 即标准高斯分布, 于是将预测的均值置0
    data_pred_mean = torch.where(alpha_mask, 0 * data_pred_mean, data_pred_mean)
    data_pred_params.append(data_pred_mean)
    
    if noise_pred_params.size(-1) >= 2:
        # 将对数标准差取自然指数复原: exp(ln(\sigma_{\epsilon})) -> \sigma_{\epsilon}
        noise_pred_dev = safe_exp(noise_pred_log_dev)
        # 将噪声分布的标准差转换为目标数据分布的标准差: \sqrt{\frac{1-\gamma(t)}{\gamma(t)}} exp(ln(\sigma_{\epsilon})) -> \mu_x
        data_pred_dev = B * noise_pred_dev
        # 时间变量的值过小则被认为是起始时刻, 等同于先验形式, 即标准高斯分布, 于是将预测的标准差置1
        data_pred_dev = torch.where(alpha_mask, 1 + (0 * data_pred_dev), data_pred_dev)
        data_pred_params.append(data_pred_dev)

    # (B,L*D,NP)
    data_pred_params = torch.cat(data_pred_params, -1)
    # (B,L,D,NP)
    data_pred_params = data_pred_params.reshape(data_shape + [-1])
    
    return data_pred_params


class PredDistToDataDistFactory(DiscreteDistributionFactory):
    def __init__(self, data_dist_factory, min_variance, min_t=1e-6):
        self.data_dist_factory = data_dist_factory
        # 之所以设为 False 是因为在以下 noise_pred_params_to_data_pred_params() 方法中会将对数标准差使用自然指数进行转换,
        # 而无需原数据分布的工厂自行转换.
        self.data_dist_factory.log_dev = False
        self.min_variance = min_variance
        self.min_t = min_t

    def get_dist(self, params, input_params, t):
        data_pred_params = noise_pred_params_to_data_pred_params(params, input_params[0], t, self.min_variance, self.min_t)
        return self.data_dist_factory.get_dist(data_pred_params)
