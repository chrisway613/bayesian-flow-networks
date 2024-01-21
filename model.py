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

"""
This file implements the Bayesian Flow and BFN loss for continuous and discrete variables.
Finally it implements the BFN using these objects.

For consistency we use always use a tuple to store input parameters.
It has just one element for discrete data (the probabilities) and two for continuous/discretized (mean & variance).
The probability distributions and network architectures are defined in 'probability.py' and 'networks dir/'.
"Cts" is an abbreviation of "Continuous".
"""

import math
from abc import abstractmethod, ABC
from typing import Union, Optional

import torch
import torch.distributions as D
import torch.nn.functional as F
from torch import nn, Tensor

from probability import (
    DiscreteDistributionFactory,
    CtsDistributionFactory,
    PredDistToDataDistFactory,
    DiscretizedCtsDistribution,
)
from utils_model import sandwich, float_to_idx


class BayesianFlow(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_prior_input_params(self, data_shape: tuple, device: torch.device) -> tuple[Tensor, ...]:
        """Returns the initial input params (for a batch) at t=0. Used during sampling.
        For discrete data, the tuple has length 1 and contains the initial class probabilities.
        For continuous data, the tuple has length 2 and contains the mean and precision.
        
        返回起始时刻 t=0 的先验参数, 这些参数会作为模型的输入, 这个方法用于采样过程."""
        pass

    @abstractmethod
    def params_to_net_inputs(self, params: tuple[Tensor, ...]) -> Tensor:
        """Utility method to convert input distribution params to network inputs if needed.
        
        如果有必要的话, 将输入分布的参数转换为适合模型输入的形式."""
        pass

    @abstractmethod
    def get_alpha(self, i: Union[int, Tensor], n_steps: int) -> float:
        """Returns the alpha at step i of total n_steps according to the flow schedule. Used:
        a) during sampling, when i and alpha are the same for all samples in the batch.
        b) during discrete time loss computation, when i and alpha are different for samples in the batch.
        
        计算离散时间步的 \alpha_i, 用于采样过程或离散时间的损失函数. """
        pass

    @abstractmethod
    def get_sender_dist(self, x: Tensor, alpha: Union[float, Tensor], shape=torch.Size([])) -> D.Distribution:
        """Returns the sender distribution with accuracy alpha obtained by adding appropriate noise to the data x. Used:
        a) during sampling (same alpha for whole batch) to sample from the output distribution produced by the net.
        b) during discrete time loss computation when alpha are different for samples in the batch.
        
        计算输入分布. """
        pass

    @abstractmethod
    def update_input_params(self, input_params: tuple[Tensor, ...], y: Tensor, alpha: float) -> tuple[Tensor, ...]:
        """Updates the distribution parameters using Bayes' theorem in light of noisy sample y.
        Used during sampling when alpha is the same for the whole batch.
        
        利用观测样本 y 进行计算出后验, 从而更新先验. """
        pass

    @abstractmethod
    def forward(self, data: Tensor, t: Tensor) -> tuple[Tensor, ...]:
        """Returns a sample from the Bayesian Flow distribution over input parameters at time t conditioned on data.
        Used during training when t (and thus accuracies) are different for different samples in the batch.
        For discrete data, the returned tuple has length 1 and contains the class probabilities.
        For continuous data, the returned tuple has length 2 and contains the mean and precision.
        
        从贝叶斯流分布中采样, 返回采样结果(也就是经过后验更新后的输入分布参数). """
        pass


class Loss(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def cts_time_loss(self, data: Tensor, output_params: Tensor, input_params: Tensor, t: Tensor) -> Tensor:
        """Returns the continuous time KL loss (and any other losses) at time t (between 0 and 1).
        The input params are only used when the network is parameterized to predict the noise for continuous data.
        
        连续时间的损失函数. """
        pass

    @abstractmethod
    def discrete_time_loss(
        self, data: Tensor,
        output_params: Tensor, input_params: Tensor,
        t: Tensor, n_steps: int, n_samples: int = 20
    ) -> Tensor:
        """Returns the discrete time KL loss for n_steps total of communication at time t (between 0 and 1) using
        n_samples for Monte Carlo estimation of the discrete loss.
        The input params are only used when the network is parameterized to predict the noise for continuous data.
        
        离散时间的损失函数, 当所需计算的 KL 散度没有解析形式时, 使用蒙特卡洛方法来近似估计. """
        pass

    @abstractmethod
    def reconstruction_loss(self, data: Tensor, output_params: Tensor, input_params: Tensor) -> Tensor:
        """Returns the reconstruction loss, i.e. the final cost of transmitting clean data.
        The input params are only used when the network is parameterized to predict the noise for continuous data.
        
        重构损失, 不参与训练. """
        pass


## Continuous or Discretized data ##


class CtsBayesianFlow(BayesianFlow):
    """建模连续/离散化数据的贝叶斯流."""
    
    def __init__(
        self,
        min_variance: float = 1e-6,
    ):
        super().__init__()
        self.min_variance = min_variance

    @torch.no_grad()
    def forward(self, data: Tensor, t: Tensor) -> tuple[Tensor, None]:
        """返回贝叶斯流分布的采样结果, 即经过后验更新的输入分布的均值向量: \mu."""
        
        # \omega_1^{2t}
        post_var = torch.pow(self.min_variance, t)
        # \gamma(t)
        alpha_t = 1 - post_var
        # \gamma(t)(1-\gamma(t))
        mean_var = alpha_t * post_var
        
        # 贝叶斯流分布的均值: \gamma(t)x
        mean_mean = alpha_t * data
        # 贝叶斯流分布的标准差: \sqrt{\gamma(t)(1-\gamma(t))}
        mean_std_dev = mean_var.sqrt()
        
        # 标准高斯噪声
        noise = torch.randn(mean_mean.shape, device=mean_mean.device)
        # 利用重参数化技术构造贝叶斯流分布的样本
        mean = mean_mean + (mean_std_dev * noise)
        
        # We don't need to compute the variance because it is not needed by the network, so set it to None
        input_params = (mean, None)
        
        return input_params

    def params_to_net_inputs(self, params: tuple[Tensor]) -> Tensor:
        # 仅取输入分布的均值向量作为 BFN 的输入
        # Only the mean is used by the network
        return params[0]

    def get_prior_input_params(self, data_shape: tuple, device: torch.device) -> tuple[Tensor, float]:
        # 起始时刻的先验是标准高斯分布, 均值为0, 方差为1(协方差矩阵是对角元均为1的对角阵)
        return torch.zeros(*data_shape, device=device), 1.0

    def get_alpha(self, i: Union[int, Tensor], n_steps: int) -> Union[float, Tensor]:
        # 根据 \beta(t_i) - \beta(t_{i-1}) 计算, 其中 t_i = \frac{i}{n}.
        sigma_1 = math.sqrt(self.min_variance)
        return (sigma_1 ** (-2 * i / n_steps)) * (1 - sigma_1 ** (2 / n_steps))

    def get_sender_dist(self, x: Tensor, alpha: Union[float, Tensor], shape=torch.Size([])) -> D.Distribution:
        # 返回输入分布, 精度 \alpha 是方差的倒数.
        dist = D.Normal(x, 1.0 / alpha**0.5)
        return dist

    def update_input_params(self, input_params: tuple[Tensor, float], y: Tensor, alpha: float) -> tuple[Tensor, float]:
        """贝叶斯更新函数, 对输入分布的参数进行后验更新."""
        
        input_mean, input_precision = input_params
        # \rho_i = \rho_{i-1} + \alpha
        new_precision = input_precision + alpha
        # 根据贝叶斯定理计算: \mu_i = \frac{ \rho_{i-1} \mu_{i-1} + \alpha y }{\rho_i}
        new_mean = ((input_precision * input_mean) + (alpha * y)) / new_precision
        
        return new_mean, new_precision


class CtsBayesianFlowLoss(Loss):
    """建模连续/离散化数据场景时所用的损失函数, 包括：
    -离散时间损失函数;
    -连续时间损失函数;
    -重构损失"""
    
    def __init__(
        self,
        bayesian_flow: CtsBayesianFlow,
        distribution_factory: Union[CtsDistributionFactory, DiscreteDistributionFactory],
        min_loss_variance: float = -1,
        noise_pred: bool = True,
    ):
        super().__init__()
        
        self.bayesian_flow = bayesian_flow
        self.distribution_factory = distribution_factory
        # \sigma_1^{2t} 的下限, 以防用作分母时溢出.
        self.min_loss_variance = min_loss_variance
        # -ln(\sigma_1)
        self.C = -0.5 * math.log(bayesian_flow.min_variance)
        
        # 是否预测噪声(亦或是直接预测数据)
        self.noise_pred = noise_pred
        if self.noise_pred:
            self.distribution_factory.log_dev = False
            # 在预测噪声的情况下, 将预测的噪声分布的参数转换为对应数据分布的参数，从而得到对应的数据分布.
            self.distribution_factory = PredDistToDataDistFactory(
                self.distribution_factory, self.bayesian_flow.min_variance
            )

    def cts_time_loss(self, data: Tensor, output_params: Tensor, input_params: Tensor, t) -> Tensor:
        # reshape
        output_params = sandwich(output_params)
        
        t = t.flatten(start_dim=1).float()
        flat_target = data.flatten(start_dim=1)
        
        # \sigma_1^{2t}
        posterior_var = torch.pow(self.bayesian_flow.min_variance, t)
        if self.min_loss_variance > 0:
            # 做最小值截断, 以防其作分母时防止溢出
            posterior_var = posterior_var.clamp(min=self.min_loss_variance)
        
        # 接收者分布
        pred_dist = self.distribution_factory.get_dist(output_params, input_params, t)
        # 接收者分布的均值 E[P(\theta, t)], 作为对真实数据的估计
        pred_mean = pred_dist.mean
        
        mse_loss = (pred_mean - flat_target).square()
        # 连续时间的损失函数计算公式: -ln(\sigma_1) \sigma_1{-2t} || x - \hat{x}(\theta, t) ||^2
        loss = self.C * mse_loss / posterior_var
        
        return loss

    def discrete_time_loss(
        self, data: Tensor,
        output_params: Tensor, input_params: Tensor,
        t: Tensor, n_steps: int, n_samples=10
    ) -> Tensor:
        # reshape
        output_params = sandwich(output_params)
        t = t.flatten(start_dim=1).float()
        
        output_dist = self.distribution_factory.get_dist(output_params, input_params, t)

        # 离散化数据的场景
        if hasattr(output_dist, "probs"):  # output distribution is discretized normal
            t = t.flatten(start_dim=1)
            i = t * n_steps + 1  # since t = (i - 1) / n
            
            alpha = self.bayesian_flow.get_alpha(i, n_steps)
            
            flat_target = data.flatten(start_dim=1)
            # 发送者分布
            sender_dist = self.bayesian_flow.get_sender_dist(flat_target, alpha)
            # 因为使用蒙特卡洛方法来估计发送者分布与接收者分布之间的 KL 散度，所以要从发送者分布中采样观测样本 y
            y = sender_dist.sample(torch.Size([n_samples]))
            
            # 模型输出的分配到各离散化区间的概率值. 
            #(B,D,K)
            receiver_mix_wts = sandwich(output_dist.probs)
            # 输出分布是类别分布, 在每个离散化区间都分配一定概率.
            receiver_mix_dist = D.Categorical(probs=receiver_mix_wts, validate_args=False)
            # 以各离散化区间的中心为均值构造多个一维高斯分布，其中每个都与发送者分布的形式一致(噪声强度相等, 即方差一致).\
            receiver_components = D.Normal(
                output_dist.class_centres, (1.0 / alpha.sqrt()).unsqueeze(-1), validate_args=False
            )
            # 接收者分布, 在数据的每个维度上都是混合高斯分布.
            receiver_dist = D.MixtureSameFamily(receiver_mix_dist, receiver_components, validate_args=False)
            
            # (B,1)
            loss = (
                (sender_dist.log_prob(y) - receiver_dist.log_prob(y))  # 发送者分布和接收者分布的概率密度对数差
                .mean(0)  # 在蒙特卡洛采样的样本数上做平均
                .flatten(start_dim=1)
                .mean(1, keepdims=True)
            )
        # 连续数据的场景
        else:  # output distribution is normal
            pred_mean = output_dist.mean
            flat_target = data.flatten(start_dim=1)
            mse_loss = (pred_mean - flat_target).square()
            i = t * n_steps + 1
            alpha = self.bayesian_flow.get_alpha(i, n_steps)
            loss = alpha * mse_loss / 2
            
        return n_steps * loss

    def reconstruction_loss(self, data: Tensor, output_params: Tensor, input_params: Tensor) -> Tensor:
        output_params = sandwich(output_params)
        flat_data = data.flatten(start_dim=1)
        
        # 重构损失只发生在最后时刻，于是 t=1.
        t = torch.ones_like(data).flatten(start_dim=1).float()
        output_dist = self.distribution_factory.get_dist(output_params, input_params, t)
        
        if hasattr(output_dist, "probs"):  # output distribution is discretized normal
            reconstruction_loss = -output_dist.log_prob(flat_data)
        else:  # output distribution is normal, but we use discretized normal to make results comparable (see Sec. 7.2)
            if self.bayesian_flow.min_variance == 1e-3:  # used for 16 bin CIFAR10
                noise_dev = 0.7 * math.sqrt(self.bayesian_flow.min_variance)
                num_bins = 16
            else:
                noise_dev = math.sqrt(self.bayesian_flow.min_variance)
                num_bins = 256
                
            mean = output_dist.mean.flatten(start_dim=1)
            final_dist = D.Normal(mean, noise_dev)
            # 离散化的正态分布
            final_dist = DiscretizedCtsDistribution(final_dist, num_bins, device=t.device, batch_dims=mean.ndim - 1)
            reconstruction_loss = -final_dist.log_prob(flat_data)
            
        return reconstruction_loss


## Discrete Data ##


class DiscreteBayesianFlow(BayesianFlow):
    def __init__(
        self,
        n_classes: int,
        min_sqrt_beta: float = 1e-10,
        discretize: bool = False,
        epsilon: float = 1e-6,
        max_sqrt_beta: float = 1,
    ):
        super().__init__()
        
        self.n_classes = n_classes
        self.epsilon = epsilon
        
        # 是否进行离散化操作
        self.discretize = discretize
        
        # \sqrt{\beta} 的下限
        self.min_sqrt_beta = min_sqrt_beta
        # \sqrt{\beta(1)}
        self.max_sqrt_beta = max_sqrt_beta
        
        # 均匀分布的期望熵
        self.uniform_entropy = math.log(self.n_classes)

    def t_to_sqrt_beta(self, t):
        """计算当前时刻的 accuracy schedule: \beta(t) 的开根:
           sqrt{\beta(t)} = t \sqrt{\beta(1)}."""
        
        return t * self.max_sqrt_beta

    def count_dist(self, x, beta=None) -> D.Distribution:
        """贝叶斯流分布中的期望部分所对应的发送者分布."""

        # Ke_x - 1
        mean = (self.n_classes * F.one_hot(x.long(), self.n_classes)) - 1
        # \sqrt{K}
        std_dev = math.sqrt(self.n_classes)
        
        if beta is not None:
            # \beta(t)(Ke_x - 1)
            mean = mean * beta
            # \sqrt{\beta(t)K}
            std_dev = std_dev * beta.sqrt()
            
        return D.Normal(mean, std_dev, validate_args=False)

    def count_sample(self, x, beta):
        """利用重参数化采样技术(rsample())采样出观测样本作为贝叶斯流分布的 logits 源(下一步将其输入 softmax 以实现后验更新)."""
        return self.count_dist(x, beta).rsample()

    @torch.no_grad()
    def get_prior_input_params(self, data_shape: tuple, device: torch.device) -> tuple[Tensor]:
        """初始先验: 各类别概率相等的均匀分布 U{1, K}."""
        
        # 注意返回的是元组, 这是为了与连续/离散化数据的场景保持一致性.
        return (torch.ones(*data_shape, self.n_classes, device=device) / self.n_classes,)

    @torch.no_grad()
    def params_to_net_inputs(self, params: tuple[Tensor]) -> Tensor:
        params = params[0]
        if self.n_classes == 2:
            # 作者使用的 MNIST 数据集是经过二值化处理的, 因此这部分针对 MNIST 操作,
            # 将模型输入的范围缩放至 [-1,1]
            params = params * 2 - 1  # We scale-shift here for MNIST instead of in the network like for text
            # 因为总共只有两个类别, 所以取其中一类所对应的概率即可.
            params = params[..., :1]
            
        return params

    def get_alpha(self, i: Union[int, Tensor], n_steps: int) -> Union[float, Tensor]:
        # 计算离散时间步所对应的精度: \alpha_i = \beta(1) \frac{2i-1}{n^2}
        return ((self.max_sqrt_beta / n_steps) ** 2) * (2 * i - 1)

    def get_sender_dist(self, x: Tensor, alpha: Union[float, Tensor], shape=torch.Size([])) -> D.Distribution:
        e_x = F.one_hot(x.long(), self.n_classes)
        alpha = alpha.unsqueeze(-1) if isinstance(alpha, Tensor) else alpha
        dist = D.Normal(alpha * ((self.n_classes * e_x) - 1), (self.n_classes * alpha) ** 0.5)
        
        return dist

    def update_input_params(self, input_params: tuple[Tensor], y: Tensor, alpha: float) -> tuple[Tensor]:
        """贝叶斯更新函数: 利用贝叶斯定理计算后验."""
        
        new_input_params = input_params[0] * y.exp()
        new_input_params /= new_input_params.sum(-1, keepdims=True)
        
        # 注意返回的是元组
        return (new_input_params,)

    @torch.no_grad()
    def forward(self, data: Tensor, t: Tensor) -> tuple[Tensor]:
        """根据贝叶斯流分布完成后验更新."""
        
        if self.discretize:
            # 若要进行离散化操作, 则将数据以对应的离散化区间索引表示.
            data = float_to_idx(data, self.n_classes)
        
        # \sqrt{\beta(t)}
        sqrt_beta = self.t_to_sqrt_beta(t.clamp(max=1 - self.epsilon))
        lo_beta = sqrt_beta < self.min_sqrt_beta
        sqrt_beta = sqrt_beta.clamp(min=self.min_sqrt_beta)
        # \beta(t)
        beta = sqrt_beta.square().unsqueeze(-1)
        
        # 从精度参数为 \beta(t) 的发送者分布中采样观测样本以作为贝叶斯流分布的 logits.
        logits = self.count_sample(data, beta)
        probs = F.softmax(logits, -1)
        # 将精度太小的部分所对应的后验以均匀先验 \frac{1}{K} 代替.
        # 这是因为精度太小, 那么对应的观测样本也"不靠谱"——所包含真实数据的信息太少,
        # 将其作为 logits 就不靠谱, 即以此为根据而实现的后验更新意义不大.
        probs = torch.where(lo_beta.unsqueeze(-1), torch.ones_like(probs) / self.n_classes, probs)
        if self.n_classes == 2:
            # 如果是二分类则只取其中一类的概率即可.
            probs = probs[..., :1]
            probs = probs.reshape_as(data)
            
        input_params = (probs,)
        
        return input_params


class DiscreteBayesianFlowLoss(Loss):
    def __init__(
        self,
        bayesian_flow: DiscreteBayesianFlow,
        distribution_factory: DiscreteDistributionFactory,
    ):
        super().__init__()
        
        self.bayesian_flow = bayesian_flow
        self.distribution_factory = distribution_factory
        self.K = self.bayesian_flow.n_classes

    def cts_time_loss(self, data: Tensor, output_params: Tensor, input_params: Tensor, t) -> Tensor:
        flat_output = sandwich(output_params)
        pred_probs = self.distribution_factory.get_dist(flat_output).probs
        
        flat_target = data.flatten(start_dim=1)
        if self.bayesian_flow.discretize:
            flat_target = float_to_idx(flat_target, self.K)

        tgt_mean = torch.nn.functional.one_hot(flat_target.long(), self.K)
        kl = self.K * ((tgt_mean - pred_probs).square()).sum(-1)
        t = t.flatten(start_dim=1).float()
        loss = t * (self.bayesian_flow.max_sqrt_beta**2) * kl
        
        return loss

    def discrete_time_loss(
        self, data: Tensor, output_params: Tensor, input_params: Tensor, t: Tensor, n_steps: int, n_samples=10
    ) -> Tensor:
        flat_target = data.flatten(start_dim=1)
        if self.bayesian_flow.discretize:
            flat_target = float_to_idx(flat_target, self.K)
        
        # 根据 t = \frac{i-1}{n} 反过来计算 i 
        i = t * n_steps + 1
        # \alpha_i
        alpha = self.bayesian_flow.get_alpha(i, n_steps).flatten(start_dim=1)

        flat_output = sandwich(output_params)
        receiver_mix_wts = self.distribution_factory.get_dist(flat_output).probs
        # 这里之所以要在倒数第2个维度上加一维是因为以下 components 在每个类别上的均值向量都是 K 维 one-hot,
        # 从而在每个类别上生成的是 K 个相互独立的正态分布. 总共有 K 类, 于是就有 K x K 个分布.
        # 因此这里增加维度是为了让 categorical 权重 与 components 对齐.
        receiver_mix_dist = D.Categorical(probs=receiver_mix_wts.unsqueeze(-2))
        
        classes = torch.arange(self.K, device=flat_target.device).long().unsqueeze(0).unsqueeze(0)
        receiver_components = self.bayesian_flow.get_sender_dist(classes, alpha.unsqueeze(-1))
        
        receiver_dist = D.MixtureSameFamily(receiver_mix_dist, receiver_components)
        
        sender_dist = self.bayesian_flow.get_sender_dist(flat_target, alpha)
        # 从发送者分布中采样, 以蒙特卡洛方法近似估计其与接收者分布之间的 KL loss
        y = sender_dist.sample(torch.Size([n_samples]))
        
        # (B,1)
        loss = n_steps * (sender_dist.log_prob(y) - receiver_dist.log_prob(y)).mean(0).sum(-1).mean(1, keepdims=True)
        
        return loss

    def reconstruction_loss(self, data: Tensor, output_params: Tensor, input_params: Tensor) -> Tensor:
        flat_outputs = sandwich(output_params)
        flat_data = data.flatten(start_dim=1)
        output_dist = self.distribution_factory.get_dist(flat_outputs)
        
        return -output_dist.log_prob(flat_data)


## Model ##


class BFN(nn.Module):
    def __init__(self, net: nn.Module, bayesian_flow: BayesianFlow, loss: Loss):
        super().__init__()
        
        self.net = net
        self.bayesian_flow = bayesian_flow
        self.loss = loss

    @staticmethod
    @torch.no_grad()
    def sample_t(data: Tensor, n_steps: Optional[int]) -> Tensor:
        """采样时间变量 t, 包括连续时间和离散时间两种情况."""
        
        # 连续时间情况不需要指定总步数, 从 U(0,1) 连续型均匀分布中采样.
        if n_steps == 0 or n_steps is None:
            # (B,1)
            t = torch.rand(data.size(0), device=data.device).unsqueeze(-1)
        # 离散时间情况则先从 U{0,n-1} 离散型均匀分布采样出时间步，然后再除总步数 n 计算出对应的时间变量值: t = \frac{i-1}{n}
        # 注意, 这是每个区间起始时刻的值.
        else:
            # (B,1)
            t = torch.randint(0, n_steps, (data.size(0),), device=data.device).unsqueeze(-1) / n_steps
        # 扩展至和数据同样的维度, 不同的数据样本的时间变量不一致, 同一个样本内所有维度上所对应的时间变量则相同.
        t = (torch.ones_like(data).flatten(start_dim=1) * t).reshape_as(data)
        
        return t

    def forward(
        self, data: Tensor, t: Optional[Tensor] = None, n_steps: Optional[int] = None
    ) -> tuple[Tensor, dict[str, Tensor], Tensor, Tensor]:
        """
        Compute an MC estimate of the continuous (when n_steps=None or 0) or discrete time KL loss.
        t is sampled randomly if None. If t is not None, expect t.shape == data.shape.
        
        使用蒙特卡洛方法估计发送者分布和接收者分布之间的 KL 散度损失:
        -采样时间变量;
        -从贝叶斯流分布中采样得到输入分布的参数(后验更新);
        -将输入分布的参数喂给模型;
        -模型输出形成输出分布;
        -计算连续/离散时间 loss.
        """

        t = self.sample_t(data, n_steps) if t is None else t
        
        # sample input parameter flow
        # 从贝叶斯流分布中采样出输入分布的参数(代表已完成后验更新).
        input_params = self.bayesian_flow(data, t)
        # 在输入模型前转换为适合于模型输入的形式(如有必要的话)
        net_inputs = self.bayesian_flow.params_to_net_inputs(input_params)
        # compute output distribution parameters
        # 注意, 这里模型输出的通常不是输出分布的参数, 而是某些变量(比如估计的噪声),
        # 它们经过后处理才最终成为输出分布的参数.
        output_params: Tensor = self.net(net_inputs, t)

        # compute KL loss in float32
        with torch.autocast(device_type=data.device.type if data.device.type != "mps" else "cpu", enabled=False):
            if n_steps == 0 or n_steps is None:
                loss = self.loss.cts_time_loss(data, output_params.float(), input_params, t)
            else:
                loss = self.loss.discrete_time_loss(data, output_params.float(), input_params, t, n_steps)

        # loss shape is (batch_size, 1)
        return loss.mean()

    @torch.inference_mode()
    def compute_reconstruction_loss(self, data: Tensor) -> Tensor:
        """计算重构损失, 仅当作指标, 不参与训练."""
        
        # 重构损失仅发生在最后时刻 t=1
        t = torch.ones_like(data).float()
        input_params = self.bayesian_flow(data, t)
        net_inputs = self.bayesian_flow.params_to_net_inputs(input_params)
        output_params: Tensor = self.net(net_inputs, t)
        
        return self.loss.reconstruction_loss(data, output_params, input_params).flatten(start_dim=1).mean()

    @torch.inference_mode()
    def sample(self, data_shape: tuple, n_steps: int) -> Tensor:
        device = next(self.parameters()).device
        
        # 起始时刻的先验
        input_params = self.bayesian_flow.get_prior_input_params(data_shape, device)
        distribution_factory = self.loss.distribution_factory

        for i in range(1, n_steps):
            # t_{i-1} = \frac{i-1}{n}
            t = torch.ones(*data_shape, device=device) * (i - 1) / n_steps
            
            # 模型接收输入分布的参数并预测，形成输出分布的参数后，再从其中采样作为预测(生成)的数据样本.
            output_params = self.net(self.bayesian_flow.params_to_net_inputs(input_params), t)
            output_sample = distribution_factory.get_dist(output_params, input_params, t).sample()
            output_sample = output_sample.reshape(*data_shape)
            
            # 计算精度 \alpha_i
            alpha = self.bayesian_flow.get_alpha(i, n_steps)
            # 采样观测样本
            y = self.bayesian_flow.get_sender_dist(output_sample, alpha).sample()
            # 后验更新
            input_params = self.bayesian_flow.update_input_params(input_params, y, alpha)

        t = torch.ones(*data_shape, device=device)
        output_params = self.net(self.bayesian_flow.params_to_net_inputs(input_params), t)
        # 概率分布的众数(mode)作为样本.
        output_sample = distribution_factory.get_dist(output_params, input_params, t).mode
        output_sample = output_sample.reshape(*data_shape)
        
        return output_sample
