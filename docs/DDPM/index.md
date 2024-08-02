<link rel="stylesheet" href="../../css/counter.css" />

# Denoising Diffusion Probabilistic Model

## 扩散模型与 DDPM

扩散模型以其训练稳定性、生成数据的多样性和高质量受到广泛关注，迅速成为目前的生成式模型前沿。DDPM (Denoising Diffusion Probabilistic Model) 是最基本的扩散模型，如下图所示：

<div style="text-align:center;">
<img src="graph/DDPM.png" alt="The directed graphical model considered in DDPM" style="margin: 0 auto; zoom: 50%;"/>
</div>

设原始的真实图像为 $x_0$，分 $T$ 步对其依次加单位高斯噪声 ($\mathcal{N}(0, I)$)，可以近似认为得到的是一个纯高斯噪声。如果能够预测出每步添加的噪声，那么将可以从一个纯高斯噪声逐步去噪生成一张新的真实图像。

## Dataset: MNIST

MNIST 数据集 (Mixed National Institute of Standards and Technology database) 是美国国家标准与技术研究院收集整理的大型手写数字数据库，包含 60,000 个示例的训练集以及 10,000 个示例的测试集。

<div style="text-align:center;">
<img src="graph/MNIST.jpeg" alt="How to Train a Model with MNIST dataset | by Abdullah Furkan Özbek | Medium" style="margin: 0 auto; zoom: 50%;"/>
</div>

一般给出的 MNIST 数据集下载链接为 http://yann.lecun.com/exdb/mnist/index.html，然而目前需要登录验证，因此使用 `torchvision.datasets` 的方法准备该数据集。

不同于常见的深度学习入门使用 LeNet-5 在 MNIST 上进行分类，本实验将基于 DDPM 建模 MNIST 手写数字的数据分布，从而能够采样**生成新的手写数字图片**。

## Preliminaries

请自行学习扩散模型的基础理论，可以参考的资料有

- Lilian Weng 的博客：[What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- 最初的 DDPM 论文：[Denoising Diffusion Probabilistic Models](http://arxiv.org/abs/2006.11239)
- DDIM 论文：[Denoising Diffusion Implicit Models](http://arxiv.org/abs/2010.02502)

### Important Formula

在此直接给出一些重要结论：

- **加噪公式**：$x_t = \sqrt{\bar{\alpha}_t} \cdot x_{t-1} + \sqrt{1 - \bar{\alpha}_t} \cdot \varepsilon$

对于这里的符号，固定加噪过程会预先指定 $\beta_t$ 如下：

$$
q(x_t|x_{t-1})=\mathcal{N}(\sqrt{1-\beta_t}x_{t-1},\beta_tI)
$$

$\alpha_t = 1 - \beta_t$ 是关于 $t$ 单调递减的，$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$。$\varepsilon\sim \mathcal{N}(0, I)$。

- **DDPM 去噪公式**：$x_{t-1}$ 从 $q(x_{t-1}|x_t)$ 中采样，该分布近似为高斯分布 $\mathcal{N}(\tilde{\mu}_t, \tilde{\beta}_tI)$，其中

$$
\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1 - \alpha_t}{\sqrt{1-\bar{\alpha}_t}} \varepsilon_ {\theta}^{(t)}(x_ t)\right), \quad \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t
$$

- **DDIM 去噪公式**：

$$
    x_{t-1} = \sqrt{\bar{\alpha}_ {t-1}}
    \underbrace{\left( \frac{x_t-\sqrt{1-\bar{\alpha}_ t}\varepsilon_ {\theta}^{(t)}(x_t)}{\sqrt{\bar{\alpha}_ t}} \right)}_ {\text{“ predicted }x_0\text{”}}
    + \underbrace{\sqrt{1-\bar{\alpha}_ {t-1}-\sigma_ t^2}\varepsilon_ {\theta}^{(t)}(x_ t)}_ {\text{“direction pointing to }x_t\text{”}}
    + \underbrace{\sigma_ t \varepsilon_ t}_ {\text{random noise}}
$$

> $\varepsilon_ {\theta}^{(t)}(x_ t)$ 是模型根据 $x_t$ 预测出的噪声，$\varepsilon_ t$ 是随机采样的单位高斯噪声

其中 $\sigma_t^2=\eta \tilde{\beta}_t$ 是去噪过程中的噪声方差，注意到有

- $\eta = 1$ 时，就等价于 DDPM 的去噪公式
- $\eta = 0$ 时，就是 DDIM 的去噪公式，不确定项 $\varepsilon_ t$ 不再存在
- $\eta \in (0, 1)$ 既不是 DDPM 也不是 DDIM

注意去噪公式中的 $t-1$ 可以变成 $t-k$，在需要加速推断时。

### Training

实际训练时，我们随机生成一系列时间 $t$，根据这些 $t$ 生成随机单位高斯噪声对数据 $x_0$ 进行加噪得到 $x_t$ ，然后使用模型预测噪声 $\varepsilon_ {\theta}^{(t)}(x_ t)$，与噪声真值计算 MSE 损失，通过反向传播更新模型参数。

然而，根据论文 [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512)，让模型预测一个速度 $v$ 会比直接预测噪声 $\varepsilon$ 具有更好的数值稳定性。训练时，需要根据原始数据 $x_0$ 和噪声 $\varepsilon$ 计算出模型所需要预测的速度 $v$：

$$
v = \sqrt{\overline{\alpha}_t} \varepsilon - \sqrt{1-\overline{\alpha}_t} x_0
$$

然后和模型预测的速度 $v_{\theta}^{(t)}(x_ t)$ 计算 MSE 损失即可。

> 关于 v-prediction 的更多内容，除了[原论文](https://arxiv.org/abs/2202.00512)外，也可以参考[本人的笔记](https://zhoutimemachine.github.io/note/readings/diffusion/v-prediction/)

### Inference

从 $x_T$ 开始，一步步去噪得到 $x_0$ 即可。对于 epsilon-prediction 模型（直接预测噪声），直接使用前面的去噪公式即可，只需要在需要加速推断时增大去噪的步长。

对于 v-prediction，关注 DDIM 去噪公式，只需要修改其中的 predicted $x_0$，即

$$
x_0 = \sqrt{\overline{\alpha}_t} x_t - \sqrt{1-\overline{\alpha}_t} v
$$

## Tasks

1. 补全 `scheduler.py` 中所有的 TODO，使得通过给定的 `unet.py` 和 `train.py` 可以训练得到 epsilon-prediction 或 v-prediction 的模型
2. 自己写一个 `infer.py`，能够载入预训练的 epsilon-prediction 或者 v-prediction 模型，推断生成新的手写数字图片
3. 研究模型预测类型、推断步数以及 DDIMScheduler 的 $\eta$ 参数对生成的影响

## Download Links

本实验暂时不公开完整内容，如果有需要可以联系 [ZhouTimeMachine](https://github.com/ZhouTimeMachine)

- train.py
- unet.py
- scheduler.py

## References

- [MNIST dataset](http://yann.lecun.com/exdb/mnist/index.html) (verification is needed)
- Lilian Weng 的博客：[What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- 最初的 DDPM 论文：[Denoising Diffusion Probabilistic Models](http://arxiv.org/abs/2006.11239)
- DDIM 论文：[Denoising Diffusion Implicit Models](http://arxiv.org/abs/2010.02502)
- v-prediction 论文：[Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512)
- 本人[关于 v-prediction 的笔记](https://zhoutimemachine.github.io/note/readings/diffusion/v-prediction/)