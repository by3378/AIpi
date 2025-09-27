# [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

## 4 OUR METHOD

We describe the simple design of LoRA and its practical benefits. The principles outlined here apply to any dense layers in deep learning models, though we only focus on certain weights in Transformer language models in our experiments as the motivating use case.

### 4.1 LOW-RANK-PARAMETRIZED UPDATE MATRICES

> A neural network contains many dense layers which perform matrix multiplication. The weight matrices in these layers typically have **full-rank**. When adapting to a specific task, Aghajanyan et al. (2020) shows that the pre-trained language models have **a low "instrisic dimension"** and can still learn efficiently despite **a random projection to a smaller subspace**.

$$\theta=\theta_0+R \cdot z$$

- $\theta \in \mathbb{R}^N$ ：模型的参数（N非常大，例如 1 亿）
- $d$ ：目标低维子空间的维度 $(d \ll N$ ，例如5，000）
- $R \in \mathbb{R}^{N \times d}$ ：随机投影矩阵
- $z \in \mathbb{R}^d$ ：低维空间中的可优化参数向量
- $\theta_0$ ：预训练模型的原始参数

$R$ 的列向量张成了 $\mathbb{R}^N$ 中的一个 $d$ 维子空间。$R \cdot z$总是落在这个$d$维子空间内。即将搜索空间限制在低维子空间内。

> Inspired by this, we hypothesize the updates to the weights also have a low "intrinsic rank" during adaptation. For a pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, we constrain its update by representing the latter with a lowrank decomposition $W_0+\Delta W=W_0+B A$, where $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$, and the $\operatorname{rank} r \ll \min (d, k)$. During training, $W_0$ is frozen and does not receive gradient updates, while $A$ and $B$ contain trainable parameters. Note both $W_0$ and $\Delta W=B A$ are multiplied with the same input, and their respective output vectors are summed coordinate-wise. For $h=W_0 x$, our modified forward pass yields:

$$
h=W_0 x+\Delta W x=W_0 x+B A x
$$

- $W_0 \in \mathbb{R}^{d \times k}$：原始权重矩阵，微调中被冻结
- $B \in \mathbb{R}^{d \times r}$：低秩矩阵，包含可训练参数，初始化为零（初始时$\Delta W=B A=0$）
- $A \in \mathbb{R}^{r \times k}$：低秩矩阵，包含可训练参数，随机高斯初始化
- $\operatorname{rank} r \ll \min (d, k)$


> We illustrate our reparametrization in Figure 1. We use a random Gaussian initialization for $A$ and zero for $B$, so $\Delta W=B A$ is zero at the beginning of training. We then scale $\Delta W x$ by $\frac{\alpha}{r}$, where $\alpha$ is a constant in $r$. When optimizing with Adam, tuning $\alpha$ is roughly the same as tuning the learning rate if we scale the initialization appropriately. As a result, we simply set $\alpha$ to the first $r$ we try and do not tune it. This scaling helps to reduce the need to retune hyperparameters when we vary $r$ (Yang \& Hu, 2021).

$$\frac{\alpha}{r} \Delta W x$$

- $\alpha$ ：超参数，通常选定为第一次尝试的秩 $r$，但不随 $r$ 变化，使得不同的$r$值可以使用相同的学习率。

随着可训练参数数量的增加，训练 LoRA 大致收敛到训练原始模型。  
LoRA不会使得推理增加额外的开销。  
切换到其他任务时只需要减去并添加不同的 $B^{\prime} A^{\prime}$。  
原则上，我们可以将 LoRA 应用于神经网络中权重矩阵的任何子集。

----

> Note that putting all the parameters in $\Delta W_q$ or $\Delta W_k$ results in significantly lower performance, while adapting both $W_q$ and $W_v$ yields the best result. This suggests that even a rank of four captures enough information in $\Delta W$ such that it is preferable to adapt more weight matrices than adapting a single type of weights with a larger rank.  

> Table 6 shows that, surprisingly, LoRA already performs competitively with a very small $r$ (more so for $\left\{W_q, W_v\right\}$ than just $W_q$ ). This suggests the update matrix $\Delta W$ could have a very small "intrinsic rank". ${ }^6$ To further support this finding, we check the overlap of the subspaces learned by different choices of $r$ and by different random seeds. We argue that increasing $r$ does not cover a more meaningful subspace, which suggests that a low-rank adaptation matrix is sufficient.

同时微调Q与K会带来更好的效果。广度优于深度。递增$r$并没有涵盖更有意义的子空间。