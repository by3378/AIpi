# [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

## 2 Methods

We explore RAG models, which **use the input sequence $x$ to retrieve text documents $z$ and use them ==as additional context== when generating the target sequence $y$**. As shown in Figure 1, our models leverage two components: (i) a retriever $p_\eta(z \mid x)$ with parameters $\eta$ that returns (top-K truncated) distributions over text passages given a query $x$ and (ii) a generator $p_\theta\left(y_i \mid x, z, y_{1: i-1}\right)$ parametrized by $\theta$ that generates a current token based on a context of the previous $i-1$ tokens $y_{1: i-1}$, the original input $x$ and a retrieved passage $z$.
> Retriver $p_\eta(z \mid x)$根据输入$x$检索文本文档$z$。  
  Generator $p_\theta\left(y_i \mid x, z, y_{1: i-1}\right)$根据$x$、$z$和之前的输出$y_{1: i-1}$生成$y_i$。

To train the retriever and generator **==end-to-end==**, we treat the retrieved document as a **latent variable**. We propose two models that **==marginalize==** over the latent documents in different ways to produce a distribution over generated text. In one approach, **==RAG-Sequence==**, the model **uses the same document to predict each target token**. The second approach, **==RAG-Token==**, can **predict each target token based on a different document**. In the following, we formally introduce both models and then describe the $p_\eta$ and $p_\theta$ components, as well as the training and decoding procedure.
> End-to-end: 端到端，整个系统从输入到输出由一个统一的模型直接学习。此处即联合优化检索器和生成器。  
  Marginalize:加权求和消去隐变量
$$p(y \mid x)=\sum_z p(z \mid x) \cdot p(y \mid x, z)$$

### 2.1 Models

**RAG-Sequence Model**  
**The RAG-Sequence model uses the same retrieved document to generate the complete sequence**. Technically, it **treats the retrieved document as a single latent variable** that is marginalized to get the seq2seq probability $p(y \mid x)$ **via a top-K approximation**. Concretely, the top K documents are retrieved using the retriever, and the generator **produces the output sequence probability for each document, which are then marginalized,**

$$
\begin{aligned}
p_{\text {RAG-Sequence }}(y \mid x)& \approx \sum_{z \in \operatorname{top-k}(p(\cdot \mid x))} p_\eta(z \mid x) p_\theta(y \mid x, z)\\
&=\sum_{z \in \operatorname{top-k}(p(\cdot \mid x))} p_\eta(z \mid x) \prod_i^N p_\theta\left(y_i \mid x, z, y_{1: i-1}\right)
\end{aligned}
$$

> $p_\eta(z \mid x)$：某篇文档的权重  
  $\prod_i^N p_\theta\left(y_i \mid x, z, y_{1: i-1}\right)$：基于某篇文档，逐词生成整个序列的概率

**RAG-Token Model**  
In the RAG-Token model we can **draw a different latent document for each target token and marginalize accordingly**. This allows the generator to choose content from several documents when producing an answer. Concretely, the top K documents are retrieved using the retriever, and then the generator produces a distribution for the next output token for each document, before marginalizing, and repeating the process with the following output token, Formally, we define:

$$
p_{\text {RAG-Token }}(y \mid x) \approx \prod_i^N \sum_{z \in \operatorname{top-k}(p(\cdot \mid x))} p_\eta(z \mid x) p_\theta\left(y_i \mid x, z, y_{1: i-1}\right)
$$

> RAG-Sequence：生成每个序列时，先对每个文档求序列积，再对文档加权平均  
  RAG-Token：生成每个词时，先对每个文档加权平均，再求序列积

Finally, we note that RAG can be used for sequence classification tasks by considering the target class as **a target sequence of length one**, in which case RAG-Sequence and RAG-Token are equivalent.

### 2.2 Retriever: DPR

The retrieval component $p_\eta(z \mid x)$ is based on DPR. DPR follows a bi-encoder architecture:
$$p_\eta(z \mid x) \propto \exp \left(\mathbf{d}(z)^{\top} \mathbf{q}(x)\right) \quad \mathbf{d}(z)=\operatorname{BERT}_d(z), \quad \mathbf{q}(x)=\operatorname{BERT}_q(x)$$

where **$\mathbf{d}(z)$ is a dense representation of a document produced by a $\operatorname{BERT}_{\text {BASE}}$ document encoder** [8], and $\mathbf{q}(x)$ **a query representation produced by a query encoder, also based on $\operatorname{BERT}_{\text {BASE. }}$**. Calculating top- $\mathrm{k}\left(p_\eta(\cdot \mid x)\right)$, the list of $k$ documents $z$ with highest prior probability $p_\eta(z \mid x)$, is a **==Maximum Inner Product Search== (MIPS)** problem, which can be approximately solved in **sub-linear time** [23]. We use a pre-trained bi-encoder from DPR to initialize our retriever and to build the document index. This retriever was trained to retrieve documents which contain answers to TriviaQA [24] questions and Natural Questions [29]. We refer to the document index as the **==non-parametric memory==**.
> DPR: Dense Passage Retrieval，双编码器架构，即query和document分别用两个独立的编码器编码。  
  通过稠密向量的内积检索相似文档。

### 2.3 Generator: BART

The generator component $p_\theta\left(y_i \mid x, z, y_{1: i-1}\right)$ could be modelled using any encoder-decoder. We use BARTlarge [32], a pre-trained seq2seq transformer [58] with 400 M parameters. To combine the input $x$ with the retrieved content $z$ when generating from BART, we **simply ==concatenate== them**. **BART was pre-trained using a ==denoising objective== and a variety of different noising functions**. It has obtained state-of-the-art results on a diverse set of generation tasks and outperforms comparably-sized T5 models [32]. We refer to the BART generator parameters $\theta$ as the **==parametric memory==** henceforth.
> 直接拼接输入与检索文档。使用BART作为生成器。

### 2.4 Training

We **jointly train the retriever and generator** components without any direct supervision on what document should be retrieved. Given a fine-tuning training corpus of input/output pairs ( $x_j, y_j$ ), **we minimize the negative marginal $\log$-likelihood of each target**, $\sum_j-\log p\left(y_j \mid x_j\right)$ using stochastic gradient descent with Adam [28]. Updating the document encoder $\mathrm{BERT}_d$ during training is costly as it requires the document index to be periodically updated as REALM does during pre-training [20]. We do not find this step necessary for strong performance, and **keep the document encoder (and index) fixed**, only fine-tuning the query encoder $\mathrm{BERT}_q$ and the BART generator.

> 通过最大似然方法同时训练检索器（保持文档编码器不变）和生成器。

### 2.5 Decoder

At test time, RAG-Sequence and RAG-Token require different ways to approximate $\arg \max _y p(y \mid x)$.

**RAG-Token**
The RAG-Token model can be seen as **a standard, autoregressive seq2seq generator** with transition probability: $p_\theta^{\prime}\left(y_i \mid x, y_{1: i-1}\right)=\sum_{z \in \operatorname{top}-k(p(\cdot \mid x))} p_\eta\left(z_i \mid x\right) p_\theta\left(y_i \mid x, z_i, y_{1: i-1}\right)$ To decode, we can plug $p_\theta^{\prime}\left(y_i \mid x, y_{1: i-1}\right)$ into a standard beam decoder.

**RAG-Sequence**
For RAG-Sequence, **the likelihood $p(y \mid x)$ does not break into a conventional per- token likelihood**, hence we cannot solve it with a single beam search. Instead, **we run beam search for each document $z$**, scoring each hypothesis using $p_\theta\left(y_i \mid x, z, y_{1: i-1}\right)$. This yields a set of hypotheses $Y$, **some of which may not have appeared in the beams of ==all== documents**. To estimate the probability of an hypothesis $y$ we run **an additional forward pass for each document $z$ for which $y$ does not appear** in the beam, multiply generator probability with $p_\eta(z \mid x)$ and then sum the probabilities across beams for the marginals. We refer to this decoding procedure as "**==Thorough Decoding==**." For longer output sequences, $|Y|$ can become large, requiring many forward passes. For more efficient decoding, we can **make a further approximation that $p_\theta\left(y \mid x, z_i\right) \approx 0$ where $y$ was not generated during beam search from $x, z_i$**. This avoids the need to run additional forward passes once the candidate set $Y$ has been generated. We refer to this decoding procedure as "**==Fast Decoding==**."

> Beam Search: 在每一步生成时，保留当前得分最高的 B 个部分序列。  
  Thorough Decoding：对每个候选答案 y ，计算它在所有检索文档 z 下的生成概率（对未生成 y 的 z ，需额外前向传播）。
  Fast Decoding：只计算 y 在那些 beam 中包含它的文档 z 下的概率，其余视为零。
