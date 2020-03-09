## [CS224N Summary 1] Word Vector的前世今生



本文将回顾以下几个方面（展示图片等结果均来自于斯坦佛公开课[CS224N](http://web.stanford.edu/class/cs224n/)）：

1. 词向量的前世 - 发展史
   * WordNet
   * Discrete Symbols - one hot 
   * Distributional Semantics
2. 词向量的今生：General的Word2Vec长什么样？
3. 怎么优化求解？
4. Word2Vec常用的模型：Skip-Gram, CBOW, GloVe 以及 Count Based Methods



### 一. 词向量的前世

#### 1.1 WordNet

很早之前学者们就在思考如果让计算机理解文字的meaning，最早的解决办法是WordNet，它是一个覆盖范围宽广的词汇语义网，包含同义词集和上位词（“是”关系，比如“苹果”是“水果”中的“水果”就是上位词）的列表。比如在下图中就展示了“good”的同义词集合以及“panda”的上位词

<img src="img/image-20200309191551631.png" alt="image-20200309191551631" style="zoom:67%;" />

这种方法不可否认的存在一些问题：

1. 缺少词语细微的差别（比如某些词语在特定语境下是同义词）
2. 缺少词语的新的含义，专家不可能一直持续维护这个表格
3. 主观的
4. 建立这个词语列表/网络需要很高的人力成本/专家知识
5. 没办法计算word similarity



#### 1.2 One-Hot Encoding - Representing words as discrete symbols

> "In traditional NLP, we regard words as discrete symbols: hotel, conference, motel – a **localist** representation"

在最早的NLP task中，我们倾向于对词语进行one-hot（独热）编码

<img src="img/image-20200309192302255.png" alt="image-20200309192302255" style="zoom:50%;" />

缺点：

1. 两个词语的similarity还是无法表示：因为两个向量是完全正交的

   解决方案：使用上文提到的wordnet去考虑词语的similarity，这种方法虽然可行但是不完备

2. Vector dimension会非常大：整个向量的长度是语料库中出现的不重复的词语的个数，会非常大，并且比较稀疏



### 1.3 Word2Vec - Representing words by their context

最终为了解决上述问题，科学家借用了著名的语义学家Firth在1957年提出的分布语义（distributional semantics），即一个词语的含义是由频繁出现在他周围的词语给出的。

> "You shall know a word by the company it keeps" (J.R. Firth 1957:11)

WordVector的核心思想就是对每个词语建立一个dense vector，使得它与出现在他上下文的词语比较相似。注意在此处有两个核心：context（上下文）以及similarity（相似）

> Note: word vectors are sometimes called word embeddings or word representations. They are a distributed representation.

接下来我们就详细介绍一下word2vector



### 二. 词向量的今生：Word2Vec

Word2Vec的核心思想主要分为以下几步：

1. 每个词语都被一个维度为d的向量代表（可以随机初始化这个向量）
2. 遍历语料库的文本中的每一个词语，当前这个词语为中心词c，他周围的一些词语为o，我们对c的词向量和周围某个词语o的词向量计算出相似度（比如点积），然后计算出在已知中心词c的情况下o出现的概率
3. 不断的调整词向量使得整体概率最大

下图给出了一个例子：

<img src="img/image-20200309194132436.png" alt="image-20200309194132436" style="zoom:50%;" />

其中into就是中心词c，假设window size=2，即我们要考虑他前后各两个词语，我们把上述问题转化成一个数学问题，即如下所示：
$$
Likelihood=L(\theta) = \prod_{t=1}^T\prod_{-m\leq j\leq m, \ j\neq0}P(w_{t+j}|w_t;\theta)
$$
其中，t代表当前遍历的中心词位置，m代表window size，$\theta$是待优化的所有变量，由于在优化理论里一般都是求目标函数的最小值，加之连加比连乘形式简单，我们对上述公式进行变换得到目标函数（损失函数）：
$$
J(\theta) = -\frac{1}{T}\log L(\theta)=-\frac{1}{T}\sum_{t=1}^T\sum_{-m\leq j\leq m, \ j\neq0}\log P(w_{t+j}|w_t;\theta)
$$
接下来的问题就很明显了，我们如何计算$P(w_{t+j}|w_t;\theta)$呢？

解决方法：我们假设每一个词语$w$都有两个向量表示它，当它是中心词的时候我们用$v_w$，当它是上下文词的时候我们用$u_w$，接下来对于一个中心词$c$和它周围的某一个上下文词$o$，
$$
P(o|c)=\frac{\exp(u_o^Tv_c)}{\sum_{w\in V}\exp(u_w^Tv_c)}
$$
分子使用两个向量的点积代表了这个中心词与这个上下文词的相似度，而分母则是这个中心词与其他所有词语的相似度求和。我们希望最终得到的词向量使得$P(o|c)$这个概率比较大， 则暗示了相比于其他词语，这个上下文词出现在这个中心词旁边的概率比较大。

那您可能会问了加上exp是为啥呢? 原因很简单，保证概率为正，同时分母作为正则项也保证了概率和为1

<img src="img/image-20200309200333061.png" alt="image-20200309200333061" style="zoom:50%;" />

其实上述这个形式也被称之为softmax函数（we will be good friends later :) ）

<img src="img/image-20200309200527104.png" alt="image-20200309200527104" style="zoom:50%;" />

最终，这个关于词向量的优化问题就可以被写成：
$$
J(\theta) = -\frac{1}{T}\sum_{t=1}^T\sum_{-m\leq j\leq m, \ j\neq0}\log P(w_{t+j}|w_t;\theta) \\
= -\frac{1}{T}\sum_{t=1}^T\sum_{-m\leq j\leq m, \ j\neq0}\log \frac{\exp(u_{t+j}^Tv_t)}{\sum_{w\in V}\exp(u_w^Tv_t)}
$$
此处的$\theta$就是所有的$u$和$w$了～

### 三. 如何求解：优化方法哪家强？

当时是梯度下降法！（Gradient Descent Method）

其实这里就会涉及到很多数学优化方面的知识了，在这里就不赘述。（可以写上三天三夜，不了解的朋友们可以知乎搜索“什么是梯度下降法”查缺补漏）梯度下降法是一阶优化算法，其核心思想就是对于优化问题的变量，只要你按照梯度下降的方向更新它，它就能够保证使目标函数减小，最终能收敛到局部最优点。对于凸问题 (convex problem)，这个局部最优点就是全局最优解啦！

对于这个问题首先就需要求导，我们需要优化的变量是$\theta=[v_1,\cdots,v_V, u_1,\cdots,u_V]$，对于某一个变量$v_t$
$$
\frac{\partial J(\theta) }{\partial v_t} = \frac{\partial }{\partial v_t}(-\frac{1}{T}\sum_{t=1}^T\sum_{-m\leq j\leq m, \ j\neq0}\log \frac{\exp(u_{t+j}^Tv_t)}{\sum_{w\in V}\exp(u_w^Tv_t)})
$$
由于除$v_t$之外的部分都可以被视作常量，因此：
$$
\frac{\partial J(\theta) }{\partial v_t} = -\frac{1}{T} \sum_{-m\leq j\leq m, \ j\neq0} \frac{\partial }{\partial v_t}\log \frac{\exp(u_{t+j}^Tv_t)}{\sum_{w\in V}\exp(u_w^Tv_t)}
$$
其中：
$$
\frac{\partial }{\partial v_t}\log \frac{\exp(u_{t+j}^Tv_t)}{\sum_{w\in V}\exp(u_w^Tv_t)} = \frac{\partial }{\partial v_t}(u_{t+j}^Tv_t) - \frac{\partial }{\partial v_t}\log \sum_{w\in V}\exp(u_w^Tv_t) \\
= u_{t+j} - \frac{1}{\sum_{w\in V}\exp(u_w^Tv_t)}\sum_{w\in V}\frac{\partial}{v_t}\exp(u_w^Tv_t) \\
= u_{t+j} - \frac{1}{\sum_{w\in V}\exp(u_w^Tv_t)}\sum_{w\in V}\exp(u_w^Tv_t)u_w \\
=\frac{\exp(u_{t+j}^Tv_t)}{\sum_{w\in V}\exp(u_w^Tv_t)} = u_{t+j} - \frac{\sum_{w\in V}(\exp(u_w^Tv_t)u_w )}{\sum_{w\in V}\exp(u_w^Tv_t)}
$$
最终可以得到：
$$
\frac{\partial }{\partial v_t}\log\frac{\exp(u_{t+j}^Tv_t)}{\sum_{w\in V}\exp(u_w^Tv_t)} = u_{t+j} - \frac{\sum_{w\in V}(\exp(u_w^Tv_t)u_w )}{\sum_{w\in V}\exp(u_w^Tv_t)} \\= u_{t+j} - \sum_{w\in V}\frac{\exp(u_w^Tv_t)}{\sum_{w\in V}\exp(u_w^Tv_t)}u_w  = u_{t+j} - \sum_{w\in V}P(w|v_t)u_w
$$
这个导数结果的形式非常有趣，我们可以将它解读为此刻这个上下文词$u_{t+j}$的词向量减去当下模型对于其他词语的词向量乘以它出现概率的乘积和（期望）；可以被视为是正确的词语向量（the actual contect word）结果减去预期词（the expected context word），对于其他$v$变量也是同理，使用求导的方法我们就可以得到整个目标函数的导数
$$
\nabla_\theta J(\theta)^T = [\frac{\partial J(\theta)}{\partial v_1}, \cdots, \frac{\partial J(\theta)}{\partial v_V}, \frac{\partial J(\theta)}{\partial u_1}, \cdots, \frac{\partial J(\theta)}{\partial u_V}] 
$$
于是我们就可以使用梯度下降法更新变量了
$$
\theta^{new} = \theta^{old} - \alpha \nabla_\theta J(\theta) \\
\theta^{new}_j = \theta^{old}_j - \alpha \frac{\partial}{\partial \theta_j^{old}} J(\theta)
$$
其中$\alpha$被称为步长（step size）或者学习率（learning rate）

但是由于计算梯度时需要使用整个语料库的数据，而这个量级是十分庞大的，于是在实际中我们会考虑只使用部分数据更新变量，甚至只是使用一个样本更新梯度，前者被称为小批量梯度下降法（Mini-batch Gradient Descent，简称MBGD），后者被称为随机梯度下降法（Stochastic Gradient Descent，简称SGD）



### 四. 子孙满堂的Word2Vec家族

### 4.1 Skip-gram Model

Predict context (“outside”) words (position independent) given center word

这个模型其实与我们之前讲过的word2vec的原始模型并不不同，主要变化是Skip-gram模型采用的负采样（negative sampling）比较巧妙的解决了分母$\sum_{w\in V}$计算量较大的问题，即将这个部分替换成随机挑选的K的词语（k在10-15之间都works fine）

[More details to be updated] 

#### 4.2 Continuous Bag of Words (CBOW) 

Predict center word from (bag of) context words

[More details to be updated] 

### 4.3  Co-occurrence Matrix Factorization 

> **But why not capture co-occurrence counts directly?**
>
> With a co-occurrence matrix *X*
>
> - 2 options: windows vs. full document
> - Window: Similar to word2vec, use window around each wordàcaptures both syntactic (POS) and semantic information
> - Word-document co-occurrence matrix will give general topics (all sports terms will have similar entries) leading to “Latent Semantic Analysis”
>
> --- Slides of Lecture 02

了解LDA或者推荐系统的朋友可能对这种矩阵分解的做法不陌生，这种做法就是计算出词语的共现矩阵，然后对其进行分解（比如SVD奇异值分解）
$$
X = U\times \Sigma \times V^T
$$
<img src="img/image-20200309211747607.png" alt="image-20200309211747607" style="zoom:67%;" />

其中U矩阵即可作为词向量了～

以矩阵分解为代表的Count based和以词向量为代表的Direct Prediction的方法各有利弊：

<img src="img/image-20200309211926184.png" alt="image-20200309211926184" style="zoom:67%;" />

于是GloVe应运而生！

### 4.4 GloVe - Global Vectors for Word Representation

GloVe非常巧妙的构建了词向量（Word Vector）和共现矩阵（Co-ocurrence Matrix）之间的关系，使得最终词向量能够刻画出共现矩阵的大小关系，从而考虑到全局信息

 [More details to be updated] 