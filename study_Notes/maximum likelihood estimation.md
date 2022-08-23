https://zhuanlan.zhihu.com/p/61593112

## 极大似然估计

极大似然估计是典型的频率学派观点，其基本思想为：待估计的参数$\theta$是客观存在的，只是未知而已，当$\hat \theta_{mle}$满足$\theta=\hat\theta_{mle}$时，该组观测样本$(x_1,\dots,x_n)$更容易被观测到，我们就说$\hat\theta_{mle}$是$\theta$的极大似然估计值。即：估计值$\hat\theta_{mle}$使得事件的发生可能性最大。

下面我们给出极大似然估计的数学描述：

$$
\begin{aligned}L(\theta|x)=f(x|\theta)&=f(x_1,\dots,x_n|\theta)=\prod_{i=1}^{n}{f(x_i|\theta)}\\
\hat\theta_{mle}&=\argmax_{\theta}L(\theta|x)\end{aligned}
$$

## 贝叶斯估计

贝叶斯估计是典型的贝叶斯学派观点，它的基本思想为：待估计参数$\theta$也是随即的，和一般随机变量没有本质区别，因此只能根据样本估计参数$\theta$的分布。

贝叶斯估计利用了贝叶斯公式，因此我们先给出贝叶斯公式的数学描述：

$$
P(B_i|A)=\frac{P(B_i)P(A|B_i)}{P(A)}=\frac{P(B_i)P(A|B_i)}{\sum_{j=1}^{n}{P(B_j)P(A|B_j)}}
$$

下面我们给出贝叶斯估计的数学描述：

$$
\begin{aligned}\pi(\theta|x)=&\frac{\pi(\theta)f(x|\theta)}{m(x)}=\frac{\pi(\theta)f(x|\theta)}{\int f(x|\theta)\pi(\theta)d\theta}\\
\\&\hat \theta_{be}=E{\pi(\theta|x)}\end{aligned}
$$

其中$\pi(\theta)$为参数$\theta$的先验分布，$\pi(\theta|x)$为参数$\theta$的后验分布。因此，贝叶斯估计可以看作是在假定$\theta$服从$\pi(\theta)$的先验分布的前提下，根据样本信息去校正先验分布从而得到后验分布$\pi(\theta|x)$。由于后验分布是一个条件分布，通常我们取后验分布的期望作为参数的估计值。

### 最大后验估计

在贝叶斯分布中，如果我们采用极大似然估计的思想，考虑后验分布最大化从而求解$\theta$，那么这就是最大后验估计：

$$
\hat \theta_{map}=\argmax_{\theta}\pi(\theta|x) = \argmax_{\theta} \frac{f(x|\theta)\pi(\theta)}{m(x)} = \argmax_{\theta} f(x|\theta) \pi(\theta)
$$

$m(x)$是参数$x$的概率分布，与$\theta$无关，因此这里我们省略了$m(x)$以简化计算。

> 作为贝叶斯估计的一种近似解，MAP有其存在的价值，因为贝叶斯估计中后验分布的计算往往是非常棘手的；而且，MAP并非简单地回到极大似然估计，它依然利用了来自先验的信息，这些信息无法从观测样本获得。

对上式稍作处理可得：

$$
\hat \theta_{map}  = \argmax_{\theta} f(x|\theta) \pi(\theta) = \argmax_{\theta}(\log{f(x|\theta)}+\log{\pi(\theta)})
$$
