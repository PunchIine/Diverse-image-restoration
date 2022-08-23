## Variational Auto Encoder

参考资料：

1. https://spaces.ac.cn/archives/5253

2. http://www.gwylab.com/note-vae.html

3. https://www.bilibili.com/video/av15889450/?

4. https://spaces.ac.cn/archives/5343

如今最常见的两大生成模型就是VAE和GAN，作为生成模型，它们二者的目标基本是一致的，即希望构建一个从隐变量$z$生成目标数据X的模型。更准确的讲，它们是假设了$z$服从某些常见的分布，然后希望训练一个模型$X=g(z)$，使这个模型能够将原来的概率分布映射到训练集的概率分布，也就是说，它们的目的都是进行分布之间的变换。

我们知道VAE的理论基础是高斯混合模型，因此在深入剖析VAE之前，我们先来看看什么是高斯分布（正态分布），标准正态分布又是什么。

高斯分布：

$$
\mathcal N(x;\mu,\sigma^2)=\sqrt{\frac{1}{2\pi\sigma^2}}\exp(-\frac{1}{2\sigma^2}(x-\mu)^2)
$$

<img src="https://user-images.githubusercontent.com/93063038/180803370-4075761d-59e3-4377-ba9e-ccf9b413b65a.png" title="" alt="image" data-align="center">

标准正态分布即均值为0，方差为1的正态分布，即：$\mathcal N(x;0,1)$

VAE要解决的问题是编码器与解码器的构造，那为什么原始的Auto Encoder模型不能满足我们想要构造的生成模型的标准呢。

<img src="https://user-images.githubusercontent.com/93063038/180694504-af2557f3-6622-4097-b6a4-e3840906419b.png" title="" alt="image" data-align="center">

上图可以看到，Deep Auto-Encoder模型把一个高维的向量压缩成较低维的向量从而得到图像的一些深层特征（Internal Representation又称为Bottleneck），然而，对于一个生成模型而言，解码器部分应该是单独能够提取出来的，并且对于在规定维度下任意采样的一个编码，都应该能通过解码器产生一张清晰且真实的图片。简单的Auto Encoder却不能完成这个任务。

<img src="https://user-images.githubusercontent.com/93063038/180717933-29843e46-1609-4b06-8921-f5ee403376a6.png" title="" alt="image" data-align="center">

如上图，我们希望在此分布中任意采样一点（如图中中点），能得到清晰的结果（介于全月与半月之间的月相），然而事实上我们将得到一张模糊的、紊乱的图片。因为我们的Deep Auto-Encoder模型是通过深层神经网络来进行encode和decode的，而这是一个非线性的变换过程，因此code分布中点与点之间的分布是不连续的、没有规律的。

<img src="https://user-images.githubusercontent.com/93063038/180720595-163424ca-a31b-4ba2-af03-fb80fb0745fb.png" title="" alt="image" data-align="center">

想要解决这个问题，我们可以引入噪声，使得图片的编码区域得到扩大，从而掩盖掉失真的空白编码点。由上图可以看到，在加入噪声后训练时能被采样到的范围增大了，但这还不够，因为在距离训练样本encode后得到code的距离更远处（如图中黄点）照样存在失真点，因此我们可以试图把噪音无限拉长，使得对于每一个样本，它的编码会覆盖整个编码空间，不过我们得保证，在原编码附近编码的概率最高，离原编码点越远，编码概率越低。在这种情况下，图像的编码就由原先离散的编码点变成了一条连续的编码分布曲线，如下图所示。

<img src="https://user-images.githubusercontent.com/93063038/180721873-a1484684-0efe-4c9f-aaa9-ce9843ac4b69.png" title="" alt="image" data-align="center">

这就是VAE所要实现的目标。

下面来看看VAE的基本原理与实现思路吧。

<img src="https://user-images.githubusercontent.com/93063038/180727028-c5d9b4f2-673c-42a7-a767-19b91aaaf408.png" title="" alt="image" data-align="center">

许多教程错误地讲解VAE从分布$p(z)$为正态分布出发，描述了一个由$z$生成$x$的模型，此时假设$p(z)$为正态分布，如果这个假设成立，那么我们将能够于$p(z)$中采样一个变量然后由生成器生成样本，如上图所示。但这么一来问题就出现了。我们如何知道采样得到的变量$z_k$对应的是真实样本$x_i$中的哪一个呢？（如：采样得到的$z_4$可能对应的真实样本为$x_2$这样通过对应$x_2$的隐变量却通过生成器得到了$x_4$，如此训练出的模型显然是效果极差的）如此直接最小化距离函数$\mathcal D(\hat{x},x)^2$是不科学的，在完成生成任务时必然是很糟糕的。

事实上，VAE并非假设$p(z)$为正态分布从而得出模型，而是假设$p(z|x)$（一个后验分布）为正态分布。原论文作者也强调了这一点。

<img src="https://user-images.githubusercontent.com/93063038/180731810-577e01c0-bb87-4489-9101-72fb94d77a8b.png" title="" alt="image" data-align="center">

即给定一个样本$x_k$，我们假设专属于这个样本的一个分布$p(z|x_k)$是正态分布。如此我们便能训练一个decoder，将$p(z|x_k)$中采样出的隐变量通过decoder还原成$x_k$，此时我们便可确定隐变量$z_k$与原始样本$x_k$的一一对应关系。

> VAE的encoder的工作可以抽象为求解$p(z|x)$的过程，而decoder的工作可以抽象为求解$p(x|z)$的过程

如此，每一个$x_k$都有了一个对应的正态分布（隐变量分布），此时我们要求解正态分布$p(z|x_k)$只需要确定该分布的两个参数：均值（$\mu$）和方差（$\sigma^2$）。有了目标，我们只需要构建神经网络来拟合它就好啦。我们构建两个神经网络$\mu_k=f_1(x_k)$和$\log\sigma^2=f_2(x_k)$，由于构建$\sigma^2$需要将值限定为非负数，需要添加一个激活函数，因此这里选择拟合$\log\sigma^2$。至此，我们通过得到均值与方差确定了每一个$x_k$对应的正态分布，现在我们貌似可以放心地通过最小化距离$\mathcal D(\hat{x},x)^2$来优化模型了。

<img src="https://user-images.githubusercontent.com/93063038/180769590-6d695d5c-7458-4fba-a729-44849df022f5.png" title="" alt="image" data-align="center">

However，事情没有那么简单！可以看到我们的decoder训练时要完成的任务是重构真实样本，而我们的隐变量$z_k$是在添加了噪声后形成的正态分布中采样的，因此我们的重构过程无疑会受到噪声大小的影响，而我们得到的正态分布的均值与方差都是通过神经网络来拟合的，因此，对应神经网络来说，想要生成更好的样本，最佳的策略就是使对应正态分布的方差最小化（趋于0）（方差衡量了一个分布的离散程度，方差最小，则分布随机性减弱，最后极端情况就是无论如何采样得到的都是均值，即模型退化为了初始的Auto Encoder）。对于这个问题VAE的解决方案是添加一个loss从而使所有的$p(z|x)$向标准正态分布看齐（均值为0方差为1）。

<img src="https://user-images.githubusercontent.com/93063038/180804052-a010bd35-d03e-49d9-8c26-802ba13783ad.png" title="" alt="image" data-align="center">

原论文中采用了KL散度来衡量$\mathcal N(\mu,\sigma^2)$和$\mathcal N(0,1)$之间的差异（作为loss）即$KL(\mathcal N(\mu,\sigma^2)||\mathcal N(0,1))$。

重温一下**KL散度**：

> 描述两个概率分布之间差异的非对称量
> 
> 其定义就是**用理论分布去拟合真实分布时产生的信息损耗**
> 
> 交叉熵，刻画的是实际输出（概率）与期望输出（概率）的距离，也就是交叉熵的值越小，两个概率分布就越接近，即拟合的更好。
> 
> 信息熵，也是平均自信息量，如公式所示，表示的是自信息量(也就是上面提到的信息量)的数学期望，表示为概率与其自信息量的乘积然后再求和。
> 
> 因此，KL散度=交叉熵-信息熵

$$
\begin{aligned}D_{KL}(p||q)&=H(p,q)-H(p)\\
&=-\sum_xp(x)(\log{q(x)}-\log{p(x)})\\
&=-\sum_x{p(x)\log{\frac{q(x)}{p(x)}}}\\
&\iff{E_{x\backsim{p}}(\log{p(x)}-\log{q(x)}})\end{aligned}
$$

下面我们来推导KL散度的计算结果

$$
\begin{aligned}&\ \ \ \ \ KL(\mathcal N(\mu,\sigma^2)||\mathcal N(0,1))\\
&=KL(\sqrt{\frac{1}{2\pi\sigma^2}}\exp(-\frac{1}{2\sigma^2}(x-\mu)^2)||\sqrt{\frac{1}{2\pi}}\exp(-\frac{1}{2}x^2))\\
&=E_{x\backsim{\mathcal N(\mu,\sigma^2)}}(\log({\sqrt{\frac{1}{2\pi\sigma^2}}\exp(-\frac{1}{2\sigma^2}(x-\mu)^2)}-\log{(\sqrt{\frac{1}{2\pi}}\exp(-\frac{1}{2}x^2)))}\\
&=E_{x\backsim{\mathcal N(\mu,\sigma^2)}}(\log{(\frac{\sqrt{\frac{1}{2\pi\sigma^2}}\exp(-\frac{1}{2\sigma^2}(x-\mu)^2)}{\sqrt{\frac{1}{2\pi}}\exp(-\frac{1}{2}x^2)}}))\\
&=\int_{\mathcal N(\mu,\sigma^2)}\sqrt{\frac{1}{2\pi\sigma^2}}\exp(-\frac{1}{2\sigma^2}(x-\mu)^2)(\log{(\frac{\sqrt{\frac{1}{2\pi\sigma^2}}\exp(-\frac{1}{2\sigma^2}(x-\mu)^2)}{\sqrt{\frac{1}{2\pi}}\exp(-\frac{1}{2}x^2)}}))dx\\
&=\int_{\mathcal N(\mu,\sigma^2)}\sqrt{\frac{1}{2\pi\sigma^2}}\exp(-\frac{1}{2\sigma^2}(x-\mu)^2)\log(\sqrt{\frac{1}{\sigma^2}}\exp(\frac{1}{2}[x^2-(x-\mu)^2/\sigma^2]))dx\\
&=\int_{\mathcal N(\mu,\sigma^2)}\sqrt{\frac{1}{2\pi\sigma^2}}\exp(-\frac{1}{2\sigma^2}(x-\mu)^2)[-\log{\sigma^2}dx+x^2-(x-\mu^2)/\sigma^2]dx\\
&=\frac{1}{2}(-\log{\sigma^2}+\mu^2+\sigma^2-1)\end{aligned}
$$

<img src="https://user-images.githubusercontent.com/93063038/180896192-bc4e9ab5-96c8-4b68-bf29-ac7aaf458ae0.png" title="" alt="image" data-align="center">

**reparameterization trick**

由于我们的$z_k$是从特定的正态分布中采样的，因此在训练网络时我们无法对其直接进行求导，采样这个行为是不可导的，但采样的结果是可导的。

由（将$\frac{1}{\sigma}$提出来凑微分）

$$
\begin{aligned}&\ \ \ \ \ \ \sqrt{\frac{1}{2\pi\sigma^2}}\exp(-\frac{1}{2\sigma^2}(x-\mu)^2)dz\\
&=\sqrt{\frac{1}{2\pi}}\exp(-\frac{1}{2\sigma^2}(x-\mu)^2)d(\frac{z-\mu}{\sigma})\end{aligned}
$$

说明$(z−μ)/σ=ε$是服从均值为0、方差为1的标准正态分布的。

<mark>要同时把dz考虑进去，是因为乘上dz才算是概率，去掉dz是概率密度而不是概率</mark>

由此我们可以知道：

> 从$\mathcal N(μ,σ^2)$中采样一个$z$，相当于从$\mathcal N(0,1)$中采样一个$ε$

我们只需让$z=μ+ε×σ$就能将从$\mathcal N(\mu,\sigma^2)$采样，变成了从$\mathcal N(0,1)$中采样后通过参数变换得到$\mathcal N(\mu,\sigma^2)$中采样的结果，如此采样的结果便能参与梯度下降，整个模型也就可以训练了。

> 至此，我们已经对VAE有了大概的了解，但对整个VAE的理论框架，我们的认知还十分模糊，苏神通过直接对联合分布进行近似的方式，简明快捷的给出了VAE的理论框架。

**直面联合分布**

首先，我们知道encoder解决的问题是在我们拥有一批数据样本$\{x_1,\dots,x_n\}$的情况下（整体描述为$x$），希望借助隐变量$z$来描述$x$的分布$p(x)$，其目标可表示为$\tilde{p}(z)=\int{p(z|x)p(x)dx}$，而decoder解决的问题是通过隐变量$z$得到生成样本$\tilde x$，使其近似于$x$。其目标为： $q(x)=\int{q(x|z)q(z)dz}$，这里的$q(z)$是先验分布（假设为标准正态分布），目的是希望$q(x)$能逼近$p(x)$。

> 我们可以把联合分布理解为解空间，即：$q(x,z)$是编码器对应的解空间，$p(x,z)$是解码器对应的解空间。若确定了样本$x$，那么$p(x,z)$是编码器拟合的分布，若确定了先验分布的决策变量$z$那么$q(x,z)$是编码器拟合的分布。

而我们又知道$q(x,z)=q(x|z)q(z)$，即我们希望$p(x,z)$与$q(x,z)$是相同的。我们定义$p(x,z)={p}(x)p(z|x)$，设想以$q(x,z)$来逼近$p(x,z)$，那么我们可以用KL散度来度量他们之间的距离：

$$
KL(p(x,z)||q(x|z))=\int\int{p(x,z)\ln{\frac{p(x,z)}{q(x,z)}dzdx  }}
$$

此时我们希望这两个分布越接近越好，因此我们要最小化KL散度，而此时由于$p(x,z)$和

$q(x,z)$都有参数，所以二者是相互接近，于是可得：

$$
\begin{aligned}KL\Big(p(x,z)\Big\Vert q(x,z)\Big) =& \int{p}(x) \left[\int p(z|x)\ln \frac{{p}(x)p(z|x)}{q(x,z)} dz\right]dx\\ 
=& \mathbb{E}_{x\sim{p}(x)} \left[\int p(z|x)\ln \frac{{p}(x)p(z|x)}{q(x,z)} dz\right] 
\end{aligned}

$$

结合**采样计算**可以写出：

$$
\mathbb{E}_{x\sim p(x)}[f(x)] = \int f(x)p(x)dx \approx \frac{1}{n}\sum_{i=1}^n f(x_i),\quad x_i\sim p(x)
$$

结合二式，又因$\ln \frac{{p}(x)p(z|x)}{q(x,z)}=\ln{p}(x) + \ln \frac{p(z|x)}{q(x,z)}$，可得：

$$
\begin{aligned}\mathbb{E}_{x\sim{p}(x)} \left[\int p(z|x)\ln{p}(x)dz\right] =& \mathbb{E}_{x\sim{p}(x)} \left[\ln{p}(x)\int p(z|x)dz\right]\\ 
=&\mathbb{E}_{x\sim{p}(x)} \big[\ln{p}(x)\big] 
\end{aligned}
$$

> 注意此处$\tilde{p}(x)$是根据样本确定的关于$x$的先验分布，尽管我们不一定能准确地写出它的形式，但他是确定的、存在的，且是一个常数，设其为c。

因此我们可以写出：

$$
\mathcal{L}=KL\Big(p(x,z)\Big\Vert q(x,z)\Big) - c
= \mathbb{E}_{x\sim{p}(x)} \left[\int p(z|x)\ln \frac{p(z|x)}{q(x,z)} dz\right]
$$

由此易知最小化$\mathcal L$即最小化KL散度，且$\mathcal L$具有下界$-\mathbb{E}_{x\sim{p}(x)} \big[\ln{p}(x)\big]$。

> 注意到$\tilde {p}(x)$不一定是概率，在连续情形时$\tilde {p}(x)$是概率密度，它可以大于1也可以小于1，所以$-\mathbb{E}_{x\sim{p}(x)} \big[\ln{p}(x)\big]$不一定是非负，即loss可能是负数。

至此，我们回顾初衷——为了得到生成模型，所以我们将$q(x,z)$卸成$q(x|z)q(z)$，于是就有：

$$
\begin{aligned}\mathcal{L} =& \mathbb{E}_{x\sim {p}(x)} \left[\int p(z|x)\ln \frac{p(z|x)}{q(x|z)q(z)} dz\right]\\ 
=&\mathbb{E}_{x\sim{p}(x)} \left[-\int p(z|x)\ln q(x|z)dz+\int p(z|x)\ln \frac{p(z|x)}{q(z)}dz\right]\\=&\mathbb{E}_{x\sim{p}(x)} \left[\mathbb{E}_{z\sim p(z|x)}\big[-\ln q(x|z)\big]+\mathbb{E}_{z\sim p(z|x)}\Big[\ln \frac{p(z|x)}{q(z)}\Big]\right]\\ 
= &\mathbb{E}_{x\sim{p}(x)} \Bigg[\mathbb{E}_{z\sim p(z|x)}\big[-\ln q(x|z)\big]+KL\Big(p(z|x)\Big\Vert q(z)\Big)\Bigg] 
\end{aligned} \tag{⭐️}
$$

可见括号内就是VAE的损失函数（即可知最小化损失函数的过程可以等价为最小化编码器与解码器解空间的过程），我们的目标就是想办法找到适当的$q(x|z)$和$q(z)$使得$\mathcal L$最小化。

**后验分布近似**

此时$q(z)$，$q(x|z)$，$p(z|x)$全都是未知的，连形式都还没有确定，但想要把模型落到实处，就得把上式的每一项都明确的写出来。而$q(x|z)$和$p(z|x)$我们分别把他们设计为解码器和编码器的任务使用神经网络来拟合。

> 若按照已知$q(x|z)$和$q(z)$，那么$p(z|x)$最合理的估计应该是：
> 
> $$
> \hat{p}(z|x) = q(z|x) = \frac{q(x|z)q(z)}{q(x)} = \frac{q(x|z)q(z)}{\int q(x|z)q(z)dz}
> $$
> 
> 这其实就是EM算法中的后验概率估计的步骤，可以参考(https://spaces.ac.cn/archives/5239) ，但事实上，分母的积分几乎不可能完成，因此这是行不通的，所以我们使用神经网络来近似它。

具体就如前文简易理解VAE时讲到的一样假设$p(z|x)$也是一个正态分布，该分布的均值与方差都由$x$来决定，我们使用神经网络来解决这个“决定”的过程。即：

$$
p(z|x)=\frac{1}{\prod\limits_{k=1}^d \sqrt{2\pi  \sigma_{(k)}^2(x)}}\exp\left(-\frac{1}{2}\left\Vert\frac{z-\mu(x)}{\sigma(x)}\right\Vert^2\right)
$$

其中$\mu(x),\sigma^(x)$就是输入为$x$，输出分别为均值与方差的两个神经网络，（其中$\mu(x)$就起到了类似encoder的作用）（为何不谈$\sigma^2(x)$呢，暂时未理解）。既然我们这里已经假定了高斯分布，那么我们便可以确定上面式子中的KL散度了：

$$
KL\Big(p(z|x)\Big\Vert q(z)\Big)=\frac{1}{2} \sum_{k=1}^d \Big(\mu_{(k)}^2(x) + \sigma_{(k)}^2(x) - \ln \sigma_{(k)}^2(x) - 1\Big)
$$

这也就是上文提到的KL loss。

**生成近似模型**

现在就只剩生成模型部分的$q(x|z)$了，原论文中给出了两种候选方案：伯努利分布或正态分布。

伯努利分布模型：

伯努利分布为一个二元分布

$$
p(\xi)=\left\{\begin{aligned}&\rho,\, \xi = 1;\\ 
&1-\rho,\, \xi = 0\end{aligned}\right.
$$

因此伯努利分布只适用于$x$是一个多元的二值向量的情况，比如$x$是二值图像时（如mnist）。这种情况下，我们用神经网络$\rho(z)$来计算参数$k$，从而得到：

$$
q(x|z)=\prod_{k=1}^D \Big(\rho_{(k)}(z)\Big)^{x_{(k)}} \Big(1 - \rho_{(k)}(z)\Big)^{1 - x_{(k)}}
$$

此时可以算出（取负对数得到信息量）

$$
-\ln q(x|z) = \sum_{k=1}^D \Big[- x_{(k)} \ln \rho_{(k)}(z) - (1-x_{(k)}) \ln \Big(1 -\rho_{(k)}(z)\Big)\Big]
$$

由对数的性质可知，此时$\rho(z)$要压缩至$0\backsim1$之间（如使用sigmoid激活函数），然后使用交叉熵作为损失函数，这里$\rho(z)$就起到了类似decoder的作用。

正态分布模型：

正态分布模型则与$p(z|x)$是一样的，只不过交换了$x$，$z$的位置：

$$
q(x|z)=\frac{1}{\prod\limits_{k=1}^D \sqrt{2\pi  \tilde{\sigma}_{(k)}^2(z)}}\exp\left(-\frac{1}{2}\left\Vert\frac{x-\tilde{\mu}(z)}{\tilde{\sigma}(z)}\right\Vert^2\right)
$$

这里的$\tilde{\mu}(z)$，$\tilde{\sigma}^2(z)$是输入为$z$，输出分别为均值和方差的神经网络，$\tilde{\mu}(z)$就起到了decoder的作用。于是：

$$
-\ln q(x|z) = \frac{1}{2}\left\Vert\frac{x-\tilde{\mu}(z)}{\tilde{\sigma}(z)}\right\Vert^2 + \frac{D}{2}\ln 2\pi + \frac{1}{2}\sum_{k=1}^D \ln \tilde{\sigma}_{(k)}^2(z)
$$

很多时候我们会固定方差为一个常数$\tilde{\sigma}^2$，此时：

$$
-\ln q(x|z) \sim \frac{1}{2\tilde{\sigma}^2}\Big\Vert x-\tilde{\mu}(z)\Big\Vert^2
$$

这就出现了MSE损失函数。

**所以现在就清楚了，对于二值数据，我们可以对decoder用sigmoid函数激活，然后用交叉熵作为损失函数，这对应于$q(x|z)$为伯努利分布；而对于一般数据，我们用MSE作为损失函数，这对应于$q(x|z)$为固定方差的正态分布**

**采样计算技巧**

上面做了这么多工作，都是为了将⭐️式确定下来，我们假设$p(z)$和$p(z|x)$是正态分布之后，$KL(p(x|z)||p(z))$就已经确定下来了，在假设$q(x|z)$为正态分布或伯努利分布之后，$-\ln{q(x|z)}$的值也确定了下来，现在我们只缺少采样了。

$p(z|x)$在整个任务中起到了两个作用，一个是计算KL loss，另一个就是计算$\mathbb E_{z\backsim{p(z|x)}}-\log{q(x|z) }$。后者可以转化为

$$
-\frac{1}{n}\sum_{i=1}^n \ln q(x|z_i),\quad z_i \sim p(z|x)
$$

我们已经假定$p(x|z)$为正态分布了，此时借助上文提到的“重参数技巧”就可以完成采样，但是采样多少个才合适呢？VAE直截了当地规定为1个，此时⭐️式就变得更简单了：

$$
\mathcal{L} = \mathbb{E}_{x\sim \tilde{p}(x)} \Bigg[-\ln q(x|z) + KL\Big(p(z|x)\Big\Vert q(z)\Big)\Bigg],\quad z\sim p(z|x)
$$

> 注意对于一个batch中的每个$x$，都需要从$p(z|x)$采样一个“专属”于$x$的$z$出来才去算$−\ln{q(x|z)}$

至此，我对于VAE已经有了一定的了解了！

（笔记思路还不是很清晰，完全在跟着苏神走，有空得理清思路重写一篇！感谢苏神！）