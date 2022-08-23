论文地址：[PD-GAN: Probabilistic Diverse GAN for Image Inpainting](https://arxiv.org/abs/2105.02201) 

## **简介：**

目前常见的图像修复方法都是采用一些Encoder-Decoder的方式去填充图像缺失的部分，这种方法在训练的时候都会采用Reconstruction Loss，这可能会限制图像修复的多样化生成内容，所以现存的图像修复方法很难兼顾多样性。

作者注意到GAN强大的生成能力可以根据不同的random noise的输入生成不同的图像内容。因此作者提出了PD-GAN，与常见的图像修复方法不同，PD-GAN不直接将输入图像送入CNN而是从随机噪声向量开始，然后对其进行Decode以生成内容，这样做的好处是能让生成的多样性提高，同时作者也提出新的Normalization和Loss，达到兼顾多样性和生成质量的效果。

## **主要贡献：**

1. 提出了SPDNorm来调整噪声向量的特征以达到上下文的约束。

2. 提出了Perceptual Diverse Loss来增强多样性。

## 网络结构：

<img src="https://user-images.githubusercontent.com/93063038/184526471-82f121bb-bc57-436a-9001-e53a2d2649b3.png" title="" alt="image" data-align="center">

## **SPDNorm：**

<img src="https://user-images.githubusercontent.com/93063038/184319607-1db56996-3b33-4132-9722-b8e79541e423.png" title="" alt="image" data-align="center">

### 1.Hard SPDNorm（$D^h$）

<img src="https://user-images.githubusercontent.com/93063038/184319799-eac13dcf-d79a-4261-8ebb-2fda2d66315f.png" title="" alt="image" data-align="center">

如图：Hard probabilistic diverse map（$D^h$）由修复掩码M决定，不需要训练的过程。 

从直觉上来讲，越接近掩码的边界，上下文对其的约束越强，因此需要更多的先验信息来指导其填充过程。而对于远离掩码边界的区域，相对上下文对其约束就较弱，因此需要的先验信息就较少。对于修复掩码M，我们进行n次迭代扩张操作，将掩码更新的过程记作$F_m$，而$M_i=F_m(M_{i-1})$且$M_0=M$。（作者在这用了dilation operation这个词，翻译过来是扩张操作的意思，emm这个词使我在理解上产生了一些谬误，在我的理解中扩张应该是一种有内向外，或者类似padding之类的操作，但(如果我没理解错的话)这里则是一个逐步迭代向mask区域中心进行的操作，暂时还不是很能接受）具体如下：

$$
M_i(x,y)=\begin{cases}1 & if\sum_{(a,b)\in\mathcal N(x,y)}M_{i-1}(a,b)\\
0 & otherwise\end{cases}
$$

最终，我们填充$D^h$的第$i$个区域$E_i=M_i-M_{i-1}$，值为$\frac{1}{k^i}$，在论文中作者根据经验设置$k=4$。

Hard SPDNorm output：

$$
F^{hard}_{x,y,c}=D^h_{x,y}(\gamma_{x,y,c}(P)\frac{F_{x,y,z}^{in}-\mu_c}{\sqrt{\sigma^{2}_c+\epsilon}}+\beta_{x,y,c}(P))
$$

### 2.Soft SPDNorm（$D^s$）

<img src="https://user-images.githubusercontent.com/93063038/184327684-51175301-3281-4f3c-845b-4fe608c9805c.png" title="" alt="image" data-align="center">

Hard SPDNorm层固定地控制$D^h$生成不同结果的概率，然而，仅仅使用Hard SPDNorm将无法很好地结合先验信息，影响生成结果的质量。因此，作者认为这个概率还应该由先验信息与修复掩码来动态控制，提出了Soft SPDNorm，通过自适应的方式来学习概率分布。

> The soft SPDNorm extracts the feature from both $P$ and $F^{in}$ to predict a soft probabilistic diversity map, guiding the diverse inpainting process.

我们先将先验信息$P$经过一个卷积层提取它的特征$F^{p}$，然后将其与$F^{in}$作一个concat，经过sigmoid将其归一化后与修复掩码结合得到$D^s$：

$$
D^s=\sigma(Conv([F^p,F^{in}])·(1-M)+M)
$$

作者发现foreground区域的$D^s$值的变化非常平稳且接近0.5，因此仅靠$D^s$无法衡量预测多样化结果的概率。

Soft SPDNorm output：

$$
F_{x,y,z}^{soft}=D^s_{x,y}(\gamma_{x,y,z}(P)\frac{F_{x,y,z}^{in}-\mu_{c}}{\sqrt{\sigma^2+\epsilon}}+\beta_{x,y,z}(P))
$$

### 3.SPDNorm ResBlock

综上，我们知道Hard SPDNorm能够很好的控制修复图像的多样性，但无法保证生成结果的质量，而Soft SPDNorm结合了先验信息能够生成质量较好的图像，但无法保证生成图像的多样性。因此我们将二者结合，使之互补，提出了SPDNorm ResBlock，对于不同尺度的Block，我们对先验信息和修复掩码进行下采样以使其匹配。

## Perceptual Diversity Loss

[Mode Seeking Generative Adversarial Networks for Diverse Image Synthesis](https://arxiv.org/abs/1903.05628)这篇论文中提出了一个多样性损失：

$$
L_{div}=\frac{\Vert z_1-z_2 \Vert_1}{\Vert I_{out1}-I_{out2} \Vert_1+\varepsilon}
$$

此损失旨在度量两个输出图像在像素空间上的距离，使输出图像在像素空间中的距离变远，而当它们对应的隐变量彼此距离很远，则此Loss控制输出图像在像素空间中的距离更远。然而，作者发现它并不适合用在图像修复任务上。首先，最小化$L_{div}$改变了上下文区域的内容，而对于不同的潜在向量这应该是恒定的。其次，作者发现使用$L_{div}$会使网络的训练十分地不稳定。由于直接在像素空间上激励输出图像的距离变远，最小化$L_{div}$甚至会使输出结果趋于全白或全黑。因此，作者提出了一个感知多样性损失（Perceptual Diversity Loss）：

$$
L_{pdiv}=\frac{1}{\sum_i \Vert F_i(I_{out1})·M-F_i(I_{out2})·M \Vert_1+\varepsilon}
$$

$F_i$在原文中指经过预训练模型VGG-19网络中的第$i$层提取出的feature map（在作者的工作中对应于层ReLU1_1、ReLU2_2 ... ReLU5_5的经过激活函数后的activation maps)。$L_{pdiv}$通过引入掩码来保持上下文不变，同时，$L_{pdiv}$是在感知空间上进行计算而非像素空间。在高度非线性的网络特征空间中最大化二者距离，集成了语义的度量，并避免了生成黑白像素的情况。注意在$L_{pdiv}$中作者并没有加入隐变量$z$的度量，因为在多元图像修复的任务中，无论两个隐变量多么接近，我们都希望生成图像的感知距离能最大化。据作者说，这同时也使训练更加地稳定了。

> 除了Perceptual Diversity Loss外，PD-GAN还遵循了SPAD（[Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291)）利用了reconstruction loss（Perceptual  losses for real-time style transfer and super-resolution），feature  matching loss（High-resolution image synthesis and semantic manipulation with conditional gans）和hinge adversarial loss来优化网络。





关于修复图像多样性的问题：

encoder-decoder的方法由于是将蒙版图像进行encode后得到的隐向量，因此，若在训练是是由整张图像激励网络优化，那么当测试时，同一张蒙版图像得到的隐向量是确定的，即生成的修复图像也是单一确定的。而PICNet的方法则是仅由蒙版图像部分激励网络优化，且通过平行路径学习mask部分的特征，因此mask区域对应的隐向量是随机的，由于生成路径的隐空间学习了重建路径得到的mask部分特征，因此能较稳定的生成具有多样性的修复图像。
