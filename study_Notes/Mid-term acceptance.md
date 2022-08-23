### **Mid-term acceptance**

我的项目是（多元）图像修复，先来介绍一下我的baseline，我的baseline是Pluralistic Image Completion（PICNet），它的亮点就是在实现多元图像修复的同时，还取得了很好的修复效果。

<img src="https://user-images.githubusercontent.com/93063038/182635932-9f8feeed-a680-4afa-8eac-208f448010eb.png" title="" alt="image" data-align="center">

传统图像修复的学习方法想要生成多元的修复结果面临的一个主要挑战是每个标签通常只有一个groundtruth训练实例。因此，大多数都是实现 one input one output的图像修复。而PICNet作者在论文中提了CVAE和Instance Blind这两种方法，CVAE通过添加随机采样的方式使输出多样化，而Instance Blind通过只匹配可见部分来使输出多样化。然而，即使从CVAE 中采样仍然会导致很小的多样性，Instance Blind则很不稳定，而PICNet提出了一种具有两条平行路径的具有概率原则的框架，将CVAE和Instance Blind相结合，具体网络结构如图。

<img src="https://user-images.githubusercontent.com/93063038/182637610-8977cccc-7ac1-4bce-b428-efc0fd537ae8.png" title="" alt="image" data-align="center">

即构建重构路径和生成路径，同时对两条路径进行训练，测试时只运行生成路径。

具体来看就是训练时令重建路径的隐空间分布去近似$\mathcal N(0,\sigma^{2(i)}(n)I)$（就是一个传统的VAE的思路，但作者没有向VAE那样选用标准正态分布），然后再固定重建路径的隐空间分布，令生成路径去近似它。从直觉来看就是重建路径o n的隐空间分布中学习到了被mask掉的区域的特征，而生成路径本身并不激励网络学习mask区域的特征，而是构建了一个KL loss使训练出的隐空间向重建路径训练出的隐空间近似，在这个过程中也学习到了部分mask区域的特征，因此相较于Instance Blind方法它更加稳定（因为学习到了mask区域的特征），相较于CVAE更具有多样性（因为它并没有直接激励生成网络直接匹配整个groundtruth），具体作者在论文中也给出了严密的数学推导。

### 我的改进：

#### **1.LOSS**

<img src="file:///home/lazy/.config/marktext/images/2022-08-08-17-54-24-image.png" title="" alt="" data-align="center">

<img src="file:///home/lazy/.config/marktext/images/2022-08-08-17-54-42-image.png" title="" alt="" data-align="center">

<img src="file:///home/lazy/.config/marktext/images/2022-08-08-17-55-00-image.png" title="" alt="" data-align="center">

<img src="file:///home/lazy/.config/marktext/images/2022-08-08-22-03-41-image.png" title="" alt="" data-align="center">

通过高斯核与图像卷积来计算图像的均值和方差

MS-SSIM就是多尺度的SSIM，多尺度的实质就是对生成图像和真实图像不断地以2为因子进行下采样（一般下采样4次，即取5个尺度（分辨率）），从而得到具有多个分辨率的图像。并对这些不同分辨率的图像依次进行SSIM的评估，最后以某种方式将这些SSIM融合成一个值，就是MS-SSIM。

SSIM和MS-SSIM都对均匀的偏差非常不敏感，这导致了（使用它们作为损失函数）会带来颜色的平移或亮度的改变，处理结果会变得阴暗。然而，相比其他损失函数，MS-SSIM可以保留高频区域的对比。另一方面，L1可以保留颜色和亮度，但是它会忽略局部结构。

<img src="file:///home/lazy/.config/marktext/images/2022-08-08-22-08-00-image.png" title="" alt="" data-align="center">

#### 2.Attention

SimAM，一个无参3D注意力模块，从神经科学理论出发，构建了一种能量函数用于计算注意力权值。

<img src="file:///home/lazy/.config/marktext/images/2022-08-08-22-12-28-image.png" title="" alt="" data-align="center">

#### 3.Normalization

<img src="file:///home/lazy/.config/marktext/images/2022-08-08-22-15-45-image.png" title="" alt="" data-align="center">

首先，重开了个题目。

由于算力原因，本来打算转paddle，转了两天，结果放到aistudio上各种跑不了，然后这paddlepaddle不支持archlinux，所以我得开虚拟机才能调试，反正最后没转成用了colab。

然后，由于baseline是一个结合了VAE和GAN的模型，去学习了一下VAE。

钱学了一下概率论

因为是新接触图像修复这个方向，刚开始也是想对网络结构作一定的改进，所以我泛读了一些论文和综述。

大致搞懂了我的baseline。

重温了一下注意力机制。

尝试了wgan-div wgan-gp

粗略地学了html和css。
