[论文：Pluralistic Image Completion](https://arxiv.org/pdf/1903.04227.pdf)

Background：现存图像补全方法大多都仅对被遮挡的图像产生一个输出，然而事实上当我们人类遇到一张不完整的图片时往往会认为这张图像的补全有多种合理的可能性。因此，作者在本文中，提出了一种多元图像补全的方法，旨在为图像补全生成多种不同的合理解决方案。

Major challenge：每个标签通常只有一个grondtruth供其训练，这会导致生成的图像缺乏多样性。

Solution：提出了一个具有两条平行路径的具有概率原则的框架。

<img title="" src="https://user-images.githubusercontent.com/93063038/182635932-9f8feeed-a680-4afa-8eac-208f448010eb.png" alt="image" data-align="center">

CVAE通过添加随机采样的方式使输出多样化。

Instance Blind通过只匹配可见部分来使输出多样化。

PICNet将二者结合，通过双路径训练，在测试时使用生成路径，但在训练时由平行的重构路径指导使输出多样化。

<img src="https://user-images.githubusercontent.com/93063038/182637610-8977cccc-7ac1-4bce-b428-efc0fd537ae8.png" title="" alt="image" data-align="center">

如图，两条路径同时进行训练，设计损失函数$KL(\mathcal N_{rec}||\mathcal  N_{g})$，固定$\mathcal N_{rec}$，使生成路径隐空间的分布向重建路径趋近，从而使得该隐空间中同时也学习到了$I_c$部分的特征，使得测试时能够稳定的输出多元化的图像补全结果。

Short+Long Term Attention

由Self-Attention GAN的基础上，作者提出不仅可以在解码器层中利用注意力图来利用远距离空间的上下文信息，还可以进一步在编码器与解码器间捕获特征之间的上下文信息。

作者认为这样做将允许网络根据情况选择关注编码器中更细粒度的特征或解码器中更语义化的生成特征。

<img src="https://user-images.githubusercontent.com/93063038/182761134-47e19db9-5846-4b97-bc8d-db4a313f1923.png" title="" alt="image" data-align="center">
