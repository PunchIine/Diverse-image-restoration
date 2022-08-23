论文地址：[UCTGAN: Diverse Image Inpainting based on Unsupervised Cross-Space Translation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhao_UCTGAN_Diverse_Image_Inpainting_Based_on_Unsupervised_Cross-Space_Translation_CVPR_2020_paper.pdf)

## 简介：

为了在图像修复任务上产生多种合理的解决方案，作者提出了**无监督跨空间转换生成对抗网络（UCTGAN）**。此网络主要由三个网络模块组成：conditional encoder module，mainfold projection module 和 generation module。 

**流形投影模块和生成模块相结合**，以无监督的方式，通过将实例图像空间和条件补全空间投影到公共的低维流形空间中，学习两个空间之间一对一的图像映射。此方法可以大大提高修复样本的多样性。

为了更好地理解全局信息，作者还**提出了一个新的cross semantic attention layer**，利用已知部分和已补全部分的长期依赖关系，提高了修复样本的真实性和外观一致性。

> 如今基于GAN的方法已经能很好地完成图像生成任务，但它们却不能直接应用在图像修复任务上，有两个原因：1. 在多元图像修复的场景下，条件标签是蒙版图像本身，并且在训练集中每个条件标签只存在一个实例（即蒙版图像就是grondtruth）。也就是说，没有明确表达条件分布的条件训练数据集。2. 多元图像修复场景相较于图像生成有很强的约束性（修复后的图像应与蒙版图像保持颜色和纹理的完整性和一致性）。因此相较于经典的图像生成，在此任务上更容易发生mode collapse。

## 主要贡献：

## **概率分析：**

> 所有修复图像$I_c$的集合被称为给定蒙版图像$I_m$为条件的修复图像空间$S_{cc}$
> 
> 用于指导训练的实例图像$I_i$来自训练数据集，所有实例图像$I_i$的集合称为实例图像空间$S_i$
> 
> 定义一个公共的低维流形空间为$S_m$

UCTGAN的网络框架将最大化训练实例的条件对数似然，其中涉及变分下限，如下：

$$
\begin{aligned}\log{p(I_c|I_m)} \ge -KL(f_{\varphi}(Z_c|I_i,I_m)||f_{\psi}(Z_c|I_m))+ \\ \mathbb E_{Z_c\backsim f_{\varphi}(Z_c|I_i,I_m)}[\log{g_{\theta}(I_c|Z_c,I_m)}]\end{aligned} \tag{1}
$$

其中$f_{\varphi}$，$f_{\psi}$，$g_{\theta}$分别是后验采样函数、条件先验和似然，而$\varphi 、\psi 、\theta$分别是它们对应函数的深度网络的参数。条件先验在这里被设置为$f_{\psi}(Z_c|I_m)=\mathcal N(0,I)$。第一项KL散度主要是将实例图像$I_i$投影到低维流形向量$Z_c$中，$Z_c$由实例图像$I_i$对应的修复图像$I_c$共享。

## 网络结构：

![image](https://user-images.githubusercontent.com/93063038/184526906-e664a176-39ce-4fe6-9a3c-5162890e3576.png)

该网络以端到端的方式进行训练，它由两个分支组成，主要分为三个网络模块：流形投影模块$E_1$，条件编码模块$E_2$，和生成模块$G$。主分支由$E_1$和$G$组成，它负责将实例图像空间$S_i$和条件修复图像空间$S_{cc}$投影到一个公共的隐式流形空间$S_m$中，以无监督的方式学习两个空间之间的一对一图像映射。另一个分支由$E_2$组成，主要是产生一个条件约束起到类似条件标签（conditional label）的作用。

对于蒙版图像$I_m$只有一个groundtruth $I_g$可用作训练数据以最大化等式（1）中的似然，*也就是说，实例图像和修复图像之间的映射只能以无监督的方式获得*，这往往会导致mode collapse。为了通过一对一映射关联两个空间（实例图像空间和条件完成图像空间），实例图像$I_i$及其对应的映射恢复图像$I_c$在低维流形空间$S_m$中应该具有相同的表示。(???)
