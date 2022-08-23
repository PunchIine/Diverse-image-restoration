Attention 

> Attention是一种提升encoder-decoder模型效果的机制，一般称作Attention mechanism。

**注意力机制的实现流程：**

1. 在所有输入信息上计算注意力分布

2. 根据注意力分布计算输入信息的加权和，以此选择关键信息

也就是说我们设计注意力模块从而得到注意力分布，然后让注意力分布通过某种方式作用于输入，产生更有效的信息。

**打分函数：**

> 注意力模块通过一个打分函数来计算输入量与查询向量之间的相关性。

打分函数又分为不需要学习的打分函数和需要学习的打分函数

需要学习的打分函数常常通过以下几种方式来计算：

1. 加代模型：$s(x_i,q)=v^Ttanh(Wx_i+Uq)$

2. 点积模型：$s(x_i,q)=x_i^Tq$

3. 缩放点积模型：$s(x_i,q)=\frac{x_i^Tq}{\sqrt{d}}$

4. 双线性模型：$s(x_i,q)=x_i^TWq$

**注意力分布（Attention distribution）：**

我们用注意力变量$z\in[1,N]$来表示被选择信息的索引位置，即$z=i$表示选择了第$i$个输入向量。这里采用Soft Attention的方式，即计算在给定$q$和$X$下，选择第$i$个输入向量的概率$α_i$。

$$
\begin{aligned}a_i&=p(z=i|X,q)\\
&=softmax(s(x_i,q))\\
&=\frac{exp(s(x_i,q))}{\sum_{j=1}^Nexp(s(x_j,q))}\end{aligned}
$$

加权平均：注意力分布$a_i$可以解释为在给定任务相关的查询$q$时，第$i$个输入向量受关注的程度。我们采用一种“软性”的信息选择机制对输入信息进行汇总。

$$
\begin{aligned}att(X,q)&=\sum^N_{i-1}a_ix_i\\
&=\mathbb{E}_{z\backsim{p(z|X,q)}}[x_z]\end{aligned}
$$

下图即为使用加权平均方法进行相关性采样的模式图。

<img src="https://user-images.githubusercontent.com/93063038/182820150-44b87cad-7850-4496-a6c5-258721b50c99.png" title="" alt="image" data-align="center">

> 除了直接加权平均的方法，还有一些更加复杂的计算方法。但是加权平均是最常用的。一般情况下，需要设计的只是打分函数，采样过程是不需要设计的。

**软注意力（soft-attention）：**

> 软注意力选择的信息是所有输入向量在注意力分布下的期望

相关性采样过程本身就是软注意力的一种形式

**强注意力（hard-attention）：**

> 硬注意力只关注某一个输入向量是否该被选择，或者说注意力分布只能是0或1

<img src="file:///home/lazy/.config/marktext/images/2022-08-04-18-18-46-image.png" title="" alt="" data-align="center">

**键值注意力（Key-Value Attention）：**

<img src="https://user-images.githubusercontent.com/93063038/182985498-c0e5991c-f322-4f3d-bff5-f76f3232ca7d.png" title="" alt="image" data-align="center">

Key与Query计算对应Value的权重（即作为打分函数的参数，经过softmax得到对应Value的注意力分布）

此时序列中的每一个元素都以（Key，Value）的形式存储，则Attention通过计算Query与Key的相似度反映了对应Value的重要程度，即权重（注意力分布），然后加权求和就得到了Attention值。

> **本质上attention机制是对source中元素的value值进行加权求和，而query和key用来计算对应value的权重系数**
> 
> 一般我们可以用键值对（key-value pair）格式来表示输入信息，其中 “键”用来计算注意力分布αn​，“值”用来计算聚合信息：

<img src="file:///home/lazy/.config/marktext/images/2022-08-05-10-06-57-image.png" title="" alt="" data-align="center">

**自注意力（Self Attention）：**

> 自注意力的核心内容是量化表示输入元素之间的相互依赖性。比如，通常情况下键盘和鼠标会同时出现，所以当输入中出现键盘时模型就可以关注相关位置是否有鼠标。这种机制允许输入与彼此“自我”简历关系并确定他们应该更多关注什么。

<img src="https://user-images.githubusercontent.com/93063038/182991671-16b69739-9e4b-42ca-8df5-09646e49cc09.png" title="" alt="image" data-align="center">

首先假设我们的input是上图中的序列$x_1\backsim{x_4}$，每一个input (vector)先乘上一个矩阵得到embedding，即向量$a_1\backsim{a_4}$​ 。接着这个embedding进入self-attention层，每一个向量分别乘上3个不同的transformation matrix $W_q$、$W_k$​和$W_v$​，以向量$a_1$​为例，就会分别得到3个不同的向量$q^1$、$k^1$和$v^1$

<img src="https://user-images.githubusercontent.com/93063038/183033525-2f48b4a3-f1a1-456d-96b6-f6508491e5e0.png" title="" alt="image" data-align="center">

如果要考虑local的特征，则只需要学习出相应的$\hat{α}_{1,i}=0$，$b^1$上就不再带有i对应分支的信息了；如果要考虑global的特征，则需要学习出相应的$\hat{a}_{1,i}\neq0$，$b^1$上就带有$i$对应分支的信息了。

以上注意力模式可以表示为公式：

$$
Attention=QK^TV
$$

令其具有自注意力机制的最简单的方法就是令$Q=K=V$，则在键值注意力中，计算相关性的过程就发生在输入本身内部（因为在键值注意力中是$Q$和各个$K$的相关性，再用$V$进行加权）。

通道注意力（）

参考资料：

1. https://ml.akasaki.space/ch3p2/[4]attention#44-%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9Bself-attention

2. [注意力机制总结 - nxf_rabbit75 - 博客园](https://www.cnblogs.com/nxf-rabbit75/p/11555683.html)

3. https://www.bilibili.com/video/BV19o4y1m7mo?spm_id_from=333.337.search-card.all.click
