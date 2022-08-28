## 简介：

提出了注意力归一化，使用注意力归一化（Attentive Normalization）来描述长期依赖，是对传统Instance Normalization的扩展。与 self-attention GAN 相比，注意力归一化不需要测量所有位置的相关性，因此可以直接应用于大尺寸特征图而没有太多计算负担。

## 贡献：

1. 提出的注意力归一化在图像生成的过程中能很好的捕捉中间特征图中的视觉远近关系，Attentive Normalization从feature map中预测语义布局后基于此布局进行区域Instance Normalization。

2. 通过在具有相似语义的区域中同时融合、传播特征语义信息，具有更小的计算复杂度（相较self-attention GAN等） ——（self-attention模块需要计算特征图中每两个点之间的相关性，因此计算成本将随着特征图的增大急剧增长）

3. 进行了广泛的实验以证明 AN 在远距离关系建模中对类条件图像生成和生成图像修复的有效性。

## 读前小问号：

1. 如何通过featrue map实现对语义布局的预测(语义布局学习模块具体实现)（semantic layout prediction, and self-sampling regularization）
2. 作者提出传统的Instance Normalization忽略了不同的位置可能以不同的均值、方差对应语义特征（这种机制往往会在空间上恶化中间特征的学习语义），那么AN又是如何在空间上对featrue map进行归一化的呢

## Attentive Normalization

### 1.Sementic Layout Learning Module

假设每个图像由　n　个语义实体组成，特征图中的每一个特征点至少由一个语义实体确认。

作者给出 n 个初始的期望语义实体，并将它们与图像特征点的相关性定义为它们的内积。

表示这些实体的语义是通过反向传播来学习的。

作者根据这些实体的激活状态将输入特征图中的特征点聚合到不同的区域。

此外为了鼓励这些实体接近不同的模式，对这些实体采用了正交正则化，即：

$$
\mathcal L_o = \lambda_o \Vert WW^T-I \Vert ^ 2 _ {F}
$$

在作者的实现中，采用具有 n 个过滤器的卷积层作为语义实体。该层将输入特征映射 X 转换为新的特征空间$ f (X) ∈ Rh×w×n$。直观地说，n 越大，可以学习到的高级特征越多样化和丰富。

### 2.Self-sampling Regularization
