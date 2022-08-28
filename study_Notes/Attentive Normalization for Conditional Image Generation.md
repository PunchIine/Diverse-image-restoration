## 简介：

提出了注意力归一化，使用注意力归一化（Attentive Normalization）来描述长期依赖，是对传统Instance Normalization的扩展。与 self-attention GAN 相比，注意力归一化不需要测量所有位置的相关性，因此可以直接应用于大尺寸特征图而没有太多计算负担。

## 贡献：

1. 提出的注意力归一化在图像生成的过程中能很好的捕捉中间特征图中的视觉远近关系，Attentive Normalization从feature map中预测语义布局后基于此布局进行区域Instance Normalization。

2. 通过在具有相似语义的区域中同时融合、传播特征语义信息，具有更小的计算复杂度（相较self-attention GAN等） ——（self-attention模块需要计算特征图中每两个点之间的相关性，因此计算成本将随着特征图的增大急剧增长）

3. 进行了广泛的实验以证明 AN 在远距离关系建模中对类条件图像生成和生成图像修复的有效性。

## 读前小问号：

1. 如何通过featrue map实现对语义布局的预测
2. 作者提出传统的Instance Normalization忽略了不同的位置可能以不同的均值、方差对应语义特征（这种机制往往会在空间上恶化中间特征的学习语义），那么AN又是如何在空间上对featrue map进行归一化的呢
3. 
