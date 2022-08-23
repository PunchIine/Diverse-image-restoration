## Cycle GAN

任务：完成图像到图像之间的转换   （image-to-image translation）

此前的resolution：使用一组对齐的图像来学习输入图像到输出图像之间的映射

遇到的问题：获得配对训练数据可能既困难又昂贵

Cycle GAN：提出了一种在没有配对示例的情况下学习将图像从源域 X 转换到目标域 Y 的方法

<img src="https://user-images.githubusercontent.com/93063038/179342276-ed78f5a0-f85d-439f-8508-a3546d1bd115.png" title="" alt="image" data-align="center">

### Our objective contains two types of terms：

> adversarial losses [16] for matching the distribution of generated images to the data distribution in the target domain; and cycle consistency losses to prevent the learned mappings G and F from contradicting each other

#### Adversarial Loss

i.e.

<img src="https://user-images.githubusercontent.com/93063038/179343799-7e1698a2-b65c-441d-9d20-a6937bc14637.png" title="" alt="image" data-align="center">

在学习Origin GAN时已经有了解过。

#### Cycle Consistency Loss

> with large enough capacity, a network can map the same set of input images to any random permutation of images in the target domain, where any of the learned mappings can induce an output distribution that matches the target distribution. 

由于在容量足够的情况下，网络可以将输入图像集合中的相同图像映射在目标域的任何随即排列上（即不能构成单射），因此，我们仅仅使用Adversarial Loss不能保证我们的网络能够将单一输入xi映射到我们想要得到的输出yi，为了进一步减小可能的映射函数的空间，作者认为映射函数应该满足循环一致性，因此引入了Cycle Consistency Loss

i.e.

<img src="https://user-images.githubusercontent.com/93063038/179344536-f00487f6-7e6c-4d2f-bc2e-a9fa382488b4.png" title="" alt="image" data-align="center">

#### Full Objective

> λ controls the relative importance of the two objectives

i.e.

<img src="https://user-images.githubusercontent.com/93063038/179346102-7da684f7-3660-498a-95ec-f0cfcd0cc849.png" title="" alt="image" data-align="center">

可以将我们要解决的问题看作

<img src="https://user-images.githubusercontent.com/93063038/179382516-73f51297-a6e1-4211-bc3d-06742b20889b.png" title="" alt="image" data-align="center">
