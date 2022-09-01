# 实验记录

## Basebase line

![image](https://user-images.githubusercontent.com/93063038/183044785-86b8f96a-2bac-48d2-8dcf-8b112fc9a01b.png)

![image](https://user-images.githubusercontent.com/93063038/183104909-c8715304-d401-440b-99a0-f19a42782879.png)

![image](https://user-images.githubusercontent.com/93063038/183049683-fefde172-3bb8-44b9-9fc9-c622fafd6859.png)

![image](https://user-images.githubusercontent.com/93063038/183049902-fbe464dd-563b-470e-80fd-22ebc54c2cde.png)

<img src="https://user-images.githubusercontent.com/93063038/187071953-9cf3d142-a6dc-4cd5-b83b-4b13f425e60b.png" title="" alt="image" data-align="center">

## LOSS

![image](https://user-images.githubusercontent.com/93063038/182867845-750cf25d-9246-4430-bec5-7f907fee8c88.png)

![image](https://user-images.githubusercontent.com/93063038/182975922-077e502a-1530-46d4-ac53-8d66d1e82d63.png)

![image](https://user-images.githubusercontent.com/93063038/183291853-57379dae-6f63-41ff-b32d-43a800ae6308.png)

![image](https://user-images.githubusercontent.com/93063038/183291886-6555bbd2-f24a-45df-9a1e-368982eeea57.png)

## 

<img src="https://user-images.githubusercontent.com/93063038/187072473-496058a2-dbcf-48ea-bb89-c27891c4f46e.png" title="" alt="image" data-align="center">

## Attention

![image](https://user-images.githubusercontent.com/93063038/183226338-1f8ad89e-f29e-4a8c-bb9e-78b2c3e5cecb.png)

![image](https://user-images.githubusercontent.com/93063038/183239024-ce0decbd-f5b0-47e3-aaf8-c89fdbf8dfe3.png)

![image](https://user-images.githubusercontent.com/93063038/183228495-0ffe60e7-78be-482b-8f12-f7be7877a868.png)

![image](https://user-images.githubusercontent.com/93063038/183228534-7caa249b-bdae-4cfb-9ea7-91cc04c12891.png)

## <img src="https://user-images.githubusercontent.com/93063038/187073902-5dc4ce4d-ebee-474b-93ca-0bff3f78c118.png" title="" alt="image" data-align="center">

## Group Normalization

![image](https://user-images.githubusercontent.com/93063038/183290998-5e440e1e-c82b-4569-8c10-ad42ee8873d2.png)

![image](https://user-images.githubusercontent.com/93063038/183331491-3bebc41a-0da5-468b-8e38-75f51eeee681.png)

![image](https://user-images.githubusercontent.com/93063038/183291697-e38e48ad-e549-4000-a7e6-a140701a12c3.png)

![image](https://user-images.githubusercontent.com/93063038/183291747-7bab93ec-0414-410f-a89f-4c5283eaa77c.png)

![image](https://user-images.githubusercontent.com/93063038/187074940-061a54d3-133e-46fa-9394-dceab5f9e2f4.png)

|                     | L1_loss↓   | PSNR↑       | TV↓        | FID↓        |
| ------------------- | ---------- | ----------- | ---------- | ----------- |
| Base                | 8.3597     | 23.3455     | 8.7266     | 13.6334     |
| MS-SSIM-L1 loss     | 8.2796     | 23.2113     | **8.3000** | 12.9456     |
| SimAM Attention     | **7.9695** | **23.5854** | 8.4483     | **12.4844** |
| Group Normalization | 8.1413     | 23.4755     | 8.4518     | 13.6261     |

SPDNorm

使用PICNet的预训练模型，生成质量较低的修复图像传入SPDNorm，由PD-GAN来生成具有更丰富的多样性、更高质量的图像。简单复现了PDGAN的SPDNorm，使用了PICNet中的block，结合self-Attention GAN，加入了TV loss，使用了LSGAN的adversarial loss，自己设计（瞎搞了）了一个loss

结论是：自己瞎搞的loss以及单一的self Attention导致了生成图像质量较低（直觉上），且多样性反而变差了！！！！！！！！！！！！！！！（大悲😭）

出乎意料的是其余的参数都有所优化（？）

<img title="" src="https://user-images.githubusercontent.com/93063038/186628066-6173b172-7a01-4d4e-bec3-994a23f01062.png" alt="image" data-align="center">

<img src="https://user-images.githubusercontent.com/93063038/186628757-a386d65f-5256-47ac-979e-ab2b35892e33.png" title="" alt="image" data-align="center">

<img src="https://user-images.githubusercontent.com/93063038/186679315-e6065ad5-4dbe-4530-b3d3-ed790037ef8d.png" title="" alt="image" data-align="center">

<img src="https://user-images.githubusercontent.com/93063038/186685136-ba1401c8-9303-49f5-9df6-a540c88aa482.png" title="" alt="image" data-align="center">

<<<<<<< HEAD
=======
![image](https://user-images.githubusercontent.com/93063038/186685136-ba1401c8-9303-49f5-9df6-a540c88aa482.png)

>>>>>>> fe4bce53c7eea27dd21702fe1a642d3218063c8a
5 结合Attentive Normalization设计一个Attention Layer

6 gated Conv 结合coord Conv

7 minibatch discriminator || multi-stage discriminator || PGGAN

8 hinge loss 变种

9 exponential moving average
