# 实验记录

## Basebase line

<img src="https://user-images.githubusercontent.com/93063038/183044785-86b8f96a-2bac-48d2-8dcf-8b112fc9a01b.png" title="" alt="image" data-align="center">

<img src="https://user-images.githubusercontent.com/93063038/183104909-c8715304-d401-440b-99a0-f19a42782879.png" title="" alt="image" data-align="center">

<img title="" src="https://user-images.githubusercontent.com/93063038/183049683-fefde172-3bb8-44b9-9fc9-c622fafd6859.png" alt="image" width="782" data-align="center">

<img title="" src="https://user-images.githubusercontent.com/93063038/183049902-fbe464dd-563b-470e-80fd-22ebc54c2cde.png" alt="image" data-align="center" width="663">

## LOSS

<img src="https://user-images.githubusercontent.com/93063038/182867845-750cf25d-9246-4430-bec5-7f907fee8c88.png" title="" alt="image" data-align="center">

<img src="https://user-images.githubusercontent.com/93063038/182975922-077e502a-1530-46d4-ac53-8d66d1e82d63.png" title="" alt="image" data-align="center">

<img src="https://user-images.githubusercontent.com/93063038/183291853-57379dae-6f63-41ff-b32d-43a800ae6308.png" title="" alt="image" data-align="center">

<img src="https://user-images.githubusercontent.com/93063038/183291886-6555bbd2-f24a-45df-9a1e-368982eeea57.png" title="" alt="image" data-align="center">

## Attention

<img src="https://user-images.githubusercontent.com/93063038/183226338-1f8ad89e-f29e-4a8c-bb9e-78b2c3e5cecb.png" title="" alt="image" data-align="center">

<img src="https://user-images.githubusercontent.com/93063038/183239024-ce0decbd-f5b0-47e3-aaf8-c89fdbf8dfe3.png" title="" alt="image" data-align="center">

<img src="https://user-images.githubusercontent.com/93063038/183228495-0ffe60e7-78be-482b-8f12-f7be7877a868.png" title="" alt="image" data-align="center">

![image](https://user-images.githubusercontent.com/93063038/183228534-7caa249b-bdae-4cfb-9ea7-91cc04c12891.png)

## Group Normalization

<img src="https://user-images.githubusercontent.com/93063038/183290998-5e440e1e-c82b-4569-8c10-ad42ee8873d2.png" title="" alt="image" data-align="center">

<img src="https://user-images.githubusercontent.com/93063038/183331491-3bebc41a-0da5-468b-8e38-75f51eeee681.png" title="" alt="image" data-align="center">

<img src="https://user-images.githubusercontent.com/93063038/183291697-e38e48ad-e549-4000-a7e6-a140701a12c3.png" title="" alt="image" data-align="center">

<img src="https://user-images.githubusercontent.com/93063038/183291747-7bab93ec-0414-410f-a89f-4c5283eaa77c.png" title="" alt="image" data-align="center">

|                     | L1_loss↓   | PSNR↑       | TV↓        | FID↓        |
| ------------------- | ---------- | ----------- | ---------- | ----------- |
| Base                | 8.3597     | 23.3455     | 8.7266     | 13.6334     |
| MS-SSIM-L1 loss     | 8.2796     | 23.2113     | **8.3000** | 12.9456     |
| SimAM Attention     | **7.9695** | **23.5854** | 8.4483     | **12.4844** |
| Group Normalization | 8.1413     | 23.4755     | 8.4518     | 13.6261     |

![](/home/lazy/.config/marktext/images/2022-08-12-22-36-35-image.png)

增加循环一致性损失

SPDNorm

![image](https://user-images.githubusercontent.com/93063038/187014085-1bdb98a0-62ac-413d-be18-b753e10924f0.png)
