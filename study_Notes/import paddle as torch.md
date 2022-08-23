# import paddle as torch     (😄)

由于算力不足，打算白嫖百度的算力，因此打算把pytorch的项目转成paddlepaddle，故作此笔记记录。

torch.nn.Module —— paddle.nn.Layer

torch.mv —— paddle.matmul (矩阵乘法)

torch.t —— paddle.t (矩阵转置)

torch.autograd.grad —— paddle.gtad (求导数，完全一样啦)

torch.randn —— paddle.randn (区别在于paddle的size得放在列表中，如[(3,2,1)])

torch.tranpose(input, dim0, dim1) —— paddle.transpose(x, perm, name=None) 

torch中的dim0,dim1表示要交换的两个维度，而paddle中perm参数接收list或tuple表示重新排列后的原维度顺序

torch.nn.ConvTranspose2d —— paddle.nn.Conv2DTranspose (bias参数pytorch默认为true而paddle中bias_attr默认为false)
