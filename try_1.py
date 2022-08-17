import torch
import torch.nn as nn

a = torch.randn(8, 3, 32, 32)

A = a.detach()

conv = nn.Conv2d(3, 3, kernel_size=1)

b = conv(a)

print(a == A)