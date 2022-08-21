import numpy as np
import torch
import torch.nn.functional as F

# With square kernels and equal stride
filters = torch.randn(8, 4, 3, 3)
inputs = torch.randn(1, 4, 5, 5)
a = F.conv2d(inputs, filters, padding=1)

print(torch.split(filters, 4, dim=0)[1].shape)