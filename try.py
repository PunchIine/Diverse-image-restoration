import torch
import torch.nn.functional as F

ker = torch.ones(1, 1, 5, 5) / 25

a = torch.ones(1, 1, 5, 5)

b = F.conv2d(a, ker, stride=1)

print(b)