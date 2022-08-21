import numpy as np
import torch
import torch.nn.functional as F

a = torch.randn(1, 2, 3, 3)

torch.cat([a, a],dim=0)

print(a.shape)