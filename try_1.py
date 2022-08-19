import numpy as np
import torch

z = torch.Tensor(np.random.normal(0, 1, (8, 100)))
print(z.shape)