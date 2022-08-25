import lpips
import numpy as np
import argparse
from PIL import Image
import torch
from dataloader.image_folder import make_dataset

parser = argparse.ArgumentParser(description='Evaluation ont the dataset')
parser.add_argument('--out0_path', type = str, default='/home/lazy/Pluralistic-Inpainting/images/out0',
                    help = 'path to original particular solutions')
parser.add_argument('--out1_path', type = str, default='/home/lazy/Pluralistic-Inpainting/images/out1',
                    help='path to save the test dataset')
parser.add_argument('--num_test', type=int, default=1,
                    help='how many images to load for each test')
args = parser.parse_args()


loss_lpips = lpips.LPIPS(net='vgg')

out0_paths, out0_size = make_dataset(args.out0_path)
out1_paths, out1_size = make_dataset(args.out1_path)

lpips = torch.zeros(args.num_test)

for i in range(args.num_test):
    out0_image = Image.open(out0_paths[i]).resize([256, 256]).convert('RGB')
    out0_tensor = torch.Tensor(np.array(out0_image).astype(np.float32))
    out1_image = Image.open(out1_paths[i]).resize([256, 256]).convert('RGB')
    out1_tensor = torch.Tensor(np.array(out1_image).astype(np.float32))
    lpips[i] = loss_lpips(out0_tensor.view(3, 256, 256), out1_tensor.view(3, 256, 256))
    print("lpips:", end='')
    print(torch.mean(lpips))

print('')
print("mean lpips:", end='')
print(torch.mean(lpips))