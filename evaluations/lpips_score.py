import lpips
import numpy as np
import argparse
from PIL import Image
import torch
# from dataloader.image_folder import make_dataset

parser = argparse.ArgumentParser(description='Evaluation ont the dataset')
parser.add_argument('--out0_path', type = str, default='/content/images/out0',
                    help = 'path to original particular solutions')
parser.add_argument('--out1_path', type = str, default='/content/images/out1',
                    help='path to save the test dataset')
parser.add_argument('--num_test', type=int, default=2000,
                    help='how many images to load for each test')
args = parser.parse_args()
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(path_files):
    if path_files.find('.txt') != -1:
        paths, size = make_dataset_txt(path_files)
    else:
        paths, size = make_dataset_dir(path_files)

    return paths, size


def make_dataset_txt(files):
    """
    :param path_files: the path of txt file that store the image paths
    :return: image paths and sizes
    """
    img_paths = []

    with open(files) as f:
        paths = f.readlines()

    for path in paths:
        path = path.strip()
        img_paths.append(path)

    return img_paths, len(img_paths)


def make_dataset_dir(dir):
    """
    :param dir: directory paths that store the image
    :return: image paths and sizes
    """
    img_paths = []

    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in os.walk(dir):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                img_paths.append(path)

    return img_paths, len(img_paths)

loss_lpips = lpips.LPIPS(net='alex', spatial=True)

out0_paths, out0_size = make_dataset(args.out0_path)
out1_paths, out1_size = make_dataset(args.out1_path)

dist_ = []

for i in range(args.num_test):
    out0_tensor = lpips.im2tensor(lpips.load_image(out0_paths[i]))
    out1_tensor = lpips.im2tensor(lpips.load_image(out1_paths[i]))
    dist = loss_lpips.forward(out0_tensor, out1_tensor)
    dist_.append(dist.mean().item())
    print("lpips:", end='')
    print(dist.mean().item())

print('')
print('Avarage LPIPS: %.3f' % (sum(dist_)/len(out0_paths)))