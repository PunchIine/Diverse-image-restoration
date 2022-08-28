import numpy as np
import argparse
from PIL import Image
from math import exp
import torch
import torch.nn.functional as F
from dataloader.image_folder import make_dataset
import os
import shutil

parser = argparse.ArgumentParser(description='Evaluation ont the dataset')
parser.add_argument('--gt_path', type = str, default='/home/lazy/my-Pluralistic-Inpainting/images/truth',
                    help = 'path to original particular solutions')
parser.add_argument('--save_path', type = str, default='/home/lazy/my-Pluralistic-Inpainting/images/out',
                    help='path to save the test dataset')
parser.add_argument('--num_test', type=int, default=500,
                    help='how many images to load for each test')
# parser.add_argument('--sample_numbers',type=int,default=50, help='how many smaple images for testing')
args = parser.parse_args()


def compute_errors(gt, pre):

    # l1 loss
    l1 = torch.mean(torch.abs(gt-pre))

    # PSNR
    mse = torch.mean((gt - pre) ** 2)
    if mse == 0:
        PSNR = 100
    else:
        PSNR = 20 * torch.log10(255.0 / torch.sqrt(mse))

    # TV
    gx = pre - torch.roll(pre, -1, dims=1)
    gy = pre - torch.roll(pre, -1, dims=0)
    grad_norm2 = gx ** 2 + gy ** 2
    TV = torch.mean(torch.sqrt(grad_norm2))

    #SSIM
    SSIM = ssim(gt, pre)

    return l1, PSNR, TV, SSIM

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True):
    (_, _, channel) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    img1 = img1.view(1, 3, 256, 256)
    img2 = img2.view(1, 3, 256, 256)

    return _ssim(img1, img2, window, window_size, channel, size_average)


if __name__ == "__main__":

    gt_paths, gt_size = make_dataset(args.gt_path)
    pre_paths, pre_size = make_dataset(args.save_path)

    for i in range(1000):
        print(i)
        name = gt_paths[i].split("/")[-1]
        path = os.path.join(args.save_path,name)
        try:
            image = Image.open(path)
        except:
            print (path)

    assert gt_size == pre_size
    #
    iters = int(2000/args.num_test)

    l1_loss = torch.zeros(iters)
    PSNR = torch.zeros(iters)
    TV = torch.zeros(iters)
    SSIM = torch.zeros(iters)

    for i in range(0, iters):
        l1_iter = torch.zeros(args.num_test)
        PSNR_iter = torch.zeros(args.num_test)
        TV_iter = torch.zeros(args.num_test)
        SSIM_iter = torch.zeros(args.num_test)

        num = i*args.num_test

        for j in range(args.num_test):
            index = num+j
            gt_image = Image.open(gt_paths[index]).resize([256,256]).convert('RGB')
            gt_numpy = torch.Tensor(np.array(gt_image).astype(np.float32)).cuda()
            l1_sample = 1000
            PSNR_sample = 0
            TV_sample = 1000
            SSIM_sample = 0
            name = gt_paths[index].split('/')[-1].split(".")[0]+"*"
            #            pre_paths = sorted(glob.glob(os.path.join(args.save_path, name)))
            num_image_files = len(pre_paths)

            for k in range(num_image_files-1):
                index2 = k
                try:
                    pre_image = Image.open(pre_paths[index2]).resize([256,256]).convert('RGB')
                    pre_numpy = torch.Tensor(np.array(pre_image).astype(np.float32)).cuda()
                    l1_temp, PSNR_temp, TV_temp, SSIM_temp = compute_errors(gt_numpy, pre_numpy)
                    # select the best results for the errors estimation
                    if l1_temp - PSNR_temp + TV_temp + SSIM_temp < l1_sample - PSNR_sample + TV_sample + SSIM_sample:
                        l1_sample, PSNR_sample, TV_sample, SSIM_sample = l1_temp, PSNR_temp, TV_temp, SSIM_temp
                        best_index = index2
                except:
                    print(pre_paths[index2])
            shutil.copy(pre_paths[best_index], '/home/lazy/Pluralistic-Inpainting/images/copy')
            print(pre_paths[best_index])
            print(l1_sample, PSNR_sample, TV_sample, SSIM_sample)

            l1_iter[j], PSNR_iter[j], TV_iter[j], SSIM_iter[j] = l1_sample, PSNR_sample, TV_sample, SSIM_sample

        l1_loss[i] = torch.mean(l1_iter)
        PSNR[i] = torch.mean(PSNR_iter)
        TV[i] = torch.mean(TV_iter)
        SSIM[i] = torch.mean(SSIM_iter)

        print(i)
        print('{:10.4f},{:10.4f},{:10.4f}'.format(l1_loss[i], PSNR[i], TV[i]))

    print('{:>10},{:>10},{:>10}'.format('L1_LOSS', 'PSNR', 'TV'))
    print('{:10.4f},{:10.4f},{:10.4f}'.format(l1_loss.mean(), PSNR.mean(), TV.mean(), SSIM.mean()))
