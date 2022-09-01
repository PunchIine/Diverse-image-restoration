import torch
from .base_model import BaseModel
from . import network, base_function, external_function
from .external_function import PD_Loss, TV_Loss, PerceptualLoss, Diversityloss
from util import task, MS_L1loss
import numpy as np
import itertools
from torchvision import transforms
from functools import reduce

toPIL = transforms.ToPILImage()

class pdgan(BaseModel):
    """This class implements the pluralistic image completion, for 256*256 resolution image inpainting"""
    def name(self):
        return "PD_GAN"

    @staticmethod
    def modify_options(parser, pd_is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--output_scale', type=int, default=4, help='# of number of the output scale')
        if pd_is_train:
            parser.add_argument('--lambda_rec', type=float, default=0.2, help='weight for image reconstruction loss')
            parser.add_argument('--lambda_g', type=float, default=1.0, help='weight for generation loss')
            parser.add_argument('--lambda_p', type=float, default=10.0, help='weight')

        return parser

    def __init__(self, opt):
        """Initial the PD_GAN model"""
        BaseModel.__init__(self, opt)

        self.batchSize = opt.batchSize

        self.loss_names = ['app_g', 'ad_g', 'img_d', 'pd', 'tv', 'diver']
        self.visual_names = ['img_m', 'img_c', 'img_truth', 'img_out_A', 'img_out_B', 'img_g_A', 'img_g_B']
        self.model_names = ['G_pd', 'D_pd']
        self.features = []

        # define the inpainting model
        self.net_G_pd = network.define_pd_g(ngf=32, z_nc=128, img_f=128, L=0, layers=5, output_scale=opt.output_scale,
                                            norm='instance', activation='LeakyReLU', init_type='orthogonal', gpu_ids=opt.gpu_ids, use_gated=opt.use_gated)
        # define the discriminator model
        self.net_D_pd = network.define_pd_d(ndf=32, img_f=128, layers=4, model_type='Mult', init_type='orthogonal', gpu_ids=opt.gpu_ids, use_gated=False)


        if self.pd_isTrain:
            # define the loss functions
            self.GANloss = external_function.GANLoss(opt.gan_mode)
            self.L1loss = torch.nn.L1Loss()
            self.L2loss = torch.nn.MSELoss()
            self.Ms_L1loss = MS_L1loss.MS_SSIM_L1_LOSS()
            self.PD_loss = PD_Loss()
            self.TV_loss = TV_Loss()
            self.Perceptual_loss = PerceptualLoss()
            self.Diversity_loss = Diversityloss()
            # define the optimizer
            self.optimizer_G = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_G_pd.parameters())), lr=opt.lr, betas=(0.0, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(filter(lambda p: p.requires_grad, self.net_D_pd.parameters())), lr=opt.lr, betas=(0.0, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        # load the pretrained model and schedulers
        self.pd_setup(opt)

    def set_input(self, input, epoch=0):
        """Unpack input data from the data loader and perform necessary pre-process steps"""
        self.input = input
        self.image_paths = self.input['img_path']
        self.img = input['img']
        self.mask = input['mask']

        if len(self.gpu_ids) > 0:
            self.img = self.img.cuda(self.gpu_ids[0])
            self.mask = self.mask.cuda(self.gpu_ids[0])

        # get I_m and I_c for image with mask and complement regions for training
        self.img_truth = self.img * 2 - 1
        self.img_m = self.mask * self.img_truth
        self.img_c = (1 - self.mask) * self.img_truth

        # get multiple scales image ground truth and mask for training
        self.scale_img = task.scale_pyramid(self.img_truth, self.opt.output_scale)
        self.scale_mask = task.scale_pyramid(self.mask, self.opt.output_scale)

    def test(self, img_p, num):
        """Forward function used in test time"""
        # save the groundtruth and masked image
        if num == 0:
            self.save_results(self.img_truth, data_name='truth')
        # self.save_results(self.img_m, data_name='mask')

        # encoder process
        # distribution, f = self.net_E(self.img_m)
        # q_distribution = torch.distributions.Normal(distribution[-1][0], distribution[-1][1])
        # scale_mask = task.scale_img(self.mask, size=[f[2].size(2), f[2].size(3)])

        # decoder process
        # print(self.opt)
        for i in range(self.opt.nsampling):
            z = torch.Tensor(np.random.normal(0, 1, (self.batchSize, 128, 8, 8)))
            self.img_g, _, _ = self.net_G_pd(z, self.mask, img_p)
            self.img_out = (1 - self.mask) * self.img_g[-1].detach() + self.mask * self.img_m
            self.score = self.net_D_pd(self.img_out)
            self.save_results(self.img_out, num, data_name='out')
            num += 1

        return num

    def forward(self, img_p):
        # self.mask = mask
        # self.img_truth = img_truth
        # pic = toPIL(self.img_truth.chunk(chunks=4)[-1].view(3, 256, 256))
        # pic.save('truth1.jpg')
        # pic = toPIL(self.mask.chunk(chunks=4)[-1].view(3, 256, 256))
        # pic.save('mask1.jpg')
        z_A = torch.Tensor(np.random.normal(0, 1, (self.batchSize, 128, 8, 8)))
        z_B = torch.Tensor(np.random.normal(0, 1, (self.batchSize, 128, 8, 8)))
        # print(z_A is z_B)
        if self.opt.use_gated:
            results, attn = self.net_G_pd(torch.cat([z_A, z_B], dim=0),
                                                    torch.cat([self.mask, self.mask], dim=0),
                                                    torch.cat([img_p, img_p], dim=0))
        else:
            results, attn = self.net_G_pd(torch.cat([z_A, z_B], dim=0),
                                                torch.cat([self.mask, self.mask], dim=0),
                                                torch.cat([img_p, img_p], dim=0))
        self.img_g_A = []
        self.img_g_B = []
        for result in results:
            img_g_A, img_g_B = torch.split(result, self.batchSize, dim=0)
            self.img_g_A.append(img_g_A)
            self.img_g_B.append(img_g_B)
        self.img_out_A = (1-self.mask) * self.img_g_A[-1].detach() + self.mask * self.img_truth
        self.img_out_B = (1-self.mask) * self.img_g_B[-1].detach() + self.mask * self.img_truth
        # print(self.img_out_A is self.img_out_B)
        # pic = toPIL(self.img_out_A.chunk(chunks=4)[-1].view(3, 256, 256))
        # pic.save('out.jpg')

        # pic = toPIL(self.mask.chunk(chunks=4)[-1].view(3, 256, 256))
        # pic.save('mask.jpg')
        # print(self.mask.chunk(chunks=4)[-1].view(3, 256, 256))

        # pic = toPIL(self.img_truth.chunk(chunks=4)[-1].view(3, 256, 256))
        # pic.save('truth.jpg')

        # pic = toPIL(self.img_g[-1].chunk(chunks=4)[-1].view(3, 256, 256))
        # pic.save('img_g.jpg')

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        D_real = netD(real)
        D_real_loss1 = self.GANloss(D_real[0], True, True)
        D_real_loss2 = self.GANloss(D_real[1], True, True)
        D_real_loss3 = self.GANloss(D_real[2], True, True)
        D_real_loss = (D_real_loss1 + D_real_loss2 + D_real_loss3) / 3.0
        # fake
        # print("fake" + str(fake.shape))
        D_fake = netD(fake.detach())
        D_fake_loss1 = self.GANloss(D_fake[0], False, True)
        D_fake_loss2 = self.GANloss(D_fake[1], False, True)
        D_fake_loss3 = self.GANloss(D_fake[2], False, True)
        D_fake_loss = (D_fake_loss1 + D_fake_loss2 + D_fake_loss3) / 3.0
        # loss for discriminator
        if self.opt.gan_mode == 'wgandiv':
            D_loss = D_real_loss - D_fake_loss
        else:
            D_loss = (D_real_loss + D_fake_loss) * 0.5
        # gradient penalty for wgan-gp
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, gradients = external_function.cal_gradient_penalty(netD, real, fake.detach())
            D_loss += gradient_penalty
        # gradient penalty for wgan-div
        if self.opt.gan_mode == 'wgandiv':
            gradient_penalty_div = external_function.cal_gradient_penalty_div(netD, real, fake.detach())
            D_loss += gradient_penalty_div

        D_loss.backward()

        return D_loss

    def backward_D(self):
        """Calculate the GAN loss for the discriminators"""
        base_function._unfreeze(self.net_D_pd)
        self.loss_img_d_A = self.backward_D_basic(self.net_D_pd, self.img_truth, self.img_g_A[-1])
        self.loss_img_d_B = self.backward_D_basic(self.net_D_pd, self.img_truth, self.img_g_B[-1])
        self.loss_img_d = self.loss_img_d_A + self.loss_img_d_B

    def backward_G(self):
        """Calculate training loss for the generator"""
        # generator adversarial loss
        base_function._freeze(self.net_D_pd)
        # g loss fake
        D_fake_A = self.net_D_pd(self.img_g_A[-1])
        D_fake_B = self.net_D_pd(self.img_g_B[-1])
        self.loss_ad_g_A1 = self.GANloss(D_fake_A[0], True, False) * self.opt.lambda_g
        self.loss_ad_g_A2 = self.GANloss(D_fake_A[1], True, False) * self.opt.lambda_g
        self.loss_ad_g_A3 = self.GANloss(D_fake_A[2], True, False) * self.opt.lambda_g
        self.loss_ad_g_A = (self.loss_ad_g_A1 + self.loss_ad_g_A2 + self.loss_ad_g_A3) / 3.0
        self.loss_ad_g_B1 = self.GANloss(D_fake_B[0], True, False) * self.opt.lambda_g
        self.loss_ad_g_B2 = self.GANloss(D_fake_B[1], True, False) * self.opt.lambda_g
        self.loss_ad_g_B3 = self.GANloss(D_fake_B[2], True, False) * self.opt.lambda_g
        self.loss_ad_g_B = (self.loss_ad_g_B1 + self.loss_ad_g_B2 + self.loss_ad_g_B3) / 3.0
        self.loss_ad_g = self.loss_ad_g_A + self.loss_ad_g_B

        # calculate l1 loss ofr multi-scale outputs
        loss_app_g_A = 0
        for i, (img_fake_i, img_real_i, mask_i) in enumerate(zip(self.img_g_A, self.scale_img, self.scale_mask)):
            loss_app_g_A += self.Ms_L1loss(img_fake_i, img_real_i)
        self.loss_app_g_A = loss_app_g_A * self.opt.lambda_rec

        loss_app_g_B = 0
        for i, (img_fake_i, img_real_i, mask_i) in enumerate(zip(self.img_g_B, self.scale_img, self.scale_mask)):
            loss_app_g_B += self.Ms_L1loss(img_fake_i, img_real_i)
        self.loss_app_g_B = loss_app_g_B * self.opt.lambda_rec

        self.loss_app_g = self.loss_app_g_A + self.loss_app_g_B

        # Perceptual Diversity Loss
        # loss_pd = 0
        # for feature in self.features:
        #     feature_A, feature_B = torch.split(feature, self.batchSize, dim=0)
        #     loss_pd += self.PD_loss(feature_A, feature_B)
        # self.loss_pd = 1 / (loss_pd + 1e-5)
        self.loss_pd = reduce(
                        lambda x, y: x + y,
                            [
                                self.Perceptual_loss(self.img_g_A[-1], self.img_truth),
                                self.Perceptual_loss(self.img_g_B[-1], self.img_truth),
                            ],
                        ) * self.opt.lambda_p

        # diversity loss
        self.loss_diver = 1 / (self.Diversity_loss(self.img_g_A[-1], self.img_g_B[-1]) + 1e-5)

        # TV loss
        loss_tv_A = 0
        loss_tv_B = 0
        comp_A = self.img_g_A[-1] * (1 - self.mask) + self.mask * self.img_truth
        comp_B = self.img_g_B[-1] * (1 - self.mask) + self.mask * self.img_truth
        loss_tv_A = self.TV_loss(comp_A, self.mask, "mean")
        loss_tv_B = self.TV_loss(comp_B, self.mask, "mean")
        self.loss_tv = loss_tv_A + loss_tv_B

        total_loss = 0

        for name in self.loss_names:
            if name != 'img_d' and name != 'img_d_rec':
                total_loss += getattr(self, "loss_" + name)

        total_loss.backward()


    def pd_optimize_parameters(self, img_p):
        """update network weights"""
        # compute the image completion results
        self.forward(img_p)
        # optimize the discrinimator network parameters
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # optimize the completion network parameters
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
