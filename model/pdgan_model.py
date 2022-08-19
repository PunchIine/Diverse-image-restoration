import torch
from .base_model import BaseModel
from . import network, base_function, external_function
from util import task, MS_L1loss
import numpy as np
import itertools


class pdgan(BaseModel):
    """This class implements the pluralistic image completion, for 256*256 resolution image inpainting"""
    def name(self):
        return "PD_GAN"

    @staticmethod
    def modify_options(parser, pd_is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--output_scale', type=int, default=4, help='# of number of the output scale')
        if pd_is_train:
            parser.add_argument('--train_paths', type=str, default='two', help='training strategies with one path or two paths')
            parser.add_argument('--lambda_rec', type=float, default=20.0, help='weight for image reconstruction loss')
            parser.add_argument('--lambda_kl', type=float, default=20.0, help='weight for kl divergence loss')
            parser.add_argument('--lambda_g', type=float, default=1.0, help='weight for generation loss')

        return parser

    def __init__(self, opt):
        """Initial the PD_GAN model"""
        BaseModel.__init__(self, opt)

        self.batchSize = opt.batchSize

        self.loss_names = ['app_g', 'ad_g', 'img_d']
        self.visual_names = ['img_m', 'img_c', 'img_truth', 'img_out', 'img_g', 'img_rec']
        self.model_names = ['G_pd', 'D_pd']
        self.distribution = []

        # define the inpainting model
        self.net_G_pd = network.define_pd_g(ngf=32, z_nc=128, img_f=128, L=0, layers=5, output_scale=opt.output_scale,
                                            norm='group', activation='LeakyReLU', init_type='orthogonal', gpu_ids=opt.gpu_ids)
        # define the discriminator model
        self.net_D_pd = network.define_pd_d(ndf=32, img_f=128, layers=5, model_type='ResDis', init_type='orthogonal', gpu_ids=opt.gpu_ids)

        if self.pd_isTrain:
            # define the loss functions
            self.GANloss = external_function.GANLoss(opt.gan_mode)
            self.L1loss = torch.nn.L1Loss()
            self.L2loss = torch.nn.MSELoss()
            self.Ms_L1loss = MS_L1loss.MS_SSIM_L1_LOSS()
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

    def forward(self, img_p, mask):
        z = torch.Tensor(np.random.normal(0, 1, (self.batchSize, 128, self.batchSize, self.batchSize)))
        results, attn = self.net_G_pd(z, mask, img_p)
        self.img_g = []
        for result in results:
            img_g = result[-1]
            self.img_g.append(img_g)
        self.img_out = (1-self.mask) * self.img_g[-1].detach() + self.mask * self.img_truth

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator"""
        # Real
        D_real = netD(real)
        D_real_loss = self.GANloss(D_real, True, True)
        # fake
        D_fake = netD(fake.detach())
        D_fake_loss = self.GANloss(D_fake, False, True)
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
        self.loss_img_d = self.backward_D_basic(self.net_D_pd, self.img_truth, self.img_g[-1])

    def backward_G(self):
        """Calculate training loss for the generator"""

        # generator adversarial loss
        base_function._freeze(self.net_D_pd)

        # g loss fake
        D_fake = self.net_D_pd(self.img_g[-1])
        self.loss_ad_g = self.GANloss(D_fake, True, False) * self.opt.lambda_g

        # calculate l1 loss ofr multi-scale outputs
        loss_app_g = 0, 0
        for i, (img_fake_i, img_real_i, mask_i) in enumerate(zip(self.img_g, self.scale_img, self.scale_mask)):
            if self.opt.train_paths == "one":
                loss_app_g += self.Ms_L1loss(img_fake_i, img_real_i)
            elif self.opt.train_paths == "two":
                loss_app_g += self.Ms_L1loss(img_fake_i*mask_i, img_real_i*mask_i)
        self.loss_app_g = loss_app_g * self.opt.lambda_rec

        # if one path during the training, just calculate the loss for generation path

        total_loss = 0

        for name in self.loss_names:
            if name != 'img_d' and name != 'img_d_rec':
                total_loss += getattr(self, "loss_" + name)

        total_loss.backward()


    def pd_optimize_parameters(self, img_p, mask):
        """update network weights"""
        # compute the image completion results
        self.forward(img_p, mask)
        # optimize the discrinimator network parameters
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # optimize the completion network parameters
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
