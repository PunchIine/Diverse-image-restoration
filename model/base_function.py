import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from .external_function import SpectralNorm, GroupNorm
import numpy as np


######################################################################################
# base function for network structure
######################################################################################


def init_weights(net, init_type='normal', gain=0.02):
    """Get different initial method for the network weights"""

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def get_norm_layer(norm_type='batch'):
    """Get the normalization layer for the networks"""
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, momentum=0.1, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'group':
        norm_layer = functools.partial(GroupNorm)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_nonlinearity_layer(activation_type='PReLU'):
    """Get the activation layer for the networks"""
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU()
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU()
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.1)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer


def get_scheduler(optimizer, opt):
    """Get the training learning rate for different epoch"""
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + 1 + opt.iter_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'exponent':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def print_network(net):
    """print the network"""
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('total number of parameters: %.3f M' % (num_params / 1e6))


def init_net(net, init_type='normal', activation='relu', gpu_ids=[]):
    """print the network structure and initial the network"""
    print_network(net)

    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.cuda()
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def _freeze(*args):
    """freeze the network for forward process"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = False


def _unfreeze(*args):
    """ unfreeze the network for parameter update"""
    for module in args:
        if module:
            for p in module.parameters():
                p.requires_grad = True


def spectral_norm(module, use_spect=True):
    """use spectral normal layer to stable the training process"""
    if use_spect:
        return SpectralNorm(module)
    else:
        return module


def coord_conv(input_nc, output_nc, use_spect=False, use_coord=False, with_r=False, use_gated=False, **kwargs):
    """use coord convolution layer to add position information"""
    if use_gated:
        return Gated_Conv(input_nc, output_nc, **kwargs)
    elif use_coord:
        return CoordConv(input_nc, output_nc, with_r, use_spect, **kwargs)
    else:
        return spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)


######################################################################################
# Network basic function
######################################################################################
class AddCoords(nn.Module):
    """
    Add Coords to a tensor
    """

    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def forward(self, x):
        """
        :param x: shape (batch, channel, x_dim, y_dim)
        :return: shape (batch, channel+2, x_dim, y_dim)
        """
        B, _, x_dim, y_dim = x.size()

        # coord calculate
        xx_channel = torch.arange(x_dim).repeat(B, 1, y_dim, 1).type_as(x)
        yy_cahnnel = torch.arange(y_dim).repeat(B, 1, x_dim, 1).permute(0, 1, 3, 2).type_as(x)
        # normalization
        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_cahnnel = yy_cahnnel.float() / (y_dim - 1)
        xx_channel = xx_channel * 2 - 1
        yy_cahnnel = yy_cahnnel * 2 - 1

        ret = torch.cat([x, xx_channel, yy_cahnnel], dim=1)

        if self.with_r:
            rr = torch.sqrt(xx_channel ** 2 + yy_cahnnel ** 2)
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):
    """
    CoordConv operation
    """

    def __init__(self, input_nc, output_nc, with_r=False, use_spect=False, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(with_r=with_r)
        input_nc = input_nc + 2
        if with_r:
            input_nc = input_nc + 1
        self.conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)

        return ret


class ResBlock(nn.Module):
    """
    Define an Residual block for different types
    """

    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 sample_type='none', use_spect=False, use_coord=False, use_gated=False):
        super(ResBlock, self).__init__()

        hidden_nc = output_nc if hidden_nc is None else hidden_nc
        self.in_nc = input_nc
        self.hi_nc = hidden_nc
        self.ou_nc = output_nc
        self.sample = True
        if sample_type == 'none':
            self.sample = False
        elif sample_type == 'up':
            output_nc = output_nc * 4
            self.pool = nn.PixelShuffle(upscale_factor=2)
        elif sample_type == 'down':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError('sample type [%s] is not found' % sample_type)

        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        self.conv1 = coord_conv(input_nc, hidden_nc, use_spect, use_coord, use_gated=use_gated, **kwargs)
        self.conv2 = coord_conv(hidden_nc, output_nc, use_spect, use_coord, use_gated=use_gated, **kwargs)
        self.bypass = coord_conv(input_nc, output_nc, use_spect, use_coord, use_gated=use_gated, **kwargs_short)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, self.conv1, nonlinearity, self.conv2, )
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, self.conv1, norm_layer(hidden_nc),
                                       nonlinearity, self.conv2, )

        self.shortcut = nn.Sequential(self.bypass, )

    def forward(self, x):
        if self.sample:
            out = self.pool(self.model(x)) + self.pool(self.shortcut(x))
        else:
            out = self.model(x) + self.shortcut(x)

        return out


class ResBlockEncoderOptimized(nn.Module):
    """
    Define an Encoder block for the first layer of the discriminator and representation network
    """

    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False,
                 use_coord=False, use_gated=False):
        super(ResBlockEncoderOptimized, self).__init__()

        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}

        self.conv1 = coord_conv(input_nc, output_nc, use_spect, use_coord, use_gated, **kwargs)
        self.conv2 = coord_conv(output_nc, output_nc, use_spect, use_coord, use_gated, **kwargs)
        self.bypass = coord_conv(input_nc, output_nc, use_spect, use_coord, use_gated, **kwargs_short)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(self.conv1, nonlinearity, self.conv2, nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            self.model = nn.Sequential(self.conv1, norm_layer(output_nc), nonlinearity, self.conv2,
                                       nn.AvgPool2d(kernel_size=2, stride=2))

        self.shortcut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2), self.bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)

        return out


class ResBlockDecoder(nn.Module):
    """
    Define a decoder block
    """

    def __init__(self, input_nc, output_nc, hidden_nc=None, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False):
        super(ResBlockDecoder, self).__init__()

        hidden_nc = output_nc if hidden_nc is None else hidden_nc

        self.conv1 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, kernel_size=3, stride=1, padding=1), use_spect)
        self.conv2 = spectral_norm(
            nn.ConvTranspose2d(hidden_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)
        self.bypass = spectral_norm(
            nn.ConvTranspose2d(input_nc, output_nc, kernel_size=3, stride=2, padding=1, output_padding=1), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, self.conv1, nonlinearity, self.conv2, )
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, self.conv1, norm_layer(hidden_nc),
                                       nonlinearity, self.conv2, )

        self.shortcut = nn.Sequential(self.bypass)

    def forward(self, x):
        out = self.model(x) + self.shortcut(x)

        return out


class Output(nn.Module):
    """
    Define the output layer
    """

    def __init__(self, input_nc, output_nc, kernel_size=3, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(),
                 use_spect=False, use_coord=False, use_gated=False):
        super(Output, self).__init__()

        kwargs = {'kernel_size': kernel_size, 'padding': 0, 'bias': True}

        self.conv1 = coord_conv(input_nc, output_nc, use_spect, use_coord, use_gated=use_gated, **kwargs)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(nonlinearity, nn.ReflectionPad2d(int(kernel_size / 2)), self.conv1, nn.Tanh())
        else:
            self.model = nn.Sequential(norm_layer(input_nc), nonlinearity, nn.ReflectionPad2d(int(kernel_size / 2)),
                                       self.conv1, nn.Tanh())

    def forward(self, x):
        out = self.model(x)

        return out


class Auto_Attn(nn.Module):
    """ Short+Long attention Layer"""

    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d):
        super(Auto_Attn, self).__init__()
        self.input_nc = input_nc

        self.query_conv = nn.Conv2d(input_nc, input_nc // 4, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        self.model = ResBlock(int(input_nc * 2), input_nc, input_nc, norm_layer=norm_layer, use_spect=True)

    def forward(self, x, pre=None, mask=None):
        """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        # print(x.shape)
        B, C, W, H = x.size()
        proj_query = self.query_conv(x).view(B, -1, W * H)  # B X (N)X C
        proj_key = proj_query  # B X C x (N)

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = x.view(B, -1, W * H)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)

        out = self.gamma * out + x

        if type(pre) != type(None):
            # using long distance attention layer to copy information from valid regions
            context_flow = torch.bmm(pre.view(B, -1, W * H), attention.permute(0, 2, 1)).view(B, -1, W, H)
            context_flow = self.alpha * (1 - mask) * context_flow + (mask) * pre
            out = self.model(torch.cat([out, context_flow], dim=1))

        return out, attention


class SimAM(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class HardSPDNorm(nn.Module):
    def __init__(self, n, k, p_input_nc, F_in_nc):
        super(HardSPDNorm, self).__init__()
        self.n = n
        self.k = k
        self.F_in_nc = F_in_nc
        self.gamma_conv = Gated_Conv(2 * p_input_nc, self.F_in_nc, kernel_size=3, stride=1, padding=1)
        self.beta_conv = Gated_Conv(2 * p_input_nc, self.F_in_nc, kernel_size=3, stride=1, padding=1)
        self.dsample_p = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dsample_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.innorm = nn.InstanceNorm2d(self.F_in_nc, affine=False)

    def forward(self, F_in, img_p, mask, n_ds):
        # print(self.F_in_nc)
        mask = mask.clone()
        # downsample
        for i in range(n_ds):
            img_p = self.dsample_p(img_p)
            mask = self.dsample_m(mask)
        img_p = torch.cat([img_p, mask], dim=1)
        mask, _, _ = torch.chunk(mask, dim=1, chunks=3)
        # D_h
        kernel = torch.ones(mask.shape[1], mask.shape[1], 3, 3).cuda()
        D_h = mask
        msk = D_h.detach()
        msk = torch.where(msk == 1, True, False)
        for i in range(1, self.n + 1):
            D_h = F.conv2d(D_h, kernel, stride=1, padding=1)
            tmp = D_h.detach()
            tmp = torch.where(tmp > 0, True, False)
            tmp = tmp & ~msk
            msk = msk | tmp
            mask[tmp] = 1 / self.k ** (i)
            D_h = mask

        gamma_hp = self.gamma_conv(img_p)
        beta_hp = self.beta_conv(img_p)
        gamma_hd = gamma_hp * D_h
        beta_hd = beta_hp * D_h

        F_in_norm = self.innorm(F_in)
        # print(gamma_hd.shape)
        # print(F_in.shape)
        F_hard = F_in_norm * (gamma_hd + 1) + beta_hd

        return F_hard


class SoftSPDNorm(nn.Module):
    def __init__(self, p_input_nc, F_in_nc):
        super(SoftSPDNorm, self).__init__()
        self.gamma_conv = Gated_Conv(2 * p_input_nc, F_in_nc, kernel_size=3, stride=1, padding=1)
        self.beta_conv = Gated_Conv(2 * p_input_nc, F_in_nc, kernel_size=3, stride=1, padding=1)
        self.p_conv = Gated_Conv(6, F_in_nc, stride=1, kernel_size=1)
        self.f_conv = Gated_Conv(2 * F_in_nc, 1, kernel_size=1, stride=1)
        self.dsample_p = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dsample_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()
        self.innorm = nn.InstanceNorm2d(F_in_nc, affine=False)

    def forward(self, F_in, img_p, mask, n_ds):
        mask = mask.clone()
        # downsample
        for i in range(n_ds):
            img_p = self.dsample_p(img_p)
            mask = self.dsample_m(mask)
        img_p = torch.cat([img_p, mask], dim=1)
        mask, _, _ = torch.chunk(mask, dim=1, chunks=3)
        F_in_norm = self.innorm(F_in)

        F_p = self.p_conv(img_p)
        F_mix = torch.cat([F_p, F_in_norm], dim=1)
        F_conv = self.f_conv(F_mix)
        D_s = self.sigmoid(F_conv * (1 - mask) + mask)

        gamma_sp = self.gamma_conv(img_p)
        beta_sp = self.beta_conv(img_p)

        gamma_sd = gamma_sp * D_s
        beta_sd = beta_sp * D_s

        f_in = F_in_norm * gamma_sd
        F_soft = f_in + beta_sd

        return F_soft, mask


class ResBlockSPDNorm(nn.Module):
    def __init__(self, F_in_nc, p_input_nc, n_ds, n, k):
        super(ResBlockSPDNorm, self).__init__()
        self.HardSPDNorm = HardSPDNorm(n, k, p_input_nc, F_in_nc)
        self.SoftSPDNorm = SoftSPDNorm(p_input_nc, F_in_nc)

        self.relu = nn.ReLU()
        self.Conv1 = Gated_Conv(F_in_nc, F_in_nc, stride=1, padding=1, kernel_size=3)
        self.Conv2 = Gated_Conv(F_in_nc, F_in_nc, stride=1, padding=1, kernel_size=3)
        self.Conv3 = Gated_Conv(F_in_nc, F_in_nc, stride=1, padding=1, kernel_size=3)

        self.n_ds = n_ds

    def forward(self, F_in, img_p, mask):
        # the HardSPDNorm
        img_p_re = F.interpolate(
            img_p, size=F_in.size()[2:], mode="nearest"
        )
        mask_re = F.interpolate(mask, size=F_in.size()[2:], mode="nearest")
        out_h1 = self.HardSPDNorm(F_in, img_p_re, mask_re, self.n_ds)
        out_h1 = self.relu(out_h1)
        out_h1 = self.Conv1(out_h1)
        out_h = self.HardSPDNorm(out_h1, img_p_re, mask_re, self.n_ds)
        out_h = self.relu(out_h)
        out_h = self.Conv2(out_h)
        # the SoftSPDNorm
        out_s, mask = self.SoftSPDNorm(F_in, img_p_re, mask_re, self.n_ds)
        out_s = self.relu(out_s)
        out_s = self.Conv3(out_s)
        # output
        out = out_h + out_s

        return out, mask


class Gated_Conv(nn.Module):
    """
    Gated Convolution with spetral normalization
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=False, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(Gated_Conv, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.activation = activation
        self.batch_norm = batch_norm
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.mask_conv2d = torch.nn.utils.spectral_norm(self.mask_conv2d)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def gated(self, mask):
        return self.sigmoid(mask)
        #return torch.clamp(mask, -1, 1)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x


class SPDNormResnetBlock(nn.Module):
    def __init__(self, fin, fout, mask_number, mask_ks):
        super().__init__()
        nhidden = 128
        fmiddle = min(fin, fout)
        lable_nc = 3
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        self.learned_shortcut = True
        # apply spectral norm if specified
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(self.conv_s)
        # define normalization layers
        self.norm_0 = SPDNorm(fin, norm_type="position")
        self.norm_1 = SPDNorm(fmiddle, norm_type="position")
        self.norm_s = SPDNorm(fin, norm_type="position")
        # define the mask weight
        self.v = nn.Parameter(torch.zeros(1))
        self.activeweight = nn.Sigmoid()
        # define the feature and mask process conv
        self.mask_number = mask_number
        self.mask_ks = mask_ks
        pw_mask = int(np.ceil((self.mask_ks - 1.0) / 2))
        self.mask_conv = MaskGet(1, 1, kernel_size=self.mask_ks, padding=pw_mask)
        self.conv_to_f = nn.Sequential(
            Gated_Conv(lable_nc * 2, nhidden, kernel_size=3, padding=1),
            nn.InstanceNorm2d(nhidden),
            nn.ReLU(),
            Gated_Conv(nhidden, fin, kernel_size=3, padding=1),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(fin * 2, fin, kernel_size=3, padding=1), nn.Sigmoid()
        )

    # the semantic feature prior_f form pretrained encoder
    def forward(self, x, prior_image, mask):
        """

        Args:
            x: input feature
            prior_image: the output of PCConv
            mask: mask


        """
        # prepare the forward
        b, c, h, w = x.size()
        prior_image_resize = F.interpolate(
            prior_image, size=x.size()[2:], mode="nearest"
        )
        mask_resize = F.interpolate(mask, size=x.size()[2:], mode="nearest")
        prior_image_resize = torch.cat([prior_image_resize, mask_resize], dim=1)
        mask_resize, _, _ = torch.chunk(mask_resize, dim=1, chunks=3)
        # Mask Original and Res path  res weight/ value attention
        prior_feature = self.conv_to_f(prior_image_resize)
        soft_map = self.attention(torch.cat([prior_feature, x], 1))
        # print(soft_map.shape)
        # print(mask_resize.shape)
        soft_map = (1 - mask_resize) * soft_map + mask_resize
        # Mask weight for next process
        mask_pre = mask_resize
        hard_map = 0.0
        for i in range(self.mask_number):
            mask_out = self.mask_conv(mask_pre)
            mask_generate = (mask_out - mask_pre) * (
                1 / (torch.exp(torch.tensor(i + 1).cuda()))
            )
            mask_pre = mask_out
            hard_map = hard_map + mask_generate
        hard_map_inner = (1 - mask_out) * (1 / (torch.exp(torch.tensor(i + 1).cuda())))
        hard_map = hard_map + mask_resize + hard_map_inner
        soft_out = self.conv_s(self.norm_s(x, prior_image_resize, soft_map))
        hard_out = self.conv_0(self.actvn(self.norm_0(x, prior_image_resize, hard_map)))
        hard_out = self.conv_1(
            self.actvn(self.norm_1(hard_out, prior_image_resize, hard_map))
        )
        # Res add
        out = soft_out + hard_out
        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


def PositionalNorm2d(x, epsilon=1e-5):
    # x: B*C*W*H normalize in C dim
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    output = (x - mean) / std
    return output


class SPDNorm(nn.Module):
    def __init__(self, norm_channel, norm_type="batch"):
        super().__init__()
        label_nc = 3
        param_free_norm_type = norm_type
        ks = 3
        if param_free_norm_type == "instance":
            self.param_free_norm = nn.InstanceNorm2d(norm_channel, affine=False)
        elif param_free_norm_type == "batch":
            self.param_free_norm = nn.BatchNorm2d(norm_channel, affine=False)
        elif param_free_norm_type == "position":
            self.param_free_norm = PositionalNorm2d
        else:
            raise ValueError(
                "%s is not a recognized param-free norm type in SPADE"
                % param_free_norm_type
            )

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        pw = ks // 2
        nhidden = 128
        self.mlp_activate = nn.Sequential(
            Gated_Conv(label_nc * 2, nhidden, kernel_size=ks, padding=pw), nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_channel, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_channel, kernel_size=ks, padding=pw)

    def forward(self, x, prior_f, weight):
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on condition feature
        actv = self.mlp_activate(prior_f)
        gamma = self.mlp_gamma(actv) * weight
        beta = self.mlp_beta(actv) * weight
        # apply scale and bias
        out = normalized * (1 + gamma) + beta
        return out


class MaskGet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        self.mask_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            False,
        )

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

    def forward(self, input):
        # depart from partial conv
        # hole region should sed to 0
        with torch.no_grad():
            output_mask = self.mask_conv(input)
        no_update_holes = output_mask == 0
        new_mask = torch.ones_like(output_mask)
        new_mask = new_mask.masked_fill_(no_update_holes.bool(), 0.0)
        return new_mask

# class SPDNormResnetBlock(nn.Module):
#     def __init__(self, fin, fout, mask_number, mask_ks):
#         super().__init__()
#         nhidden = 128
#         fmiddle = min(fin, fout)
#         lable_nc = 3
#         # create conv layers
#         self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
#         self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
#         self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
#         self.learned_shortcut = True
#         # apply spectral norm if specified
#         self.conv_0 = spectral_norm(self.conv_0)
#         self.conv_1 = spectral_norm(self.conv_1)
#         if self.learned_shortcut:
#             self.conv_s = spectral_norm(self.conv_s)
#         # define normalization layers
#         self.norm_0 = SPDNorm(fin, norm_type="position")
#         self.norm_1 = SPDNorm(fmiddle, norm_type="position")
#         self.norm_s = SPDNorm(fin, norm_type="position")
#         # define the mask weight
#         self.v = nn.Parameter(torch.zeros(1))
#         self.activeweight = nn.Sigmoid()
#         # define the feature and mask process conv
#         self.mask_number = mask_number
#         self.mask_ks = mask_ks
#         pw_mask = int(np.ceil((self.mask_ks - 1.0) / 2))
#         self.mask_conv = MaskGet(1, 1, kernel_size=self.mask_ks, padding=pw_mask)
#         self.conv_to_f = nn.Sequential(
#             nn.Conv2d(lable_nc, nhidden, kernel_size=3, padding=1),
#             nn.InstanceNorm2d(nhidden),
#             nn.ReLU(),
#             nn.Conv2d(nhidden, fin, kernel_size=3, padding=1),
#         )
#         self.attention = nn.Sequential(
#             nn.Conv2d(fin * 2, fin, kernel_size=3, padding=1), nn.Sigmoid()
#         )

    # # the semantic feature prior_f form pretrained encoder
    # def forward(self, x, prior_image, mask):
    #     """

        # Args:
        #     x: input feature
        #     prior_image: the output of PCConv
        #     mask: mask


        # """
        # # prepare the forward
        # b, c, h, w = x.size()
        # prior_image_resize = F.interpolate(
        #     prior_image, size=x.size()[2:], mode="nearest"
        # )
        # mask_resize = F.interpolate(mask, size=x.size()[2:], mode="nearest")
        # mask_resize, _, _ = torch.chunk(mask_resize, dim=1, chunks=3)
        # # Mask Original and Res path  res weight/ value attention
        # prior_feature = self.conv_to_f(prior_image_resize)
        # soft_map = self.attention(torch.cat([prior_feature, x], 1))
        # # print(soft_map.shape)
        # # print(mask_resize.shape)
        # soft_map = (1 - mask_resize) * soft_map + mask_resize
        # # Mask weight for next process
        # mask_pre = mask_resize
        # hard_map = 0.0
        # for i in range(self.mask_number):
        #     mask_out = self.mask_conv(mask_pre)
        #     mask_generate = (mask_out - mask_pre) * (
        #             1 / (torch.exp(torch.tensor(i + 1).cuda()))
        #     )
        #     mask_pre = mask_out
        #     hard_map = hard_map + mask_generate
        # hard_map_inner = (1 - mask_out) * (1 / (torch.exp(torch.tensor(i + 1).cuda())))
        # hard_map = hard_map + mask_resize + hard_map_inner
        # soft_out = self.conv_s(self.norm_s(x, prior_image_resize, soft_map))
        # hard_out = self.conv_0(self.actvn(self.norm_0(x, prior_image_resize, hard_map)))
        # hard_out = self.conv_1(
        #     self.actvn(self.norm_1(hard_out, prior_image_resize, hard_map))
        # )
        # # Res add
        # out = soft_out + hard_out
        # return out

    # def actvn(self, x):
    #     return F.leaky_relu(x, 2e-1)


# def PositionalNorm2d(x, epsilon=1e-5):
#     # x: B*C*W*H normalize in C dim
#     mean = x.mean(dim=1, keepdim=True)
#     std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
#     output = (x - mean) / std
#     return output


# class SPDNorm(nn.Module):
#     def __init__(self, norm_channel, norm_type="batch"):
#         super().__init__()
#         label_nc = 3
#         param_free_norm_type = norm_type
#         ks = 3
#         if param_free_norm_type == "instance":
#             self.param_free_norm = nn.InstanceNorm2d(norm_channel, affine=False)
#         elif param_free_norm_type == "batch":
#             self.param_free_norm = nn.BatchNorm2d(norm_channel, affine=False)
#         elif param_free_norm_type == "position":
#             self.param_free_norm = PositionalNorm2d
#         else:
#             raise ValueError(
#                 "%s is not a recognized param-free norm type in SPADE"
#                 % param_free_norm_type
#             )

        # # The dimension of the intermediate embedding space. Yes, hardcoded.
        # pw = ks // 2
        # nhidden = 128
        # self.mlp_activate = nn.Sequential(
        #     nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU()
        # )
        # self.mlp_gamma = nn.Conv2d(nhidden, norm_channel, kernel_size=ks, padding=pw)
        # self.mlp_beta = nn.Conv2d(nhidden, norm_channel, kernel_size=ks, padding=pw)

    # def forward(self, x, prior_f, weight):
    #     normalized = self.param_free_norm(x)
    #     # Part 2. produce scaling and bias conditioned on condition feature
    #     actv = self.mlp_activate(prior_f)
    #     gamma = self.mlp_gamma(actv) * weight
    #     beta = self.mlp_beta(actv) * weight
    #     # apply scale and bias
    #     out = normalized * (1 + gamma) + beta
    #     return out


# class MaskGet(nn.Module):
#     def __init__(
#             self,
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=1,
#             padding=0,
#             dilation=1,
#             groups=1,
#             bias=True,
#     ):
#         super().__init__()
#         self.mask_conv = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride,
#             padding,
#             dilation,
#             groups,
#             False,
#         )

        # torch.nn.init.constant_(self.mask_conv.weight, 1.0)

    # def forward(self, input):
    #     # depart from partial conv
    #     # hole region should sed to 0
    #     with torch.no_grad():
    #         output_mask = self.mask_conv(input)
    #     no_update_holes = output_mask == 0
    #     new_mask = torch.ones_like(output_mask)
    #     new_mask = new_mask.masked_fill_(no_update_holes.bool(), 0.0)
    #     return new_mask

