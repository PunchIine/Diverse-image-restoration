import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import torchvision.models as models
import copy
# import lpips

####################################################################################################
# spectral normalization layer to decouple the magnitude of a weight tensor
####################################################################################################
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.max3 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])
        for x in range(16, 17):
            self.max3.add_module(str(x), features[x])

        for x in range(17, 19):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(19, 21):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(23, 26):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(26, 28):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(28, 30):
            self.relu5_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        max_3 = self.max3(relu3_3)

        relu4_1 = self.relu4_1(max_3)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)

        relu5_1 = self.relu5_1(relu4_3)
        relu5_2 = self.relu5_1(relu5_1)
        relu5_3 = self.relu5_1(relu5_2)
        out = {
            "relu1_1": relu1_1,
            "relu1_2": relu1_2,
            "relu2_1": relu2_1,
            "relu2_2": relu2_2,
            "relu3_1": relu3_1,
            "relu3_2": relu3_2,
            "relu3_3": relu3_3,
            "max_3": max_3,
            "relu4_1": relu4_1,
            "relu4_2": relu4_2,
            "relu4_3": relu4_3,
            "relu5_1": relu5_1,
            "relu5_2": relu5_2,
            "relu5_3": relu5_3,
        }
        return out


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    spectral normalization
    code and idea originally from Takeru Miyato's work 'Spectral Normalization for GAN'
    https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


####################################################################################################
# adversarial loss for different gan mode
####################################################################################################


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'hinge':
            self.loss = nn.ReLU()
        elif gan_mode == 'wgangp':
            self.loss = None
        elif gan_mode == 'wgandiv':
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def __call__(self, prediction, target_is_real, is_disc=False):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            labels = (self.real_label if target_is_real else self.fake_label).expand_as(prediction).type_as(prediction)
            loss = self.loss(prediction, labels)
        elif self.gan_mode in ['hinge', 'wgangp']:
            if is_disc:
                if target_is_real:
                    prediction = -prediction
                if self.gan_mode == 'hinge':
                    loss = self.loss(1 + prediction).mean()
                elif self.gan_mode == 'wgangp':
                    loss = prediction.mean()
            else:
                loss = -prediction.mean()
        elif self.gan_mode in ['wgandiv']:
            loss = prediction.mean()

        return loss


class PD_Loss(nn.Module):

    def __init__(self):
        super(PD_Loss, self).__init__()
        self.criterion = torch.nn.L1Loss()

    def __call__(self, x, y):
        # Compute features
        pd_loss = 0.0
        pd_loss = self.criterion(x, y)
        return pd_loss


class TV_Loss(nn.Module):
    def __init__(self):
        super(TV_Loss, self).__init__()

    def __call__(self, image, mask, method):
        hole_mask = 1 - mask
        b, ch, h, w = hole_mask.shape
        dilation_conv = nn.Conv2d(ch, ch, 3, padding=1, bias=False).to(hole_mask)
        torch.nn.init.constant_(dilation_conv.weight, 1.0)
        with torch.no_grad():
            output_mask = dilation_conv(hole_mask)
        updated_holes = output_mask != 0
        dilated_holes = updated_holes.float()
        colomns_in_Pset = dilated_holes[:, :, :, 1:] * dilated_holes[:, :, :, :-1]
        rows_in_Pset = dilated_holes[:, :, 1:, :] * dilated_holes[:, :, :-1:, :]
        if method == "sum":
            loss = torch.sum(
                torch.abs(colomns_in_Pset * (image[:, :, :, 1:] - image[:, :, :, :-1]))
            ) + torch.sum(
                torch.abs(rows_in_Pset * (image[:, :, :1, :] - image[:, :, -1:, :]))
            )
        else:
            loss = torch.mean(
                torch.abs(colomns_in_Pset * (image[:, :, :, 1:] - image[:, :, :, :-1]))
            ) + torch.mean(
                torch.abs(rows_in_Pset * (image[:, :, :1, :] - image[:, :, -1:, :]))
            )
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.type_as(real_data)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).type_as(real_data),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


def cal_gradient_penalty_div(netD, real_data, fake_data, type='mixed', const_power=6.0, const_kappa=2.0):

    if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
        interpolatesv = real_data
    elif type == 'fake':
        interpolatesv = fake_data
    elif type == 'mixed':
        alpha = torch.rand(real_data.shape[0], 1)
        alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
            *real_data.shape)
        alpha = alpha.type_as(real_data)
        interpolatesv = (1 - alpha) * real_data + alpha * fake_data
    else:
        raise NotImplementedError('{} not implemented'.format(type))
    interpolatesv.requires_grad_(True)
    disc_interpolates = netD(interpolatesv)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                    grad_outputs=torch.ones(disc_interpolates.size()).type_as(real_data),
                                    create_graph=True, retain_graph=True, only_inputs=True)
    gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
    gradients_penalty_div = torch.pow((gradients + 1e-16).norm(2, dim=1), const_power).mean() * const_kappa
    return gradients_penalty_div


####################################################################################################
# neural style transform loss from neural_style_tutorial of pytorch
####################################################################################################


def ContentLoss(input, target):
    target = target.detach()
    loss = F.l1_loss(input, target)
    return loss


def GramMatrix(input):
    s = input.size()
    features = input.view(s[0], s[1], s[2]*s[3])
    features_t = torch.transpose(features, 1, 2)
    G = torch.bmm(features, features_t).div(s[1]*s[2]*s[3])
    return G


def StyleLoss(input, target):
    target = GramMatrix(target).detach()
    input = GramMatrix(input)
    loss = F.l1_loss(input, target)
    return loss


def img_crop(input, size=224):
    input_cropped = F.upsample(input, size=(size, size), mode='bilinear', align_corners=True)
    return input_cropped


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, input):
        return (input-self.mean) / self.std


class get_features(nn.Module):
    def __init__(self, cnn):
        super(get_features, self).__init__()

        vgg = copy.deepcopy(cnn)

        self.conv1 = nn.Sequential(vgg[0], vgg[1], vgg[2], vgg[3], vgg[4])
        self.conv2 = nn.Sequential(vgg[5], vgg[6], vgg[7], vgg[8], vgg[9])
        self.conv3 = nn.Sequential(vgg[10], vgg[11], vgg[12], vgg[13], vgg[14], vgg[15], vgg[16])
        self.conv4 = nn.Sequential(vgg[17], vgg[18], vgg[19], vgg[20], vgg[21], vgg[22], vgg[23])
        self.conv5 = nn.Sequential(vgg[24], vgg[25], vgg[26], vgg[27], vgg[28], vgg[29], vgg[30])

    def forward(self, input, layers):
        input = img_crop(input)
        output = []
        for i in range(1, layers):
            layer = getattr(self, 'conv'+str(i))
            input = layer(input)
            output.append(input)
        return output


class GroupNorm(nn.Module):
    def __init__(self, num_features, num_groups=32, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups
#        assert C % G == 0

        x = x.view(N,G,-1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class FullAttention(nn.Module):
    """
    Layer implements my version of the self-attention module
    it is mostly same as self attention, but generalizes to
    (k x k) convolutions instead of (1 x 1)
    args:
        in_channels: number of input channels
        out_channels: number of output channels
        activation: activation function to be applied (default: lrelu(0.2))
        kernel_size: kernel size for convolution (default: (1 x 1))
        transpose_conv: boolean denoting whether to use convolutions or transpose
                        convolutions
        squeeze_factor: squeeze factor for query and keys (default: 8)
        stride: stride for the convolutions (default: 1)
        padding: padding for the applied convolutions (default: 1)
        bias: whether to apply bias or not (default: True)
    """

    def __init__(self, in_channels, out_channels,
                 activation=None, kernel_size=(3, 3), transpose_conv=False,
                 use_spectral_norm=True, use_batch_norm=True,
                 squeeze_factor=8, stride=1, padding=1, bias=True):
        """ constructor for the layer """

        from torch.nn import Conv2d, Parameter, \
            Softmax, ConvTranspose2d, BatchNorm2d

        # base constructor call
        super().__init__()

        # state of the layer
        self.activation = activation
        self.gamma = Parameter(torch.zeros(1))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.squeezed_channels = in_channels // squeeze_factor
        self.use_batch_norm = use_batch_norm

        # Modules required for computations
        if transpose_conv:
            self.query_conv = ConvTranspose2d(  # query convolution
                in_channels=in_channels,
                out_channels=in_channels // squeeze_factor,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            )

            self.key_conv = ConvTranspose2d(
                in_channels=in_channels,
                out_channels=in_channels // squeeze_factor,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            )

            self.value_conv = ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            )

            self.residual_conv = ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            ) if not use_spectral_norm else SpectralNorm(
                ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias
                )
            )

        else:
            self.query_conv = Conv2d(  # query convolution
                in_channels=in_channels,
                out_channels=in_channels // squeeze_factor,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            )

            self.key_conv = Conv2d(
                in_channels=in_channels,
                out_channels=in_channels // squeeze_factor,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            )

            self.value_conv = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            )

            self.residual_conv = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            ) if not use_spectral_norm else SpectralNorm(
                Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias
                )
            )

        # softmax module for applying attention
        self.softmax = Softmax(dim=-1)
        self.batch_norm = BatchNorm2d(out_channels)

    def forward(self, x):
        """
        forward computations of the layer
        :param x: input feature maps (B x C x H x W)
        :return:
            out: self attention value + input feature (B x O x H x W)
            attention: attention map (B x C x H x W)
        """

        # extract the batch size of the input tensor
        m_batchsize, _, _, _ = x.size()

        # create the query projection
        proj_query = self.query_conv(x).view(
            m_batchsize, self.squeezed_channels, -1).permute(0, 2, 1)  # B x (N) x C

        # create the key projection
        proj_key = self.key_conv(x).view(
            m_batchsize, self.squeezed_channels, -1)  # B x C x (N)

        # calculate the attention maps
        energy = torch.bmm(proj_query, proj_key)  # energy
        attention = self.softmax(energy)  # attention (B x (N) x (N))

        # create the value projection
        proj_value = self.value_conv(x).view(
            m_batchsize, self.out_channels, -1)  # B X C X N

        # calculate the output
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))

        # calculate the residual output
        res_out = self.residual_conv(x)

        out = out.view(m_batchsize, self.out_channels,
                       res_out.shape[-2], res_out.shape[-1])

        attention = attention.view(m_batchsize, -1,
                                   res_out.shape[-2], res_out.shape[-1])

        if self.use_batch_norm:
            res_out = self.batch_norm(res_out)

        if self.activation is not None:
            out = self.activation(out)
            res_out = self.activation(res_out)

        # apply the residual connections
        out = (self.gamma * out) + ((1 - self.gamma) * res_out)
        return out, attention


class Diversityloss(nn.Module):
    def __init__(self):
        super(Diversityloss, self).__init__()
        self.vgg = VGG16().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0, 1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        diversity_loss = 0.0
        diversity_loss += self.weights[4] * self.criterion(
            x_vgg["relu4_1"], y_vgg["relu4_1"]
        )
        return diversity_loss
    
    
class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0 / 2, 1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module("vgg", VGG16().cuda())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(
            x_vgg["relu1_1"], y_vgg["relu1_1"]
        )
        content_loss += self.weights[1] * self.criterion(
            x_vgg["relu2_1"], y_vgg["relu2_1"]
        )
        content_loss += self.weights[2] * self.criterion(
            x_vgg["relu3_1"], y_vgg["relu3_1"]
        )
        content_loss += self.weights[3] * self.criterion(
            x_vgg["relu4_1"], y_vgg["relu4_1"]
        )
        content_loss += self.weights[4] * self.criterion(
            x_vgg["relu5_1"], y_vgg["relu5_1"]
        )
        return content_loss


class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module("vgg", VGG16().cuda())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(
            self.compute_gram(x_vgg["relu2_2"]), self.compute_gram(y_vgg["relu2_2"])
        )
        style_loss += self.criterion(
            self.compute_gram(x_vgg["relu3_3"]), self.compute_gram(y_vgg["relu3_3"])
        )
        style_loss += self.criterion(
            self.compute_gram(x_vgg["relu4_3"]), self.compute_gram(y_vgg["relu4_3"])
        )
        return style_loss
