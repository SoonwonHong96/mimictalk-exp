import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class PerceptualLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.loss_weight = loss_weight

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss * self.loss_weight

class FacialComponentPerceptualLoss(PerceptualLoss):
    def __init__(self, loss_weight=1.0):
        super().__init__(loss_weight)

    def forward(self, x, y, mask):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            mask_i = F.interpolate(mask, size=x_vgg[i].shape[2:], mode='bilinear', align_corners=False)
            loss += self.weights[i] * self.criterion(x_vgg[i] * mask_i, y_vgg[i].detach() * mask_i)
        return loss * self.loss_weight

class GramMatrix(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

class FeatureStyleMatchingLoss(nn.Module):
    def __init__(self, loss_weight=10.0):
        super().__init__()
        self.criterion = nn.L1Loss()
        self.gram = GramMatrix()
        self.loss_weight = loss_weight

    def forward(self, real_feats, fake_feats):
        loss = 0
        for real_feat_pyramid, fake_feat_pyramid in zip(real_feats, fake_feats):
            for real_feat, fake_feat in zip(real_feat_pyramid, fake_feat_pyramid):
                loss += self.criterion(self.gram(real_feat), self.gram(fake_feat))
        return loss * self.loss_weight

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, num_D=3, get_intermediate_features=True):
        super().__init__()
        self.num_D = num_D
        self.get_intermediate_features = get_intermediate_features
        
        for i in range(num_D):
            netD = self._make_net(input_nc, ndf, n_layers, norm_layer)
            setattr(self, 'layer' + str(i), netD)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def _make_net(self, input_nc, ndf, n_layers, norm_layer):
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        return nn.Sequential(*sequence)

    def forward(self, input):
        result = []
        input_downsampled = input
        for i in range(self.num_D):
            net = getattr(self, 'layer' + str(i))
            if self.get_intermediate_features:
                intermediate_result = net(input_downsampled)
                result.append(intermediate_result)
            else:
                result.append(net(input_downsampled))
            
            if i != (self.num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result

    def get_features(self, input):
        """ A helper to extract intermediate features for FSM Loss """
        if not self.get_intermediate_features:
            raise Exception("Set get_intermediate_features=True to use this method.")
        
        result = []
        input_downsampled = input
        for i in range(self.num_D):
            model = getattr(self, 'layer' + str(i))
            # Extract features from all but the last layer of each sub-discriminator
            intermediate_features = []
            x = input_downsampled
            for layer_idx, layer in enumerate(model):
                 x = layer(x)
                 if layer_idx < len(model) - 1: # Don't take the final prediction layer
                    intermediate_features.append(x)
            result.append(intermediate_features)
            
            if i != (self.num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result

class GANLoss(nn.Module):
    def __init__(self, gan_mode='hinge', target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'hinge':
            self.loss = nn.ReLU()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = torch.tensor(self.real_label).to(input.device)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = torch.tensor(self.fake_label).to(input.device)
            return self.fake_label_tensor.expand_as(input)

    def forward(self, input, target_is_real):
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.forward_single(pred_i, target_is_real)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_avg_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_avg_loss
            return loss / len(input)
        else:
            return self.forward_single(input, target_is_real)

    def forward_single(self, input, target_is_real):
        if self.gan_mode == 'hinge':
            if target_is_real:
                return self.loss(1 - input).mean()
            else:
                return self.loss(1 + input).mean()
        else:
            target_tensor = self.get_target_tensor(input, target_is_real)
            return self.loss(input, target_tensor) 