from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian
from torchvision import models
import numpy as np
from torch.autograd import grad

class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
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

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection", align_corners=True)

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed

    def jacobian(self, coordinates):
        new_coordinates = self.warp_coordinates(coordinates)
        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}

class GeneratorFullModel_kp_texture(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    with additional keypoints supervised 将所有与生成器相关的更新合并到单个模型中，以更好地使用多 GPU，并监督额外的关键点
    """

    def __init__(self, kp_extractor, generator, bg_predictor, discriminator, train_params, device_ids):
        super(GeneratorFullModel_kp_texture, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        # add supervised keypoints
        self.sup_kp_num = train_params['sup_kp_num']

        self.bg_predictor = None
        if bg_predictor:
            self.bg_predictor = bg_predictor
            self.bg_start = train_params['bg_start']


        if torch.cuda.is_available():
            self.pyramid = self.pyramid.to(device_ids[0])

        self.loss_weights = train_params['loss_weights']
        self.dropout_epoch = train_params['dropout_epoch']
        self.dropout_maxp = train_params['dropout_maxp']
        self.dropout_inc_epoch = train_params['dropout_inc_epoch']
        self.dropout_startp =train_params['dropout_startp']

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

    def forward(self, x, epoch):
        kp_source = self.kp_extractor(x['source'])  # value (8, 10, 2), jacobian (8, 10, 2, 2)
        kp_driving = self.kp_extractor(x['driving'])  # value (8, 10, 2), jacobian (8, 10, 2, 2)
        
        bg_param = None
        if self.bg_predictor:
            if(epoch>=self.bg_start):
                bg_param = self.bg_predictor(x['source'], x['driving'])

        if(epoch>=self.dropout_epoch):
            dropout_flag = False
            dropout_p = 0
        else:
            # dropout_p will linearly increase from dropout_startp to dropout_maxp 
            dropout_flag = True
            dropout_p = min(epoch/self.dropout_inc_epoch * self.dropout_maxp + self.dropout_startp, self.dropout_maxp)

        generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving, bg_param = bg_param, 
                                                    dropout_flag = dropout_flag, dropout_p = dropout_p)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

        loss_values = {}

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated_low = self.pyramid(generated['prediction_low'])
        pyramide_generated = self.pyramid(generated['prediction'])

        if sum(self.loss_weights['rec_l1']) != 0:
            value_total = 0
            for i, (scale, weight) in enumerate(zip(self.scales, self.loss_weights['rec_l1'])):
                x_ = pyramide_generated_low['prediction_' + str(scale)]
                y_ = pyramide_real['prediction_' + str(scale)]
                value = torch.abs(x_ - y_.detach()).mean()
                # value = torch.abs(x_[i] - y_[i].detach()).mean()
                value_total += self.loss_weights['rec_l1'][i] * value

                loss_values['rec_l1'] = value_total

        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

        if self.loss_weights['generator_gan'] != 0:
            discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])
            transformed_kp = self.kp_extractor(transformed_frame)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                value = torch.abs(kp_driving['value'][:, :-self.sup_kp_num] - transform.warp_coordinates(transformed_kp['value'][:, :-self.sup_kp_num])).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

            ## jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0:
                jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['value'][:, :-self.sup_kp_num]),
                                                    transformed_kp['jacobian'][:, :-self.sup_kp_num])

                normed_driving = torch.inverse(kp_driving['jacobian'][:, :-self.sup_kp_num])
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_driving, normed_transformed)

                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value
        
        if self.loss_weights['supervised_kp_l2'] != 0:
            kp_s = {}
            kp_d = {}
            kp_s_gt = {}
            kp_d_gt = {}
            # predicted: 
            kp_s['value'] = kp_source['value'][:, -self.sup_kp_num:, :]
            kp_d['value'] = kp_driving['value'][:, -self.sup_kp_num:, :]
            # gt
            kp_d_gt['value'] = x['driving_kp'].float()
            kp_s_gt['value'] = x['source_kp'].float()
            spatial_size = x['source'].shape[2:]
        
            gaussian_source = kp2gaussian(kp_s, spatial_size=spatial_size, kp_variance=1)
            gaussian_driving = kp2gaussian(kp_d, spatial_size=spatial_size, kp_variance=1)
            gaussian_source_gt = kp2gaussian(kp_s_gt, spatial_size=spatial_size, kp_variance=1)
            gaussian_driving_gt = kp2gaussian(kp_d_gt, spatial_size=spatial_size, kp_variance=1)

            # print("gaussian_driving:", gaussian_driving.shape, "gaussian_driving_gt", gaussian_driving_gt.shape)
            value_d = ((torch.abs(gaussian_driving - gaussian_driving_gt))**2).mean()
            value_s = ((torch.abs(gaussian_source - gaussian_source_gt))**2).mean()
            value = value_d + value_s
            loss_values['supervised_kp_l2'] = self.loss_weights['supervised_kp_l2'] * value
        
        # bg loss
        if self.bg_predictor and epoch >= self.bg_start and self.loss_weights['bg'] != 0:
            bg_param_reverse = self.bg_predictor(x['driving'], x['source'])
            value = torch.matmul(bg_param, bg_param_reverse)
            eye = torch.eye(3).view(1, 1, 3, 3).type(value.type())
            value = torch.abs(eye - value).mean()
            loss_values['bg'] = self.loss_weights['bg'] * value

        return loss_values, generated

class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params, device_ids):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.to(device_ids[0])

        self.loss_weights = train_params['loss_weights']
        self.vitloss = nn.CrossEntropyLoss()

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'].detach())

        kp_driving = generated['kp_driving']
        discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
        discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            # LSGAN
            # value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            # GAN
            # real_label = torch.ones(discriminator_maps_generated[key].shape)
            # fake_label = torch.zeros(discriminator_maps_real[key].shape)
            # if torch.cuda.is_available():
            #     real_label = real_label.to(self.device_ids[0])
            #     fake_label = fake_label.to(self.device_ids[0])
            # value = self.criterionGAN(torch.sigmoid(discriminator_maps_real[key]), real_label) + self.criterionGAN(torch.sigmoid(discriminator_maps_generated[key]), fake_label)
            # WGAN
            # value = -torch.mean(torch.sigmoid(discriminator_maps_real[key])) + torch.mean(torch.sigmoid(discriminator_maps_generated[key]))
            value_total += self.loss_weights['discriminator_gan'] * value.mean()
        
        loss_values['disc_gan'] = value_total

        value_total = 0
        key_vit = 'prediction_map_vit'
        input = discriminator_maps_generated[key_vit]
        # target = torch.tensor([[1.,0.],[1.,0.],[1.,0.],[1.,0.]]).cuda()
        #=======================================================# 
        batch_size = input.size(0)  # 或 input.shape[0]
        target = torch.tensor([[1., 0.]] * batch_size, device=input.device)  # RUSI: 改了这里，bs更加通用
        #=======================================================# 
        output_loss = self.vitloss(input, target)
        value_total += self.loss_weights['discriminator_vitgan'] * output_loss
        #真实的loss
        input = discriminator_maps_real[key_vit]
        # target = torch.tensor([[0.,1.],[0.,1.],[0.,1.],[0.,1.]]).cuda()
        #=======================================================# 
        batch_size = input.size(0)  # 或 input.shape[0]
        target = torch.tensor([[0., 1.]] * batch_size, device=input.device)  # RUSI: 改了这里，bs更加通用
        #=======================================================# 
        output_loss_real = self.vitloss(input, target)
        value_total += self.loss_weights['discriminator_vitgan'] * output_loss_real

        loss_values['discriminator_vitgan'] = value_total

        return loss_values
