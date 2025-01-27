from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid
from torchvision import models
import numpy as np
from torch.autograd import grad
import pdb
import depth

import cv2
from torchmetrics.regression import MeanAbsolutePercentageError
import sys
sys.path.append("../")
from mp_face_landmarker import PyTorchMediapipeFaceLandmarker
from mp_alignment_differentiable import MPAligner
from hadleigh_utils import compare_to_real_mediapipe


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
        return F.grid_sample(frame, grid, padding_mode="reflection")

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


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params,opt):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()
        self.opt = opt
        self.loss_weights = train_params['loss_weights']

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()
        self.depth_encoder = depth.ResnetEncoder(50, False).cuda()
        self.depth_decoder = depth.DepthDecoder(num_ch_enc=self.depth_encoder.num_ch_enc, scales=range(4)).cuda()
        loaded_dict_enc = torch.load('depth/models/encoder.pth',map_location='cpu')
        loaded_dict_dec = torch.load('depth/models/depth.pth',map_location='cpu')
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.depth_encoder.state_dict()}
        self.depth_encoder.load_state_dict(filtered_dict_enc)
        self.depth_decoder.load_state_dict(loaded_dict_dec)
        self.set_requires_grad(self.depth_encoder, False) 
        self.set_requires_grad(self.depth_decoder, False) 
        self.depth_decoder.eval()
        self.depth_encoder.eval()

        # define MediaPipe face landmarker for attack
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mp = PyTorchMediapipeFaceLandmarker(device, long_range_face_detect=False, short_range_face_detect=False).to(device)
        self.mp_target_features = [(0, 17), (40, 17), (270, 17), (0, 91), (0, 321),
                                 6, 7, 8, 9, 10, 11, 12, 23, 25, 50, 51] 
        self.mp_aligner = MPAligner().to(device)
        self.mape = MeanAbsolutePercentageError()
        self.cos_sim = torch.nn.CosineSimilarity(dim=0)

    def cosine_distance(self, vec1, vec2):
        normed_vec1 = vec1 / torch.norm(vec1)
        normed_vec2 = vec2 / torch.norm(vec2)
        cos_dist = 1 - self.cos_sim(normed_vec1, normed_vec2)
        return cos_dist
           
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def get_mp_bbox(self, coords):
        """
        Get face bounding box coordinates for a frame with frame index based on MediaPipe's extracted landmarks 

        Parameters
        ----------
        coords : list of 2D tuples
            2D facial landmarks
        """
        cx_min = torch.min(coords[:, 0])
        cy_min = torch.min(coords[:, 1])
        cx_max = torch.max(coords[:, 0])
        cy_max = torch.max(coords[:, 1])
        bbox = torch.tensor([[cx_min, cy_min], [cx_max, cy_max]])
        return bbox
    
    def get_mp_features(self, x, gen_vis = False):
        """
        img : torch, RGB, N x 3 x H x W, range 0-1 because float, N is batch size
        """
        mp_feature_values =  torch.zeros(x.shape[0], len(self.mp_target_features)) # list of feature values per frame
        if gen_vis:
            vis_imgs = []
        else:
            vis_imgs = None
        for i in range(x.shape[0]):
            landmarks, blendshapes, padded_face = self.mp(x[i, :, :, :].permute(1, 2, 0) * 255)
            if torch.all(landmarks == 0) and torch.all(blendshapes == 0) and torch.all(padded_face == 0):
                if gen_vis:
                    vis_imgs.append(None)
                continue

            # VIS ONLY #
            if gen_vis:
                padded_face = padded_face.detach().cpu().numpy().astype(np.uint8)
                blendshapes_np = blendshapes.detach().cpu().numpy()
                landmarks_np = landmarks.detach().cpu().numpy()
                vis_img = compare_to_real_mediapipe(landmarks, blendshapes_np, padded_face, save_landmark_comparison=False, display=False, save_path=None)
                vis_imgs.append(vis_img)

            # align
            W, H = torch.tensor(padded_face.shape[1]), torch.tensor(padded_face.shape[0])
            _, landmark_coords_2d_aligned = self.mp_aligner(landmarks, W, H, W, H)

            # compute feature values for the frame
            bbox = self.get_mp_bbox(landmark_coords_2d_aligned)
            bbox_W = bbox[1, 0] - bbox[0, 0]
            bbox_H = bbox[1, 1] - bbox[0, 1]
            for feat_num, feature in enumerate(self.mp_target_features):
                if type(feature) == int: # this is a blendshape
                    mp_feature_values[i, feat_num] = blendshapes[feature]
                else: # this is a facial landmark distance
                    lm1 = landmark_coords_2d_aligned[feature[0]]
                    lm2 = landmark_coords_2d_aligned[feature[1]]
                    x_diff = lm1[0] - lm2[0]
                    x_diff /= bbox_W
                    y_diff = lm1[1] - lm2[1]
                    y_diff /= bbox_H
                    distance = torch.sqrt(x_diff**2 + y_diff**2)
                    mp_feature_values[i, feat_num] = distance

          
        return mp_feature_values, vis_imgs

    def forward(self, x, gen_vis = False):
        
        source_mp_features, source_vis_imgs = self.get_mp_features(x['source'], gen_vis = gen_vis)

        depth_source = None
        depth_driving = None
        outputs = self.depth_decoder(self.depth_encoder(x['source']))
        depth_source = outputs[("disp", 0)]
        outputs = self.depth_decoder(self.depth_encoder(x['driving']))
        depth_driving = outputs[("disp", 0)]
        
        if self.opt.use_depth:
            kp_source = self.kp_extractor(depth_source)
            kp_driving = self.kp_extractor(depth_driving)
        elif self.opt.rgbd:
            source = torch.cat((x['source'],depth_source),1)
            driving = torch.cat((x['driving'],depth_driving),1)
            kp_source = self.kp_extractor(source)
            kp_driving = self.kp_extractor(driving)
        else:
            kp_source = self.kp_extractor(x['source'])
            kp_driving = self.kp_extractor(x['driving'])
        generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving, source_depth = depth_source, driving_depth = depth_driving)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})
        loss_values = {}
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])
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

            discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])
            if self.opt.use_depth:
                outputs = self.depth_decoder(self.depth_encoder(transformed_frame))
                depth_transform = outputs[("disp", 0)]
                transformed_kp = self.kp_extractor(depth_transform)
            elif self.opt.rgbd:
                outputs = self.depth_decoder(self.depth_encoder(transformed_frame))
                depth_transform = outputs[("disp", 0)]
                transform_img = torch.cat((transformed_frame,depth_transform),1)
                transformed_kp = self.kp_extractor(transform_img)
            else:
                transformed_kp = self.kp_extractor(transformed_frame)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                value = torch.abs(kp_driving['value'] - transform.warp_coordinates(transformed_kp['value'])).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

            ## jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0:
                jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['value']),
                                                    transformed_kp['jacobian'])

                normed_driving = torch.inverse(kp_driving['jacobian'])
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_driving, normed_transformed)

                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value


        if self.loss_weights['kp_distance']:
            bz,num_kp,kp_dim = kp_source['value'].shape
            sk = kp_source['value'].unsqueeze(2)-kp_source['value'].unsqueeze(1)
            dk = kp_driving['value'].unsqueeze(2)-kp_driving['value'].unsqueeze(1)
            source_dist_loss = (-torch.sign((torch.sqrt((sk*sk).sum(-1)+1e-8)+torch.eye(num_kp).cuda()*0.2)-0.2)+1).mean()
            driving_dist_loss = (-torch.sign((torch.sqrt((dk*dk).sum(-1)+1e-8)+torch.eye(num_kp).cuda()*0.2)-0.2)+1).mean()
            # driving_dist_loss = (torch.sign(1-(torch.sqrt((dk*dk).sum(-1)+1e-8)+torch.eye(num_kp).cuda()))+1).mean()
            value_total = self.loss_weights['kp_distance']*(source_dist_loss+driving_dist_loss)
            loss_values['kp_distance'] = value_total
        if self.loss_weights['kp_prior']:
            bz,num_kp,kp_dim = kp_source['value'].shape
            sk = kp_source['value'].unsqueeze(2)-kp_source['value'].unsqueeze(1)
            dk = kp_driving['value'].unsqueeze(2)-kp_driving['value'].unsqueeze(1)
            dis_loss = torch.relu(0.1-torch.sqrt((sk*sk).sum(-1)+1e-8))+torch.relu(0.1-torch.sqrt((dk*dk).sum(-1)+1e-8))
            bs,nk,_=kp_source['value'].shape
            scoor_depth = F.grid_sample(depth_source,kp_source['value'].view(bs,1,nk,-1))
            dcoor_depth = F.grid_sample(depth_driving,kp_driving['value'].view(bs,1,nk,-1))
            sd_loss = torch.abs(scoor_depth.mean(-1,keepdim=True) - kp_source['value'].view(bs,1,nk,-1)).mean()
            dd_loss = torch.abs(dcoor_depth.mean(-1,keepdim=True) - kp_driving['value'].view(bs,1,nk,-1)).mean()
            value_total = self.loss_weights['kp_distance']*(dis_loss+sd_loss+dd_loss)
            loss_values['kp_distance'] = value_total


        if self.loss_weights['kp_scale']:
            bz,num_kp,kp_dim = kp_source['value'].shape
            if self.opt.rgbd:
                outputs = self.depth_decoder(self.depth_encoder(generated['prediction']))
                depth_pred = outputs[("disp", 0)]
                pred = torch.cat((generated['prediction'],depth_pred),1)
                kp_pred = self.kp_extractor(pred)
            elif self.opt.use_depth:
                outputs = self.depth_decoder(self.depth_encoder(generated['prediction']))
                depth_pred = outputs[("disp", 0)]
                kp_pred = self.kp_extractor(depth_pred)
            else:
                kp_pred = self.kp_extractor(generated['prediction'])

            pred_mean = kp_pred['value'].mean(1,keepdim=True)
            driving_mean = kp_driving['value'].mean(1,keepdim=True)
            pk = kp_source['value']-pred_mean
            dk = kp_driving['value']- driving_mean
            pred_dist_loss = torch.sqrt((pk*pk).sum(-1)+1e-8)
            driving_dist_loss = torch.sqrt((dk*dk).sum(-1)+1e-8)
            scale_vec = driving_dist_loss/pred_dist_loss
            bz,n = scale_vec.shape
            value = torch.abs(scale_vec[:,:n-1]-scale_vec[:,1:]).mean()
            value_total = self.loss_weights['kp_scale']*value
            loss_values['kp_scale'] = value_total
        if self.loss_weights['depth_constraint']:
            bz,num_kp,kp_dim = kp_source['value'].shape
            outputs = self.depth_decoder(self.depth_encoder(generated['prediction']))
            depth_pred = outputs[("disp", 0)]
            value_total = self.loss_weights['depth_constraint']*torch.abs(depth_driving-depth_pred).mean()
            loss_values['depth_constraint'] = value_total

        gen_mp_features, gen_vis_imgs = self.get_mp_features(generated['prediction'], gen_vis)
        if self.loss_weights["verilight"]:
            # ignore cases where the entire row of features is zero, as this corresponds to a MP extraction failure
            # and we don't want it considered in the loss function
            source_mp_features_ok = (source_mp_features != 0).all(dim=-1).detach()
            gen_mp_features_ok  = (gen_mp_features != 0).all(dim=-1).detach()
            source_mp_features_ok_indices =  source_mp_features_ok.nonzero().squeeze().tolist()
            gen_mp_features_ok_indices = gen_mp_features_ok.nonzero().squeeze().tolist()
            if type(source_mp_features_ok_indices) == int: # squeezing on just one el list yields a scalar, which calling .tolist() yields int
                source_mp_features_ok_indices = [source_mp_features_ok_indices]
            if type(gen_mp_features_ok_indices) == int: 
                gen_mp_features_ok_indices = [gen_mp_features_ok_indices]
            
            ok_indices = list(set(source_mp_features_ok_indices) & set(gen_mp_features_ok_indices))
            source_mp_features = source_mp_features[ok_indices]
            gen_mp_features = gen_mp_features[ok_indices]
    
            # mp_rmse = torch.mean((source_mp_features - gen_mp_features)**2)
            # mp_mape = self.mape(source_mp_features, gen_mp_features)        
            cos_dist = self.cosine_distance(source_mp_features, gen_mp_features)
            value = cos_dist * self.loss_weights["verilight"]
            assert value.grad_fn is not None
            loss_values["verilight"] = value
            
            # vector1 = torch.tensor([1.0, 2.0, 3.0])
            # vector2 = torch.tensor([4.0, 5.0, 6.0])
            # print(self.cosine_distance(vector1, vector2))

            # vector1 = torch.tensor([1.0, 2.0, 3.0])
            # vector2 = torch.tensor([1.0, 2.0, 3.0])
            # print(self.cosine_distance(vector1, vector2))

            # vector1 = torch.tensor([1.0, 2.0, 3.0])
            # vector2 = torch.tensor([2.0, 3.0, 4.0])
            # print(self.cosine_distance(vector1, vector2))

    
        if gen_vis:
            # create source, drive, generated visualization
            generated_vis = []
            mp_vis = []
            for i in range(generated['prediction'].shape[0]):
                gen = generated['prediction'][i,:,:,:].detach().cpu().permute(1, 2, 0).numpy() * 255
                gen = gen.astype(np.uint8)
                source = x['source'][i,:,:,:].detach().cpu().permute(1, 2, 0).numpy() * 255
                source = source.astype(np.uint8)
                driving = x['driving'][i,:,:,:].detach().cpu().permute(1, 2, 0).numpy() * 255
                driving = driving.astype(np.uint8)
                stacked = np.hstack((source, driving, gen))
                stacked = stacked[:,:,::-1]
                generated_vis.append(stacked)
            
                # stack the mediapipe visualizations to created one
                if source_vis_imgs[i] is None:
                    continue
                if gen_vis_imgs[i] is None:
                    continue

                stacked_mp_vis = np.vstack((source_vis_imgs[i], gen_vis_imgs[i]))
                mp_vis.append(stacked_mp_vis)

            generated["generated_vis"] = generated_vis
            generated["mp_vis"] = mp_vis
        else:
            generated["generated_vis"] = None
            generated["mp_vis"] = None

        return loss_values, generated



class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

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
            value_total += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values['disc_gan'] = value_total

        return loss_values
