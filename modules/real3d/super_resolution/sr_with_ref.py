import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from modules.eg3ds.models.networks_stylegan2 import SynthesisBlock
from modules.eg3ds.models.superresolution import SynthesisBlockNoUp

from modules.eg3ds.models.superresolution import SuperresolutionHybrid8XDC
from modules.real3d.facev2v_warp.model import WarpBasedTorsoModelMediaPipe as torso_model_v1
from modules.real3d.facev2v_warp.model2 import WarpBasedTorsoModelMediaPipe as torso_model_v2

from utils.commons.hparams import hparams
from utils.commons.image_utils import dilate, erode


class SuperresolutionHybrid8XDC_Warp(SuperresolutionHybrid8XDC):
    def __init__(self, channels, img_resolution, sr_num_fp16_res, sr_antialias, **block_kwargs):
        super().__init__(channels, img_resolution, sr_num_fp16_res, sr_antialias, **block_kwargs)
        if hparams.get("torso_model_version", "v1") == 'v1':
            self.torso_model = torso_model_v1('standard')
        elif hparams.get("torso_model_version", "v1") == 'v2':
            self.torso_model = torso_model_v2('standard')
        else: raise NotImplementedError()
        # self.torso_model = WarpBasedTorsoModelMediaPipe('small')
        self.torso_encoder = nn.Sequential(*[
            nn.Conv2d(64, 256, 1, 1, padding=0),
        ])
        self.bg_encoder = nn.Sequential(*[
            nn.Conv2d(3, 64, 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 256, 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=1),
        ])

        if hparams.get("weight_fuse", True):
            if hparams['htbsr_head_weight_fuse_mode'] in ['v1']:
                fuse_in_dim = 512
            # elif hparams['htbsr_head_weight_fuse_mode'] in ['v2']:
            else:
                fuse_in_dim = 512
                self.head_torso_alpha_predictor = nn.Sequential(*[
                    nn.Conv2d(3+1+3, 32, 3, 1, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(32, 32, 3, 1, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(32, 1, 3, 1, padding=1),
                    nn.Sigmoid(),
                ])
                self.fuse_head_torso_convs = nn.Sequential(*[
                    nn.Conv2d(256+256, 256, 3, 1, padding=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(256, 256, 3, 1, padding=1),
                ])
                self.head_torso_block = SynthesisBlockNoUp(256, 256, w_dim=512, resolution=256,
                    img_channels=3, is_last=False, use_fp16=False, conv_clamp=None, **block_kwargs)
        else:
            fuse_in_dim = 768
        self.fuse_fg_bg_convs = nn.Sequential(*[
            nn.Conv2d(fuse_in_dim, 64, 1, 1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 256, 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=1),
        ])

        self.mouth_encode_mode = hparams.get('mouth_encode_mode', 'none')
        if self.mouth_encode_mode == 'concat':
            # Full-resolution encoder for concatenation
            self.mouth_encoder_concat = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            )
            self.fuse_mouth_conv = nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1)
        
        if self.mouth_encode_mode == 'add':
            # Use a full-resolution encoder, same as 'concat', to avoid interpolation.
            self.mouth_encoder_add = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            )

        if self.mouth_encode_mode == 'style_latent':
            w_dim = hparams.get('w_dim', 512)
            self.mouth_encoder_style = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, w_dim)
            )
            self.style_fusion_layer = nn.Linear(w_dim + w_dim, w_dim)

        if self.mouth_encode_mode == 'adain':
            num_channels = hparams.get('renderer_out_channels', 256)
            self.mouth_encoder_adain = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, 512) # Intermediate latent
            )
            # This layer generates gamma (scale) and beta (bias) for AdaIN
            self.adain_param_generator = nn.Linear(512, num_channels * 2)

        if self.mouth_encode_mode == 'gated':
            num_channels = hparams.get('renderer_out_channels', 256)
            self.mouth_encoder_gated = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, num_channels, kernel_size=3, stride=1, padding=1),
            )
            self.gating_network = nn.Sequential(
                nn.Conv2d(num_channels + num_channels, 128, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 1, kernel_size=1, padding=0),
                nn.Sigmoid()
            )

        if self.mouth_encode_mode == 'film':
            num_channels = hparams.get('renderer_out_channels', 256)
            self.mouth_encoder_film = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, 256)
            )
            # This layer generates gamma (scale) and beta (shift) for FiLM
            self.film_generator = nn.Linear(256, num_channels * 2)

    def initialize_gating_bias(self, mouth_mask):
        """
        Initializes the bias of the final layer of the gating network.
        This encourages the network's initial output to resemble the provided mouth mask,
        giving it a strong prior for where the gate should be active.
        """
        if self.mouth_encode_mode != 'gated':
            print("| Gating bias initialization skipped: mouth_encode_mode is not 'gated'.")
            return

        print("| Initializing gating network bias...")
        # We need to find the bias 'b' such that sigmoid(x + b) is close to our mask.
        # The inverse of sigmoid is the logit function: logit(p) = log(p / (1 - p)).
        # We can treat the average mask values as the desired initial probabilities.
        
        # Ensure mask is on the correct device
        mouth_mask = mouth_mask.to(self.gating_network[-2].weight.device)
        
        # Calculate the mean of the mask for the batch
        # Add a small epsilon to avoid log(0) for pixels where mask is 0
        epsilon = 1e-4
        avg_mask = torch.mean(mouth_mask, dim=0, keepdim=True)
        avg_mask = torch.clamp(avg_mask, epsilon, 1 - epsilon)
        
        # Calculate the logit of the average mask
        initial_bias = torch.log(avg_mask / (1 - avg_mask))

        # The gating network's final conv layer is at index -2
        # (because the last element, -1, is the Sigmoid activation)
        with torch.no_grad():
            self.gating_network[-2].bias.data.copy_(initial_bias.squeeze(0))
        
        print("| Gating network bias initialized successfully.")

    def forward(self, rgb, x, ws, ref_torso_rgb, ref_bg_rgb, weights_img, segmap, kp_s, kp_d, target_torso_mask=None, mouth_ref_img=None, **block_kwargs):
        weights_img = weights_img.detach()
        ws = ws[:, -1:, :].expand([rgb.shape[0], 3, -1])
        
        if x.shape[-1] != self.input_resolution:
            x = torch.nn.functional.interpolate(x, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)
            rgb = torch.nn.functional.interpolate(rgb, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)
        
        rgb_256 = torch.nn.functional.interpolate(rgb, size=(256, 256), mode='bilinear', align_corners=False, antialias=self.sr_antialias)
        weights_256 = torch.nn.functional.interpolate(weights_img, size=(256, 256), mode='bilinear', align_corners=False, antialias=self.sr_antialias)
        
        ref_torso_rgb_256 = torch.nn.functional.interpolate(ref_torso_rgb, size=(256, 256), mode='bilinear', align_corners=False, antialias=self.sr_antialias)
        
        ref_bg_rgb_256 = torch.nn.functional.interpolate(ref_bg_rgb, size=(256, 256), mode='bilinear', align_corners=False, antialias=self.sr_antialias)
        x, rgb = self.block0(x, rgb, ws, **block_kwargs) # sr branch, 128x128 head img ==> 256x256 head img
        if hparams.get("torso_model_version", "v1") == 'v1':
            rgb_torso, facev2v_ret = self.torso_model.forward(ref_torso_rgb_256, segmap, kp_s, kp_d, rgb_256.detach(), cal_loss=True, target_torso_mask=target_torso_mask)
        elif hparams.get("torso_model_version", "v1") == 'v2':
            rgb_torso, facev2v_ret = self.torso_model.forward(ref_torso_rgb_256, segmap, kp_s, kp_d, rgb_256.detach(), weights_256.detach(), cal_loss=True, target_torso_mask=target_torso_mask)
        x_torso = self.torso_encoder(facev2v_ret['deformed_torso_hid'])

        x_bg = self.bg_encoder(ref_bg_rgb_256)
        
        if hparams.get("weight_fuse", True):
            if hparams['htbsr_head_weight_fuse_mode'] == 'v1':
                rgb = rgb * weights_256 + rgb_torso * (1-weights_256) # get person img
                x = x * weights_256 + x_torso * (1-weights_256) # get person img
                head_occlusion = weights_256.clone()    
                htbsr_head_threshold = hparams['htbsr_head_threshold']
                head_occlusion[head_occlusion > htbsr_head_threshold] = 1.
                torso_occlusion = torch.nn.functional.interpolate(facev2v_ret['occlusion_2'], size=(256, 256), mode='bilinear', align_corners=False, antialias=self.sr_antialias)
                person_occlusion = (torso_occlusion + head_occlusion).clamp_(0,1)
                rgb = rgb * person_occlusion + ref_bg_rgb_256 * (1-person_occlusion) # run6
                x = torch.cat([x * person_occlusion, x_bg * (1-person_occlusion)], dim=1) # run6
                x = self.fuse_fg_bg_convs(x)

                x_for_block1 = x
                ws_for_block1 = ws
                if self.mouth_encode_mode != 'none' and mouth_ref_img is not None:
                    mouth_ref_img_256 = F.interpolate(mouth_ref_img, size=(256, 256), mode='bilinear', align_corners=False, antialias=True)
                    if self.mouth_encode_mode == 'concat':
                        mouth_feat = self.mouth_encoder_concat(mouth_ref_img_256)
                        x_for_block1 = self.fuse_mouth_conv(torch.cat([x, mouth_feat], dim=1))
                    elif self.mouth_encode_mode == 'add':
                        mouth_feat = self.mouth_encoder_add(mouth_ref_img_256)
                        x_for_block1 = x + mouth_feat
                    elif self.mouth_encode_mode == 'style_latent':
                        mouth_style_vector = self.mouth_encoder_style(mouth_ref_img_256)
                        original_style_vector = ws[:, 0, :]
                        fused_style = self.style_fusion_layer(torch.cat([original_style_vector, mouth_style_vector], dim=-1))
                        ws_for_block1 = fused_style.unsqueeze(1).expand(-1, 3, -1)
                    elif self.mouth_encode_mode == 'adain':
                        # Normalize the content features (x)
                        normalized_x = F.instance_norm(x)
                        # Generate AdaIN parameters from mouth reference
                        mouth_embedding = self.mouth_encoder_adain(mouth_ref_img_256)
                        adain_params = self.adain_param_generator(mouth_embedding)
                        gamma, beta = torch.chunk(adain_params, 2, dim=1)
                        # Reshape for broadcasting
                        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                        beta = beta.unsqueeze(-1).unsqueeze(-1)
                        # Apply the new style
                        x_for_block1 = gamma * normalized_x + beta
                    elif self.mouth_encode_mode == 'gated':
                        mouth_feat = self.mouth_encoder_gated(mouth_ref_img_256)
                        gate_input = torch.cat([x, mouth_feat], dim=1)
                        alpha_gate = self.gating_network(gate_input)
                        x_for_block1 = (1 - alpha_gate) * x + alpha_gate * mouth_feat
                    elif self.mouth_encode_mode == 'film':
                        mouth_embedding = self.mouth_encoder_film(mouth_ref_img_256)
                        film_params = self.film_generator(mouth_embedding)
                        gamma, beta = torch.chunk(film_params, 2, dim=1)
                        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                        beta = beta.unsqueeze(-1).unsqueeze(-1)
                        x_for_block1 = (1 + gamma) * x + beta

                x, rgb = self.block1(x_for_block1, rgb, ws_for_block1, **block_kwargs)

            elif hparams['htbsr_head_weight_fuse_mode'] == 'v2':
                # 用alpha-cat实现head torso的x的融合；替代了之前的直接alpha相加
                head_torso_alpha = weights_256.clone()
                head_torso_alpha[head_torso_alpha>weights_256] = weights_256[head_torso_alpha>weights_256]
                rgb = rgb * head_torso_alpha + rgb_torso * (1-head_torso_alpha) # get person img
                x = torch.cat([x * head_torso_alpha, x_torso * (1-head_torso_alpha)], dim=1) 
                x = self.fuse_head_torso_convs(x)
                x, rgb = self.head_torso_block(x, rgb, ws, **block_kwargs)

                head_occlusion = head_torso_alpha.clone()    
                # 鼓励weights与mask逼近后，不再需要手动修改head weights threshold到很小的值了，0.7就行
                htbsr_head_threshold = hparams['htbsr_head_threshold']
                head_occlusion[head_occlusion > htbsr_head_threshold] = 1.
                torso_occlusion = torch.nn.functional.interpolate(facev2v_ret['occlusion_2'], size=(256, 256), mode='bilinear', align_corners=False, antialias=self.sr_antialias)
                person_occlusion = (torso_occlusion + head_occlusion).clamp_(0,1)
                rgb = rgb * person_occlusion + ref_bg_rgb_256 * (1-person_occlusion) # run6
                x = torch.cat([x * person_occlusion, x_bg * (1-person_occlusion)], dim=1) # run6
                x = self.fuse_fg_bg_convs(x)

                x_for_block1 = x
                ws_for_block1 = ws
                if self.mouth_encode_mode != 'none' and mouth_ref_img is not None:
                    mouth_ref_img_256 = F.interpolate(mouth_ref_img, size=(256, 256), mode='bilinear', align_corners=False, antialias=True)
                    if self.mouth_encode_mode == 'concat':
                        mouth_feat = self.mouth_encoder_concat(mouth_ref_img_256)
                        x_for_block1 = self.fuse_mouth_conv(torch.cat([x, mouth_feat], dim=1))
                    elif self.mouth_encode_mode == 'add':
                        mouth_feat = self.mouth_encoder_add(mouth_ref_img_256)
                        x_for_block1 = x + mouth_feat
                    elif self.mouth_encode_mode == 'style_latent':
                        mouth_style_vector = self.mouth_encoder_style(mouth_ref_img_256)
                        original_style_vector = ws[:, 0, :]
                        fused_style = self.style_fusion_layer(torch.cat([original_style_vector, mouth_style_vector], dim=-1))
                        ws_for_block1 = fused_style.unsqueeze(1).expand(-1, 3, -1)
                    elif self.mouth_encode_mode == 'adain':
                        # Normalize the content features (x)
                        normalized_x = F.instance_norm(x)
                        # Generate AdaIN parameters from mouth reference
                        mouth_embedding = self.mouth_encoder_adain(mouth_ref_img_256)
                        adain_params = self.adain_param_generator(mouth_embedding)
                        gamma, beta = torch.chunk(adain_params, 2, dim=1)
                        # Reshape for broadcasting
                        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                        beta = beta.unsqueeze(-1).unsqueeze(-1)
                        # Apply the new style
                        x_for_block1 = gamma * normalized_x + beta
                    elif self.mouth_encode_mode == 'gated':
                        mouth_feat = self.mouth_encoder_gated(mouth_ref_img_256)
                        gate_input = torch.cat([x, mouth_feat], dim=1)
                        alpha_gate = self.gating_network(gate_input)
                        x_for_block1 = (1 - alpha_gate) * x + alpha_gate * mouth_feat
                    elif self.mouth_encode_mode == 'film':
                        mouth_embedding = self.mouth_encoder_film(mouth_ref_img_256)
                        film_params = self.film_generator(mouth_embedding)
                        gamma, beta = torch.chunk(film_params, 2, dim=1)
                        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                        beta = beta.unsqueeze(-1).unsqueeze(-1)
                        x_for_block1 = (1 + gamma) * x + beta

                x, rgb = self.block1(x_for_block1, rgb, ws_for_block1, **block_kwargs)

            elif hparams['htbsr_head_weight_fuse_mode'] == 'v3':
                # v2：用alpha-cat实现head torso的x的融合；替代了之前的直接alpha相加
                # v3: 用nn额外后处理head mask
                head_torso_alpha_inp = torch.cat([rgb.clamp(-1,1)/2+0.5, weights_256, rgb_torso.clamp(-1,1)/2+0.5], dim=1)
                head_torso_alpha_ = self.head_torso_alpha_predictor(head_torso_alpha_inp)
                head_torso_alpha = head_torso_alpha_.clone()
                head_torso_alpha[head_torso_alpha>weights_256] = weights_256[head_torso_alpha>weights_256]
                rgb = rgb * head_torso_alpha + rgb_torso * (1-head_torso_alpha) # get person img
                
                x = torch.cat([x * head_torso_alpha, x_torso * (1-head_torso_alpha)], dim=1) # run6
                x = self.fuse_head_torso_convs(x)
                x, rgb = self.head_torso_block(x, rgb, ws, **block_kwargs)

                head_occlusion = head_torso_alpha.clone()    
                htbsr_head_threshold = hparams['htbsr_head_threshold']
                if not self.training:
                    head_occlusion_ = head_occlusion[head_occlusion>0.05]
                    htbsr_head_threshold = max(head_occlusion_.quantile(0.05), htbsr_head_threshold) # 过滤掉比0.05大的最后5% voxels
                head_occlusion[head_occlusion > htbsr_head_threshold] = 1.
                torso_occlusion = torch.nn.functional.interpolate(facev2v_ret['occlusion_2'], size=(256, 256), mode='bilinear', align_corners=False, antialias=self.sr_antialias)
                person_occlusion = (torso_occlusion + head_occlusion).clamp_(0,1)
                rgb = rgb * person_occlusion + ref_bg_rgb_256 * (1-person_occlusion) # run6
                # Todo: 修改这里，把cat的occlusion去掉？或者把occlusion截断一下。
                x = torch.cat([x * person_occlusion, x_bg * (1-person_occlusion)], dim=1) # run6
                x = self.fuse_fg_bg_convs(x)

                x_for_block1 = x
                ws_for_block1 = ws
                if self.mouth_encode_mode != 'none' and mouth_ref_img is not None:
                    mouth_ref_img_256 = F.interpolate(mouth_ref_img, size=(256, 256), mode='bilinear', align_corners=False, antialias=True)
                    if self.mouth_encode_mode == 'concat':
                        mouth_feat = self.mouth_encoder_concat(mouth_ref_img_256)
                        x_for_block1 = self.fuse_mouth_conv(torch.cat([x, mouth_feat], dim=1))
                    elif self.mouth_encode_mode == 'add':
                        mouth_feat = self.mouth_encoder_add(mouth_ref_img_256)
                        x_for_block1 = x + mouth_feat
                    elif self.mouth_encode_mode == 'style_latent':
                        mouth_style_vector = self.mouth_encoder_style(mouth_ref_img_256)
                        original_style_vector = ws[:, 0, :]
                        fused_style = self.style_fusion_layer(torch.cat([original_style_vector, mouth_style_vector], dim=-1))
                        ws_for_block1 = fused_style.unsqueeze(1).expand(-1, 3, -1)
                    elif self.mouth_encode_mode == 'adain':
                        # Normalize the content features (x)
                        normalized_x = F.instance_norm(x)
                        # Generate AdaIN parameters from mouth reference
                        mouth_embedding = self.mouth_encoder_adain(mouth_ref_img_256)
                        adain_params = self.adain_param_generator(mouth_embedding)
                        gamma, beta = torch.chunk(adain_params, 2, dim=1)
                        # Reshape for broadcasting
                        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                        beta = beta.unsqueeze(-1).unsqueeze(-1)
                        # Apply the new style
                        x_for_block1 = gamma * normalized_x + beta
                    elif self.mouth_encode_mode == 'gated':
                        mouth_feat = self.mouth_encoder_gated(mouth_ref_img_256)
                        gate_input = torch.cat([x, mouth_feat], dim=1)
                        alpha_gate = self.gating_network(gate_input)
                        x_for_block1 = (1 - alpha_gate) * x + alpha_gate * mouth_feat
                    elif self.mouth_encode_mode == 'film':
                        mouth_embedding = self.mouth_encoder_film(mouth_ref_img_256)
                        film_params = self.film_generator(mouth_embedding)
                        gamma, beta = torch.chunk(film_params, 2, dim=1)
                        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                        beta = beta.unsqueeze(-1).unsqueeze(-1)
                        x_for_block1 = (1 + gamma) * x + beta

                x, rgb = self.block1(x_for_block1, rgb, ws_for_block1, **block_kwargs)

            else:
                # v4 尝试直接用cat处理head-torso的hid的融合，发现不好
                # v5 try1处理x的时候也把cat里的alpha去掉了，但是try1发现导致occlusion直接变1.所以去掉
                # v5 try2给torso也加了threshold让他算rgb的时候更加sharp,  会导致torso周围黑边？
                raise NotImplementedError()
        else:
            x = torch.cat([x, x_torso, x_bg], dim=1) # run6
            x = self.fuse_fg_bg_convs(x)
            
            x_for_block1 = x
            ws_for_block1 = ws
            if self.mouth_encode_mode != 'none' and mouth_ref_img is not None:
                mouth_ref_img_256 = F.interpolate(mouth_ref_img, size=(256, 256), mode='bilinear', align_corners=False, antialias=True)
                if self.mouth_encode_mode == 'concat':
                    mouth_feat = self.mouth_encoder_concat(mouth_ref_img_256)
                    x_for_block1 = self.fuse_mouth_conv(torch.cat([x, mouth_feat], dim=1))
                elif self.mouth_encode_mode == 'add':
                    mouth_feat = self.mouth_encoder_add(mouth_ref_img_256)
                    x_for_block1 = x + mouth_feat
                elif self.mouth_encode_mode == 'style_latent':
                    mouth_style_vector = self.mouth_encoder_style(mouth_ref_img_256)
                    original_style_vector = ws[:, 0, :]
                    fused_style = self.style_fusion_layer(torch.cat([original_style_vector, mouth_style_vector], dim=-1))
                    ws_for_block1 = fused_style.unsqueeze(1).expand(-1, 3, -1)
                elif self.mouth_encode_mode == 'adain':
                    # Normalize the content features (x)
                    normalized_x = F.instance_norm(x)
                    # Generate AdaIN parameters from mouth reference
                    mouth_embedding = self.mouth_encoder_adain(mouth_ref_img_256)
                    adain_params = self.adain_param_generator(mouth_embedding)
                    gamma, beta = torch.chunk(adain_params, 2, dim=1)
                    # Reshape for broadcasting
                    gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                    beta = beta.unsqueeze(-1).unsqueeze(-1)
                    # Apply the new style
                    x_for_block1 = gamma * normalized_x + beta
                elif self.mouth_encode_mode == 'gated':
                    mouth_feat = self.mouth_encoder_gated(mouth_ref_img_256)
                    gate_input = torch.cat([x, mouth_feat], dim=1)
                    alpha_gate = self.gating_network(gate_input)
                    x_for_block1 = (1 - alpha_gate) * x + alpha_gate * mouth_feat
                elif self.mouth_encode_mode == 'film':
                    mouth_embedding = self.mouth_encoder_film(mouth_ref_img_256)
                    film_params = self.film_generator(mouth_embedding)
                    gamma, beta = torch.chunk(film_params, 2, dim=1)
                    gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                    beta = beta.unsqueeze(-1).unsqueeze(-1)
                    x_for_block1 = (1 + gamma) * x + beta

                x, rgb = self.block1(x_for_block1, rgb, ws_for_block1, **block_kwargs)
        return rgb, facev2v_ret
   
    @torch.no_grad()
    def infer_forward_stage1(self, rgb, x, ws, ref_torso_rgb, ref_bg_rgb, weights_img, segmap, kp_s, kp_d, mouth_ref_img=None, **block_kwargs):
        weights_img = weights_img.detach()
        ws = ws[:, -1:, :].repeat(1, 3, 1)

        if x.shape[-1] != self.input_resolution:
            x = torch.nn.functional.interpolate(x, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)
            rgb = torch.nn.functional.interpolate(rgb, size=(self.input_resolution, self.input_resolution),
                                                  mode='bilinear', align_corners=False, antialias=self.sr_antialias)
        
        rgb_256 = torch.nn.functional.interpolate(rgb, size=(256, 256), mode='bilinear', align_corners=False, antialias=self.sr_antialias)
        weights_256 = torch.nn.functional.interpolate(weights_img, size=(256, 256), mode='bilinear', align_corners=False, antialias=self.sr_antialias)
        ref_torso_rgb_256 = torch.nn.functional.interpolate(ref_torso_rgb, size=(256, 256), mode='bilinear', align_corners=False, antialias=self.sr_antialias)
        ref_bg_rgb_256 = torch.nn.functional.interpolate(ref_bg_rgb, size=(256, 256), mode='bilinear', align_corners=False, antialias=self.sr_antialias)

        x, rgb = self.block0(x, rgb, ws, **block_kwargs)

        facev2v_ret = self.torso_model.infer_forward_stage1(ref_torso_rgb_256, segmap, kp_s, kp_d, rgb_256.detach(), cal_loss=True)
        facev2v_ret['ref_bg_rgb_256'] = ref_bg_rgb_256
        facev2v_ret['weights_256'] = weights_256
        facev2v_ret['x'] = x
        facev2v_ret['ws'] = ws
        facev2v_ret['rgb'] = rgb
        if mouth_ref_img is not None:
            facev2v_ret['mouth_ref_img'] = mouth_ref_img
        return facev2v_ret
    
    @torch.no_grad()
    def infer_forward_stage2(self, facev2v_ret, **block_kwargs):
        x = facev2v_ret['x']
        ws = facev2v_ret['ws']
        rgb = facev2v_ret['rgb']
        ref_bg_rgb_256 = facev2v_ret['ref_bg_rgb_256']
        weights_256 = facev2v_ret['weights_256']
        rgb_torso = self.torso_model.infer_forward_stage2(facev2v_ret)
        x_torso = self.torso_encoder(facev2v_ret['deformed_torso_hid'])
        x_bg = self.bg_encoder(ref_bg_rgb_256)
        
        if hparams.get("weight_fuse", True):
            rgb = rgb * weights_256 + rgb_torso * (1-weights_256) # get person img
            x = x * weights_256 + x_torso * (1-weights_256) # get person img

            head_occlusion = weights_256.clone()
            head_occlusion[head_occlusion > 0.5] = 1.
            torso_occlusion = torch.nn.functional.interpolate(facev2v_ret['occlusion_2'], size=(256, 256), mode='bilinear', align_corners=False, antialias=self.sr_antialias)
            person_occlusion = (torso_occlusion + head_occlusion).clamp_(0,1)

            rgb = rgb * person_occlusion + ref_bg_rgb_256 * (1-person_occlusion) # run6
            x = torch.cat([x * person_occlusion, x_bg * (1-person_occlusion)], dim=1) # run6
            x = self.fuse_fg_bg_convs(x)

            x_for_block1 = x
            ws_for_block1 = ws
            if self.mouth_encode_mode != 'none' and facev2v_ret.get('mouth_ref_img') is not None:
                mouth_ref_img = facev2v_ret['mouth_ref_img']
                mouth_ref_img_256 = F.interpolate(mouth_ref_img, size=(256, 256), mode='bilinear', align_corners=False, antialias=True)
                if self.mouth_encode_mode == 'concat':
                    mouth_feat = self.mouth_encoder_concat(mouth_ref_img_256)
                    x_for_block1 = self.fuse_mouth_conv(torch.cat([x, mouth_feat], dim=1))
                elif self.mouth_encode_mode == 'add':
                    mouth_feat = self.mouth_encoder_add(mouth_ref_img_256)
                    x_for_block1 = x + mouth_feat
                elif self.mouth_encode_mode == 'style_latent':
                    mouth_style_vector = self.mouth_encoder_style(mouth_ref_img_256)
                    original_style_vector = ws[:, 0, :]
                    fused_style = self.style_fusion_layer(torch.cat([original_style_vector, mouth_style_vector], dim=-1))
                    ws_for_block1 = fused_style.unsqueeze(1).expand(-1, 3, -1)
                elif self.mouth_encode_mode == 'adain':
                    # Normalize the content features (x)
                    normalized_x = F.instance_norm(x)
                    # Generate AdaIN parameters from mouth reference
                    mouth_embedding = self.mouth_encoder_adain(mouth_ref_img_256)
                    adain_params = self.adain_param_generator(mouth_embedding)
                    gamma, beta = torch.chunk(adain_params, 2, dim=1)
                    # Reshape for broadcasting
                    gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                    beta = beta.unsqueeze(-1).unsqueeze(-1)
                    # Apply the new style
                    x_for_block1 = gamma * normalized_x + beta
                elif self.mouth_encode_mode == 'gated':
                    mouth_feat = self.mouth_encoder_gated(mouth_ref_img_256)
                    gate_input = torch.cat([x, mouth_feat], dim=1)
                    alpha_gate = self.gating_network(gate_input)
                    x_for_block1 = (1 - alpha_gate) * x + alpha_gate * mouth_feat
                elif self.mouth_encode_mode == 'film':
                    mouth_embedding = self.mouth_encoder_film(mouth_ref_img_256)
                    film_params = self.film_generator(mouth_embedding)
                    gamma, beta = torch.chunk(film_params, 2, dim=1)
                    gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                    beta = beta.unsqueeze(-1).unsqueeze(-1)
                    x_for_block1 = (1 + gamma) * x + beta

                x, rgb = self.block1(x_for_block1, rgb, ws_for_block1, **block_kwargs)
        else:
            x = torch.cat([x, x_torso, x_bg], dim=1) # run6
            x = self.fuse_fg_bg_convs(x)
            
            x_for_block1 = x
            ws_for_block1 = ws
            if self.mouth_encode_mode != 'none' and facev2v_ret.get('mouth_ref_img') is not None:
                mouth_ref_img = facev2v_ret['mouth_ref_img']
                mouth_ref_img_256 = F.interpolate(mouth_ref_img, size=(256, 256), mode='bilinear', align_corners=False, antialias=True)
                if self.mouth_encode_mode == 'concat':
                    mouth_feat = self.mouth_encoder_concat(mouth_ref_img_256)
                    x_for_block1 = self.fuse_mouth_conv(torch.cat([x, mouth_feat], dim=1))
                elif self.mouth_encode_mode == 'add':
                    mouth_feat = self.mouth_encoder_add(mouth_ref_img_256)
                    x_for_block1 = x + mouth_feat
                elif self.mouth_encode_mode == 'style_latent':
                    mouth_style_vector = self.mouth_encoder_style(mouth_ref_img_256)
                    original_style_vector = ws[:, 0, :]
                    fused_style = self.style_fusion_layer(torch.cat([original_style_vector, mouth_style_vector], dim=-1))
                    ws_for_block1 = fused_style.unsqueeze(1).expand(-1, 3, -1)
                elif self.mouth_encode_mode == 'adain':
                    # Normalize the content features (x)
                    normalized_x = F.instance_norm(x)
                    # Generate AdaIN parameters from mouth reference
                    mouth_embedding = self.mouth_encoder_adain(mouth_ref_img_256)
                    adain_params = self.adain_param_generator(mouth_embedding)
                    gamma, beta = torch.chunk(adain_params, 2, dim=1)
                    # Reshape for broadcasting
                    gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                    beta = beta.unsqueeze(-1).unsqueeze(-1)
                    # Apply the new style
                    x_for_block1 = gamma * normalized_x + beta
                elif self.mouth_encode_mode == 'gated':
                    mouth_feat = self.mouth_encoder_gated(mouth_ref_img_256)
                    gate_input = torch.cat([x, mouth_feat], dim=1)
                    alpha_gate = self.gating_network(gate_input)
                    x_for_block1 = (1 - alpha_gate) * x + alpha_gate * mouth_feat
                elif self.mouth_encode_mode == 'film':
                    mouth_embedding = self.mouth_encoder_film(mouth_ref_img_256)
                    film_params = self.film_generator(mouth_embedding)
                    gamma, beta = torch.chunk(film_params, 2, dim=1)
                    gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                    beta = beta.unsqueeze(-1).unsqueeze(-1)
                    x_for_block1 = (1 + gamma) * x + beta

                x, rgb = self.block1(x_for_block1, rgb, ws_for_block1, **block_kwargs)
        return rgb, facev2v_ret
   

if __name__ == '__main__':
    model = SuperresolutionHybrid8XDC_Warp(32,512,0, False)
    model.cuda()
    rgb = torch.randn([4, 3, 128, 128]).cuda()
    x = torch.randn([4, 32, 128, 128]).cuda()
    ws = torch.randn([4, 14, 512]).cuda()
    ref_rgb = torch.randn([4, 3, 128, 128]).cuda()
    ref_torso_rgb = torch.randn([4, 3, 128, 128]).cuda()
    y = model(rgb, x, ws, ref_rgb, ref_torso_rgb)
    print(" ")