"""
将One-shot的说话人大模型(os_secc2plane or os_secc2plane_torso)在单一说话人(一张照片或一段视频)上overfit, 实现和GeneFace++类似的效果
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import random
import time
import numpy as np
import importlib
import tqdm
import copy
import cv2
import glob
import imageio
import subprocess
import shutil
# common utils
from utils.commons.hparams import hparams, set_hparams
from utils.commons.tensor_utils import move_to_cuda, convert_to_tensor
from utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint
# 3DMM-related utils
from deep_3drecon.deep_3drecon_models.bfm import ParametricFaceModel
from data_util.face3d_helper import Face3DHelper
from data_gen.utils.process_image.fit_3dmm_landmark import fit_3dmm_for_a_image
from data_gen.utils.process_video.fit_3dmm_landmark import fit_3dmm_for_a_video
from data_gen.utils.process_video.extract_segment_imgs import decode_segmap_mask_from_image
from deep_3drecon.secc_renderer import SECC_Renderer
from data_gen.eg3d.convert_to_eg3d_convention import get_eg3d_convention_camera_pose_intrinsic
from data_gen.runs.binarizer_nerf import get_lip_rect, get_eye_rects, create_component_masks_from_landmarks, get_lip_rect_from_mp
# Face Parsing 
from data_gen.utils.mp_feature_extractors.mp_segmenter import MediapipeSegmenter
from data_gen.utils.process_video.extract_segment_imgs import inpaint_torso_job, extract_background
# other inference utils
from inference.infer_utils import mirror_index, load_img_to_512_hwc_array, load_img_to_normalized_512_bchw_tensor
from inference.infer_utils import smooth_camera_sequence, smooth_features_xd
from inference.edit_secc import blink_eye_for_secc, hold_eye_opened_for_secc
from modules.commons.loralib.utils import mark_only_lora_as_trainable
from utils.nn.model_utils import num_params
import lpips
from utils.commons.meters import AvgrageMeter
meter = AvgrageMeter()
from torch.utils.tensorboard import SummaryWriter
# New imports for advanced losses
from modules.losses.gan_loss import MultiScaleDiscriminator, PerceptualLoss, FacialComponentPerceptualLoss, FeatureStyleMatchingLoss, GANLoss
from modules.losses.sd_loss import SDPerceptualLoss

def crop_video_to_16by9(input_file: str, output_file: str):
    """
    Check the resolution of the original video and if it's not 16:9 ratio,
    crop it to 16:9 based on the center.
    If it's already 16:9, copy the original video as is.
    """
    # Check video resolution with ffprobe
    ffprobe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        input_file
    ]
    try:
        ffprobe_output = subprocess.check_output(ffprobe_cmd).decode().strip()
        width_str, height_str = ffprobe_output.split('x')
        width = int(width_str)
        height = int(height_str)
    except Exception as e:
        print(f"Error reading video dimensions: {e}. Copying file without cropping.")
        # if ffprobe fails, just copy
        shutil.copy(input_file, output_file)
        return

    current_ratio = width / height
    desired_ratio = 16 / 9

    # If already 16:9, just copy
    if abs(current_ratio - desired_ratio) < 1e-3:
        print("Video is already 16:9 ratio. Skipping crop.")
        shutil.copy(input_file, output_file)
        return

    # Calculate crop area for 16:9 ratio (ensure even pixels)
    if current_ratio > desired_ratio:
        new_width = int(height * desired_ratio)
        new_width = (new_width // 2) * 2
        crop_x = (width - new_width) // 2
        crop_filter = f"crop={new_width}:{height}:{crop_x}:0"
    else:
        new_height = int(width / desired_ratio)
        new_height = (new_height // 2) * 2
        crop_y = (height - new_height) // 2
        crop_filter = f"crop={width}:{new_height}:0:{crop_y}"

    ffmpeg_cmd = [
        "ffmpeg",
        "-i", input_file,
        "-vf", crop_filter,
        "-c:v", "libx264",
        "-crf", "18",
        "-c:a", "copy",
        "-y",
        output_file
    ]

    print(f"Applying crop filter: {crop_filter}")
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Crop completed: {output_file}")

def rescale_video(input_file: str, output_file: str, scale_width: int, scale_height: int):
    ffmpeg_command = [
        'ffmpeg',
        '-i', input_file,
        '-vf', f'scale={scale_width}:{scale_height},fps=25',
        '-c:v', 'libx264',
        '-crf', '18',
        '-c:a', 'copy',
        '-y',
        output_file
    ]
    try:
        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        print(f"Successfully rescaled: {input_file} to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg rescale command: {e}")
        print(f"FFmpeg stderr: {e.stderr}")
        raise

def crop_face_area(
   input_file: str,
   output_file: str,
   crop_size: int = 512,
   offset_x: int = 424,
   offset_y: int = 60):
    ffmpeg_command = [
        'ffmpeg',
        '-i', str(input_file),
        '-vf', f'crop={crop_size}:{crop_size}:{offset_x}:{offset_y}',
        '-q:v', '1',
        '-qmin', '1',
        '-c:a', 'copy',
        '-y',
        str(output_file)
    ]
    try:
        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        print(f"Successfully cropped face area: {input_file} -> {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg crop_face_area command: {e}")
        print(f"FFmpeg stderr: {e.stderr}")
        raise

class LoRATrainer(nn.Module):
    def __init__(self, inp):
        super().__init__()
        self.inp = inp
        self.lora_args = {'lora_mode': inp['lora_mode'], 'lora_r': inp['lora_r']}
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        head_model_dir = inp['head_ckpt']
        torso_model_dir = inp['torso_ckpt']
        model_dir = torso_model_dir if torso_model_dir != '' else head_model_dir
        cmd = f"cp {os.path.join(model_dir, 'config.yaml')} {self.inp['work_dir']}"
        print(cmd)
        os.system(cmd)
        with open(os.path.join(self.inp['work_dir'], 'config.yaml'), "a") as f:
            f.write(f"\nlora_r: {inp['lora_r']}")
            f.write(f"\nlora_mode: {inp['lora_mode']}")
            f.write(f"\nmouth_encode_mode: {inp['mouth_encode_mode']}")
            f.write(f"\n")
        
        # Define helpers first
        self.seg_model = MediapipeSegmenter()
        self.secc_renderer = SECC_Renderer(512)
        self.face3d_helper = Face3DHelper(use_gpu=True, keypoint_mode='lm68')
        self.mp_face3d_helper = Face3DHelper(use_gpu=True, keypoint_mode='mediapipe')

        # Load the main model
        self.secc2video_model = self.load_secc2video(model_dir)
        self.secc2video_model.to(device)
        # print(f"self.secc2video_model: {self.secc2video_model}")
        
        # Now, explicitly unfreeze parameters on the correct model attribute
        mark_only_lora_as_trainable(self.secc2video_model, bias='none')
        if self.lora_args['lora_mode'] != 'none':
            print(f"| Manually unfreezing LoRA parameters for mode: {self.lora_args['lora_mode']}")
            for n, p in self.secc2video_model.named_parameters():
                if 'lora_' in n:
                    p.requires_grad = True
        
        # Also unfreeze mouth encoder params on the correct model
        mode = self.inp.get('mouth_encode_mode', 'none')
        if mode != 'none':
            print(f"| Mouth encode mode is '{mode}'. Unfreezing related modules for training.")
            sr_module = self.secc2video_model.superresolution
            if mode == 'concat':
                for p in sr_module.mouth_encoder_concat.parameters(): p.requires_grad = True
                for p in sr_module.fuse_mouth_conv.parameters(): p.requires_grad = True
            elif mode == 'add':
                for p in sr_module.mouth_encoder_add.parameters(): p.requires_grad = True
            elif mode == 'style_latent':
                for p in sr_module.mouth_encoder_style.parameters(): p.requires_grad = True
                for p in sr_module.style_fusion_layer.parameters(): p.requires_grad = True
            elif mode == 'adain':
                for p in sr_module.mouth_encoder_adain.parameters(): p.requires_grad = True
                for p in sr_module.adain_param_generator.parameters(): p.requires_grad = True
            elif mode == 'gated':
                for p in sr_module.mouth_encoder_gated.parameters(): p.requires_grad = True
                for p in sr_module.gating_network.parameters(): p.requires_grad = True
            elif mode == 'film':
                for p in sr_module.mouth_encoder_film.parameters(): p.requires_grad = True
                for p in sr_module.film_generator.parameters(): p.requires_grad = True

        # --- DEBUG: Print trainable parameters ---
        print("-" * 50)
        print("| Trainable Parameters:")
        trainable_params = 0
        not_trainable_params = 0
        for n, p in self.secc2video_model.named_parameters():
            if p.requires_grad:
                print(f"|  - {n}")
                trainable_params += p.numel()
            else:
                not_trainable_params += p.numel()
        print(f"| Total Trainable Parameters: {trainable_params / 1e6:.3f}M")
        print(f"| Total Not Trainable Parameters: {not_trainable_params / 1e6:.3f}M")
        print("-" * 50)

        
        # --- END DEBUG ---
        
        self.secc2video_model.eval()
        
        # Now that helpers and the main model exist, run preprocessing and data loading
        self.auto_preprocessing(inp)
        self.init_perceptual_losses(inp)
        self.init_advanced_losses(inp)
        self.load_training_data(inp)

    def auto_preprocessing(self, inp):
        video_id = os.path.basename(inp['video_id'])
        if video_id.endswith((".mp4", ".png", ".jpg", ".jpeg")):
            video_id = video_id[:-4]

        raw_video_path = inp['video_id']
        processed_video_path = f'data/raw/videos/{video_id}.mp4'
        head_img_pattern = f'data/processed/videos/{video_id}/head_imgs/*.png'
        coeff_path = f"data/processed/videos/{video_id}/coeff_fit_mp.npy"
        lm2d_path = f'data/processed/videos/{video_id}/lms_2d.npy'

        # Check if all necessary files exist
        if (os.path.exists(processed_video_path) and 
            glob.glob(head_img_pattern) and 
            os.path.exists(coeff_path) and 
            os.path.exists(lm2d_path)):
            print("| All necessary processed files found. Skipping preprocessing.")
            return

        print("| Some processed files are missing. Running automatic preprocessing...")
        
        
        # 1. Crop and rescale video
        if not os.path.exists(processed_video_path):
            print("| Step 1/3: Cropping and resizing video...")
            os.makedirs(os.path.dirname(processed_video_path), exist_ok=True)
            temp_dir = os.path.join(self.inp['work_dir'], 'temp_preprocessing')
            os.makedirs(temp_dir, exist_ok=True)
            cropped_16by9_file = os.path.join(temp_dir, "temp_16x9.mp4")
            rescaled_video_file = os.path.join(temp_dir, "temp_1360x764.mp4")
            crop_video_to_16by9(raw_video_path, cropped_16by9_file)
            rescale_video(cropped_16by9_file, rescaled_video_file, scale_width=1360, scale_height=764)
            crop_face_area(rescaled_video_file, processed_video_path, crop_size=512, 
                           offset_x=inp['offset_x'], offset_y=inp['offset_y'])
            
            print("| Video preprocessing finished.")
        
        # 2. Extract frames, segmentation, and background
        if not glob.glob(head_img_pattern):
            print("| Step 2/3: Extracting frames, segmentation maps, and background...")
            gt_img_dir = f"data/processed/videos/{video_id}/gt_imgs"
            os.makedirs(gt_img_dir, exist_ok=True)
            cmd = f"ffmpeg -i {processed_video_path} -vf fps=25,scale=w=512:h=512 -qmin 1 -q:v 1 -start_number 0 -y {gt_img_dir}/%08d.jpg"
            print(f"| {cmd}")
            os.system(cmd)
            cmd = f"python data_gen/utils/process_video/extract_segment_imgs.py --ds_name=nerf --vid_dir={processed_video_path} --force_single_process" 
            print(f"| {cmd}")
            os.system(cmd)
            print("| Frame extraction finished.")

        # 3. Fit 3DMM and get landmarks. This will create both coeff_path and lm2d_path
        if not os.path.exists(coeff_path) or not os.path.exists(lm2d_path):
            print("| Step 3/3: Fitting 3DMM coefficients and extracting landmarks...")
            # Set save=True to ensure lms_2d.npy is also saved by the fitting function
            fit_3dmm_for_a_video(processed_video_path, save=True)
            print("| 3DMM fitting finished.")
        
        print("| Automatic preprocessing complete.")

    def init_perceptual_losses(self, inp):
        self.criterion_lpips, self.criterion_sd = None, None
        if inp['lambda_lpips'] > 0:
            self.criterion_lpips = lpips.LPIPS(net='vgg',lpips=True).cuda()
            print(f"| LPIPS (VGG) perceptual loss enabled with weight: {inp['lambda_lpips']}")
        if inp['lambda_sd'] > 0:
            self.criterion_sd = SDPerceptualLoss(loss_weight=1.0).cuda()
            print(f"| Stable Diffusion perceptual loss enabled with weight: {inp['lambda_sd']}")

    def init_advanced_losses(self, inp):
        self.adv_loss_fns = inp['adv_loss_fns'].split()
        self.use_adv_loss = len(self.adv_loss_fns) > 0
        if not self.use_adv_loss:
            return

        self.adv_loss_weights = {
            'gan': inp['lambda_gan'],
            'gan_lip': inp['lambda_gan'],
            'fsm_lip': inp['lambda_fsm'],
            'gan_eye': inp['lambda_gan'],
            'fsm_eye': inp['lambda_fsm'],
            'comp_lp': inp['lambda_comp_lp'],
            'comp_lp_eye': inp['lambda_comp_lp'],
            'comp_lp_lip': inp['lambda_comp_lp'],
        }

        # Initialize required models and optimizers based on selected losses
        self.D_full, self.D_lip, self.D_eye = None, None, None
        self.optimizer_D_full, self.optimizer_D_lip, self.optimizer_D_eye = None, None, None
        self.criterion_gan, self.criterion_fsm, self.criterion_comp_lp = None, None, None

        if 'gan' in self.adv_loss_fns:
            self.D_full = MultiScaleDiscriminator(num_D=2).cuda()
            self.optimizer_D_full = torch.optim.Adam(self.D_full.parameters(), lr=inp['lr_d'], betas=(0.9, 0.999))
            self.criterion_gan = GANLoss(gan_mode='hinge').cuda()

        if 'gan_lip' in self.adv_loss_fns or 'fsm_lip' in self.adv_loss_fns:
            self.D_lip = MultiScaleDiscriminator(num_D=2).cuda()
            self.optimizer_D_lip = torch.optim.Adam(self.D_lip.parameters(), lr=inp['lr_d'], betas=(0.9, 0.999))
            if self.criterion_gan is None: self.criterion_gan = GANLoss(gan_mode='hinge').cuda()
        
        if 'fsm_lip' in self.adv_loss_fns:
             self.criterion_fsm = FeatureStyleMatchingLoss().cuda()

        if 'gan_eye' in self.adv_loss_fns or 'fsm_eye' in self.adv_loss_fns:
            self.D_eye = MultiScaleDiscriminator(num_D=2).cuda()
            self.optimizer_D_eye = torch.optim.Adam(self.D_eye.parameters(), lr=inp['lr_d'], betas=(0.9, 0.999))
            if self.criterion_gan is None: self.criterion_gan = GANLoss(gan_mode='hinge').cuda()

        if 'fsm_eye' in self.adv_loss_fns:
            if self.criterion_fsm is None: self.criterion_fsm = FeatureStyleMatchingLoss().cuda()

        if any(k in self.adv_loss_fns for k in ['comp_lp', 'comp_lp_eye', 'comp_lp_lip']):
            self.criterion_comp_lp = FacialComponentPerceptualLoss(loss_weight=1.0).cuda()

        # Load checkpoints for initialized models
        lora_ckpt_path = os.path.join(inp['work_dir'], 'checkpoint.ckpt')
        if os.path.exists(lora_ckpt_path):
             print("| Loading advanced loss models from checkpoint...")
             if self.D_full: load_ckpt(self.D_full, lora_ckpt_path, model_name='D_full', strict=False)
             if self.D_lip: load_ckpt(self.D_lip, lora_ckpt_path, model_name='D_lip', strict=False)
             if self.D_eye: load_ckpt(self.D_eye, lora_ckpt_path, model_name='D_eye', strict=False)
             if self.optimizer_D_full: load_ckpt(self.optimizer_D_full, lora_ckpt_path, model_name='optimizer_D_full', strict=False)
             if self.optimizer_D_lip: load_ckpt(self.optimizer_D_lip, lora_ckpt_path, model_name='optimizer_D_lip', strict=False)
             if self.optimizer_D_eye: load_ckpt(self.optimizer_D_eye, lora_ckpt_path, model_name='optimizer_D_eye', strict=False)

    def load_secc2video(self, model_dir):
        inp = self.inp
        from modules.real3d.secc_img2plane_torso import OSAvatarSECC_Img2plane, OSAvatarSECC_Img2plane_Torso
        hp = set_hparams(f"{model_dir}/config.yaml", print_hparams=False, global_hparams=True)
        
        # FIX: Manually update hparams to sync command-line args with the model's config
        hparams['mouth_encode_mode'] = self.inp.get('mouth_encode_mode', 'none')
        hparams['large_sr'] = self.inp.get('large_sr', False)
        
        hp['htbsr_head_threshold'] = 1.0
        self.neural_rendering_resolution = hp['neural_rendering_resolution']
        if 'torso' in hp['task_cls'].lower():
            self.torso_mode = True
            model = OSAvatarSECC_Img2plane_Torso(hp=hp, lora_args=self.lora_args)
        else:
            self.torso_mode = False
            model = OSAvatarSECC_Img2plane(hp=hp, lora_args=self.lora_args)
        
        lora_ckpt_path = os.path.join(inp['work_dir'], 'checkpoint.ckpt')
        if os.path.exists(lora_ckpt_path):
            self.learnable_triplane = nn.Parameter(torch.zeros([1, 3, model.triplane_hid_dim*model.triplane_depth, 256, 256]).float().cuda(), requires_grad=True)
            model._last_cano_planes = self.learnable_triplane
            load_ckpt(model, lora_ckpt_path, model_name='model', strict=False)   
        else:
            load_ckpt(model, f"{model_dir}", model_name='model', strict=False)   
            
        num_params(model)
        self.model = model 
        return model

    def load_training_data(self, inp):
        video_id = inp['video_id']
        if video_id.endswith((".mp4", ".png", ".jpg", ".jpeg")):
            video_id = os.path.basename(video_id)[:-4]
            inp['video_id'] = video_id
            
        # All preprocessing is assumed to be done by auto_preprocessing now.
        # We just need to load the files.
        coeff_path = f"data/processed/videos/{video_id}/coeff_fit_mp.npy"
        lm2d_path = f'data/processed/videos/{video_id}/lms_2d.npy'

        coeff_dict = np.load(coeff_path, allow_pickle=True).tolist()
        mp_lm2ds = np.load(lm2d_path)
        
        ids = convert_to_tensor(coeff_dict['id']).reshape([-1,80]).cuda()
        exps = convert_to_tensor(coeff_dict['exp']).reshape([-1,64]).cuda()
        eulers = convert_to_tensor(coeff_dict['euler']).reshape([-1,3]).cuda()
        trans = convert_to_tensor(coeff_dict['trans']).reshape([-1,3]).cuda()
        WH = 512

        if mp_lm2ds.shape[1] > 468: # Use the first 468 for consistency if 478 are present
            mp_lm2ds = mp_lm2ds[:, :468, :]
        
        # Create tight component masks using the new utility
        lip_masks, eye_masks = create_component_masks_from_landmarks(mp_lm2ds, WH, WH)

        mouth_ref_img = None
        if self.inp.get('mouth_encode_mode', 'none') != 'none':
            # We need to use the 68-landmark subset for these specific indices
            lm68_for_lip_dist = self.face3d_helper.reconstruct_lm2d(ids, exps, eulers, trans).cpu().numpy() * WH
            lip_dists = lm68_for_lip_dist[:, 66, 1] - lm68_for_lip_dist[:, 62, 1]
            mouth_open_idx = np.argmax(lip_dists)
            print(f"| Found most open mouth frame at index: {mouth_open_idx}")

            # Load the reference image, trying head_imgs first, then gt_imgs as a fallback.
            mouth_ref_img_name = f'data/processed/videos/{video_id}/head_imgs/{format(mouth_open_idx, "08d")}.png'
            if not os.path.exists(mouth_ref_img_name):
                mouth_ref_img_name = f'data/processed/videos/{video_id}/gt_imgs/{format(mouth_open_idx, "08d")}.jpg'
            
            if os.path.exists(mouth_ref_img_name):
                full_ref_img_bgr = cv2.imread(mouth_ref_img_name)
                
                # Get the tight lip mask for the reference frame
                lip_mask_for_ref = lip_masks[mouth_open_idx].cpu().numpy() # [1, H, W]
                lip_mask_for_ref_bgr = np.stack([lip_mask_for_ref[0]]*3, axis=-1) # [H, W, 3]

                # Create a black background and copy the lip region onto it.
                masked_lip_img_bgr = np.zeros_like(full_ref_img_bgr)
                masked_lip_img_bgr[lip_mask_for_ref_bgr > 0] = full_ref_img_bgr[lip_mask_for_ref_bgr > 0]

                # Convert the masked image to a tensor for the model.
                mouth_ref_img = torch.tensor(masked_lip_img_bgr[..., ::-1].copy() / 127.5 - 1).permute(2, 0, 1).float()
                
                # Save the chosen MASKED image for verification.
                save_path = os.path.join(self.inp['work_dir'], 'mouth_reference_frame_masked.png')
                cv2.imwrite(save_path, masked_lip_img_bgr)
                print(f"| Saved MASKED mouth reference frame to {save_path}")
            else:
                print(f"| WARNING: Could not find mouth reference image: {mouth_ref_img_name}")

        kps = self.face3d_helper.reconstruct_lm2d(ids, exps, eulers, trans).cuda()
        kps = (kps-0.5) / 0.5 # rescale to -1~1
        kps = torch.cat([kps, torch.zeros([*kps.shape[:-1], 1]).cuda()], dim=-1)
        
        camera_ret = get_eg3d_convention_camera_pose_intrinsic({'euler': torch.tensor(coeff_dict['euler']).reshape([-1,3]), 'trans': torch.tensor(coeff_dict['trans']).reshape([-1,3])})
        c2w, intrinsics = camera_ret['c2w'], camera_ret['intrinsics']
        cameras = torch.tensor(np.concatenate([c2w.reshape([-1,16]), intrinsics.reshape([-1,9])], axis=-1)).cuda()
        camera_smo_ksize = 7
        cameras = smooth_camera_sequence(cameras.cpu().numpy(), kernel_size=camera_smo_ksize) # [T, 25]
        cameras = torch.tensor(cameras).cuda()
        zero_eulers = eulers * 0
        zero_trans = trans * 0
        _, cano_secc_color = self.secc_renderer(ids[0:1], exps[0:1]*0, zero_eulers[0:1], zero_trans[0:1])
        src_idx = 0
        _, src_secc_color = self.secc_renderer(ids[0:1], exps[src_idx:src_idx+1], zero_eulers[0:1], zero_trans[0:1])
        
        # Initialize lazy-loading placeholders
        num_samples = len(cameras)
        drv_secc_colors = [None for _ in range(num_samples)]
        drv_head_imgs = [None for _ in range(num_samples)]
        drv_torso_imgs = [None for _ in range(num_samples)]
        drv_com_imgs = [None for _ in range(num_samples)]
        segmaps = [None for _ in range(num_samples)]

        img_name = f'data/processed/videos/{video_id}/bg.jpg'
        bg_img = torch.tensor(cv2.imread(img_name)[..., ::-1] / 127.5 - 1).permute(2,0,1).float() # [3, H, W]
        ds = {
            'id': ids.cuda().float(),
            'exps': exps.cuda().float(),
            'eulers': eulers.cuda().float(),
            'trans': trans.cuda().float(),
            'cano_secc_color': cano_secc_color.cuda().float(),
            'src_secc_color': src_secc_color.cuda().float(),
            'cameras': cameras.float(),
            'video_id': video_id,
            'head_imgs': drv_head_imgs,
            'torso_imgs': drv_torso_imgs,
            'com_imgs': drv_com_imgs,
            'bg_img': bg_img,
            'segmaps': segmaps,
            'kps': kps,
            'mouth_ref_img': mouth_ref_img.cuda() if mouth_ref_img is not None else None,
            'lip_masks': lip_masks.cuda(),
            'eye_masks': eye_masks.cuda(),
        }
        self.ds = ds
        # Add eye rects for facial component loss
        lm68_for_rects = self.face3d_helper.reconstruct_lm2d(ids, exps, eulers, trans).cpu().numpy() * WH
        self.ds['eye_rects'] = [get_eye_rects(lm68_for_rects[i], WH, WH) for i in range(len(lm68_for_rects))]
        return ds
    
    def training_loop(self, inp):
        trainer = self
        video_id = self.ds['video_id']
        
        # Collect all parameters that require gradients for the optimizer
        main_model_trainable_params = [p for p in self.secc2video_model.parameters() if p.requires_grad]
        num_main_params = sum(p.numel() for p in main_model_trainable_params)
        print(f"| Found {num_main_params} trainable parameters in the main model.")

        # self.criterion_lpips is now initialized in init_perceptual_losses
        
        self.logger = SummaryWriter(log_dir=inp['work_dir'])
        if not hasattr(self, 'learnable_triplane'):
            src_idx = 0 # init triplane from the first frame's prediction
            self.learnable_triplane = nn.Parameter(torch.zeros([1, 3, self.secc2video_model.triplane_hid_dim*self.secc2video_model.triplane_depth, 256, 256]).float().cuda(), requires_grad=True)
            img_name = f'data/processed/videos/{video_id}/head_imgs/{format(src_idx, "08d")}.png'
            img = torch.tensor(cv2.imread(img_name)[..., ::-1] / 127.5 - 1).permute(2,0,1).float().cuda().float() # [3, H, W]
            cano_plane = self.secc2video_model.cal_cano_plane(img.unsqueeze(0)) # [1, 3, CD, h, w]
            self.learnable_triplane.data = cano_plane.data
            self.secc2video_model._last_cano_planes = self.learnable_triplane
        if len(main_model_trainable_params) == 0:
            # Case where only the triplane is being trained
            self.optimizer = torch.optim.AdamW([self.learnable_triplane], lr=inp['lr_triplane'], weight_decay=0.01, betas=(0.9,0.98))
        else:
            # Case where LoRA and/or SR parameters are being trained
            self.optimizer = torch.optim.Adam(main_model_trainable_params, lr=inp['lr'], betas=(0.9,0.98))
            self.optimizer.add_param_group({
                'params': [self.learnable_triplane],
                'lr': inp['lr_triplane'],
                'betas': (0.9, 0.98)
            })
        
        ids = self.ds['id']
        exps = self.ds['exps']
        zero_eulers = self.ds['eulers']*0
        zero_trans = self.ds['trans']*0
        num_updates = inp['max_updates']
        batch_size = inp['batch_size'] # 1 for lower gpu mem usage
        num_samples = len(self.ds['cameras'])
        init_plane = self.learnable_triplane.detach().clone()
        if num_samples <= 5:
            lambda_reg_triplane = 1.0
        elif num_samples <= 250:
            lambda_reg_triplane = 0.1
        else:
            lambda_reg_triplane = 0.
        
        loss_weights = {
            'v2v_occlusion_reg_l1_loss': 0.001,
            'v2v_occlusion_2_reg_l1_loss': 0.001,
            'v2v_occlusion_2_weights_entropy_loss': hparams['lam_occlusion_weights_entropy'],
            'density_weight_l2_loss': 0.01,
            'density_weight_entropy_loss': 0.001,
            'mse_loss': 1.,
            'head_mse_loss': 0.2 * inp['head_loss_w_mult'],
            'lip_mse_loss': 1.0 * inp['lip_loss_w_mult'],
            'blink_reg_loss': inp['blink_reg_loss_w'],
            'triplane_reg_loss': lambda_reg_triplane,
            'secc_reg_loss': inp['secc_reg_loss_w'],
        }

        # Add perceptual loss weights if used
        if self.criterion_lpips:
            loss_weights['lpips_loss'] = inp['lambda_lpips']
            loss_weights['head_lpips_loss'] = 0.1 * inp['lambda_lpips']
            loss_weights['lip_lpips_loss'] = 1.0 * inp['lambda_lpips']
        
        if self.criterion_sd:
            loss_weights['sd_loss'] = inp['lambda_sd']
            loss_weights['head_sd_loss'] = 0.1 * inp['lambda_sd']
            loss_weights['lip_sd_loss'] = 1.0 * inp['lambda_sd']

        # Add advanced loss weights if used
        for loss_name in self.adv_loss_fns:
            loss_weights[loss_name] = self.adv_loss_weights[loss_name]

        for i_step in tqdm.trange(num_updates+1,desc="training lora..."):
            milestone_steps = []
            # milestone_steps = [100, 200, 500]
            if i_step % 1000 == 0 or i_step in milestone_steps:
                # Save a checkpoint BEFORE running validation, especially for step 0
                if i_step == 0:
                    filepath = os.path.join(inp['work_dir'], f"model_ckpt_steps_{i_step}.ckpt") 
                    checkpoint = self.dump_checkpoint(inp)
                    tmp_path = str(filepath) + ".part"
                    torch.save(checkpoint, tmp_path, _use_new_zipfile_serialization=False)
                    os.replace(tmp_path, filepath)

                trainer.test_loop(inp, step=i_step)
                if i_step != 0:
                    filepath = os.path.join(inp['work_dir'], f"model_ckpt_steps_{i_step}.ckpt") 
                    checkpoint = self.dump_checkpoint(inp)
                    tmp_path = str(filepath) + ".part"
                    torch.save(checkpoint, tmp_path, _use_new_zipfile_serialization=False)
                    os.replace(tmp_path, filepath)
            
            drv_idx = [random.randint(0, num_samples-1) for _ in range(batch_size)]
            drv_secc_colors = []
            gt_imgs = []
            head_imgs = []
            segmaps_0 = []
            segmaps = []
            torso_imgs = []
            drv_eye_rects = []
            kp_src = []
            kp_drv = []
            mouth_masks_for_batch = []
            eye_masks_for_batch = []
            for di in drv_idx:
                # 读取target image
                if self.torso_mode:
                    if self.ds['com_imgs'][di] is None:
                        # img_name = f'data/processed/videos/{video_id}/gt_imgs/{format(di, "08d")}.jpg'
                        img_name = f'data/processed/videos/{video_id}/com_imgs/{format(di, "08d")}.jpg'
                        img = torch.tensor(cv2.imread(img_name)[..., ::-1] / 127.5 - 1).permute(2,0,1).float() # [3, H, W]
                        self.ds['com_imgs'][di] = img
                    gt_imgs.append(self.ds['com_imgs'][di])
                else:
                    if self.ds['head_imgs'][di] is None:
                        img_name = f'data/processed/videos/{video_id}/head_imgs/{format(di, "08d")}.png'
                        img = torch.tensor(cv2.imread(img_name)[..., ::-1] / 127.5 - 1).permute(2,0,1).float() # [3, H, W]
                        self.ds['head_imgs'][di] = img
                    gt_imgs.append(self.ds['head_imgs'][di])
                
                # The head_imgs list must always be populated for the head losses
                if self.ds['head_imgs'][di] is None:
                    img_name = f'data/processed/videos/{video_id}/head_imgs/{format(di, "08d")}.png'
                    img = torch.tensor(cv2.imread(img_name)[..., ::-1] / 127.5 - 1).permute(2,0,1).float() # [3, H, W]
                    self.ds['head_imgs'][di] = img
                head_imgs.append(self.ds['head_imgs'][di])

                # 使用第一帧的torso作为face v2v的输入
                if self.ds['torso_imgs'][0] is None:
                    img_name = f'data/processed/videos/{video_id}/inpaint_torso_imgs/{format(0, "08d")}.png'
                    img = torch.tensor(cv2.imread(img_name)[..., ::-1] / 127.5 - 1).permute(2,0,1).float() # [3, H, W]
                    self.ds['torso_imgs'][0] = img
                torso_imgs.append(self.ds['torso_imgs'][0])
                # 所以segmap也用第一帧的了
                if self.ds['segmaps'][0] is None:
                    img_name = f'data/processed/videos/{video_id}/segmaps/{format(0, "08d")}.png'
                    seg_img = cv2.imread(img_name)[:,:, ::-1]
                    segmap = torch.from_numpy(decode_segmap_mask_from_image(seg_img)) # [6, H, W]
                    self.ds['segmaps'][0] = segmap
                segmaps_0.append(self.ds['segmaps'][0])
                if self.ds['segmaps'][di] is None:
                    img_name = f'data/processed/videos/{video_id}/segmaps/{format(di, "08d")}.png'
                    seg_img = cv2.imread(img_name)[:,:, ::-1]
                    segmap = torch.from_numpy(decode_segmap_mask_from_image(seg_img)) # [6, H, W]
                    self.ds['segmaps'][di] = segmap
                segmaps.append(self.ds['segmaps'][di])
                
                # Get the pre-computed mouth mask for the current frame
                mouth_masks_for_batch.append(self.ds['lip_masks'][di])
                eye_masks_for_batch.append(self.ds['eye_masks'][di])
                
                _, secc_color = self.secc_renderer(ids[0:1], exps[di:di+1], zero_eulers[0:1], zero_trans[0:1])
                drv_secc_colors.append(secc_color)
                drv_eye_rects.append(self.ds['eye_rects'][di])
                kp_src.append(self.ds['kps'][0])
                kp_drv.append(self.ds['kps'][di])
            bg_img = self.ds['bg_img'].unsqueeze(0).repeat([batch_size, 1, 1, 1]).cuda()
            ref_torso_imgs = torch.stack(torso_imgs).float().cuda()
            kp_src = torch.stack(kp_src).float().cuda()
            kp_drv = torch.stack(kp_drv).float().cuda()
            segmaps = torch.stack(segmaps).float().cuda()
            segmaps_0 = torch.stack(segmaps_0).float().cuda()
            tgt_imgs = torch.stack(gt_imgs).float().cuda()
            head_imgs = torch.stack(head_imgs).float().cuda()
            drv_secc_color = torch.cat(drv_secc_colors)
            cano_secc_color = self.ds['cano_secc_color'].repeat([batch_size, 1, 1, 1])
            src_secc_color = self.ds['src_secc_color'].repeat([batch_size, 1, 1, 1])
            mouth_masks_for_batch = torch.stack(mouth_masks_for_batch).float().cuda()
            eye_masks_for_batch = torch.stack(eye_masks_for_batch).float().cuda()
            cond = {'cond_cano': cano_secc_color,'cond_src': src_secc_color, 'cond_tgt': drv_secc_color,
                    'ref_torso_img': ref_torso_imgs, 'bg_img': bg_img, 
                    'segmap': segmaps_0, # v2v使用第一帧的torso作为source image来warp
                    'kp_s': kp_src, 'kp_d': kp_drv, 'mouth_mask': mouth_masks_for_batch}
            if self.inp.get('mouth_encode_mode', 'none') != 'none' and self.ds['mouth_ref_img'] is not None:
                cond['mouth_ref_img'] = self.ds['mouth_ref_img'].unsqueeze(0).repeat([batch_size, 1, 1, 1])
            camera = self.ds['cameras'][drv_idx]
            gen_output = self.secc2video_model.forward(img=None, camera=camera, cond=cond, ret={}, cache_backbone=False, use_cached_backbone=True)
            pred_imgs = gen_output['image']
            pred_imgs_raw = gen_output['image_raw']

            losses = {}
            occlusion_reg_l1 = gen_output.get("losses", {}).get('facev2v/occlusion_reg_l1', 0.)
            occlusion_2_reg_l1 = gen_output.get("losses", {}).get('facev2v/occlusion_2_reg_l1', 0.)
            occlusion_2_weights_entropy = gen_output.get("losses", {}).get('facev2v/occlusion_2_weights_entropy', 0.)
            losses['v2v_occlusion_reg_l1_loss'] = occlusion_reg_l1
            losses['v2v_occlusion_2_reg_l1_loss'] = occlusion_2_reg_l1
            losses['v2v_occlusion_2_weights_entropy_loss'] = occlusion_2_weights_entropy

            # Weights Reg loss in torso
            neural_rendering_reso = self.neural_rendering_resolution
            alphas = gen_output['weights_img'].clamp(1e-5, 1 - 1e-5)
            loss_weights_entropy = torch.mean(- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas))
            mv_head_masks = segmaps[:, [1,3,5]].sum(dim=1)
            mv_head_masks_raw = F.interpolate(mv_head_masks.unsqueeze(1), size=(neural_rendering_reso,neural_rendering_reso)).squeeze(1)
            face_mask = mv_head_masks_raw.bool().unsqueeze(1)
            nonface_mask = ~ face_mask
            loss_weights_l2_loss = (alphas[nonface_mask]-0).pow(2).mean() + (alphas[face_mask]-1).pow(2).mean()
            losses['density_weight_l2_loss'] = loss_weights_l2_loss
            losses['density_weight_entropy_loss'] = loss_weights_entropy

            # --- Loss Calculation ---
            if inp.get('l1', False):
                mse_loss = (pred_imgs - tgt_imgs).abs().mean()
                head_mse_loss = (pred_imgs_raw - F.interpolate(head_imgs, size=(neural_rendering_reso,neural_rendering_reso), mode='bilinear', antialias=True)).abs().mean()
            else:
                mse_loss = (pred_imgs - tgt_imgs).pow(2).mean()
                head_mse_loss = (pred_imgs_raw - F.interpolate(head_imgs, size=(neural_rendering_reso,neural_rendering_reso), mode='bilinear', antialias=True)).pow(2).mean()
            
            if self.criterion_lpips:
                losses['lpips_loss'] = self.criterion_lpips(pred_imgs, tgt_imgs).mean()
                losses['head_lpips_loss'] = self.criterion_lpips(pred_imgs_raw, F.interpolate(head_imgs, size=(neural_rendering_reso,neural_rendering_reso), mode='bilinear', antialias=True)).mean()
            
            if self.criterion_sd:
                losses['sd_loss'] = self.criterion_sd(pred_imgs, tgt_imgs)
                losses['head_sd_loss'] = self.criterion_sd(pred_imgs_raw, F.interpolate(head_imgs, size=(neural_rendering_reso,neural_rendering_reso), mode='bilinear', antialias=True))

            # Prepare for component losses
            lip_mse_loss = 0
            lip_lpips_loss = 0
            lip_sd_loss = 0
            
            lip_mask_for_loss = mouth_masks_for_batch.float()
            
            # Masked MSE (L1) loss for lips
            lip_pixel_count = lip_mask_for_loss.sum()
            if lip_pixel_count > 0:
                if inp.get('l1', False):
                    lip_mse_loss = (pred_imgs - tgt_imgs).abs() * lip_mask_for_loss
                else:
                    lip_mse_loss = (pred_imgs - tgt_imgs).pow(2) * lip_mask_for_loss
                lip_mse_loss = lip_mse_loss.sum() / lip_pixel_count
            
            # Masked LPIPS loss for lips
            if self.criterion_lpips:
                # We apply the mask to both images before passing to LPIPS
                # The background will be black (value=-1), focusing the loss on the unmasked region
                lip_pred_masked = pred_imgs * lip_mask_for_loss - (1 - lip_mask_for_loss)
                lip_tgt_masked = tgt_imgs * lip_mask_for_loss - (1 - lip_mask_for_loss)
                lip_lpips_loss = self.criterion_lpips(lip_pred_masked, lip_tgt_masked).mean()

            # Masked SD loss for lips
            if self.criterion_sd:
                lip_pred_masked = pred_imgs * lip_mask_for_loss - (1 - lip_mask_for_loss)
                lip_tgt_masked = tgt_imgs * lip_mask_for_loss - (1 - lip_mask_for_loss)
                lip_sd_loss = self.criterion_sd(lip_pred_masked, lip_tgt_masked)

            if self.use_adv_loss:
                if 'comp_lp' in self.adv_loss_fns:
                    component_mask = (eye_masks_for_batch + mouth_masks_for_batch).clamp(0, 1)
                    losses['comp_lp'] = self.criterion_comp_lp(pred_imgs, tgt_imgs, component_mask)
                if 'comp_lp_eye' in self.adv_loss_fns:
                    losses['comp_lp_eye'] = self.criterion_comp_lp(pred_imgs, tgt_imgs, eye_masks_for_batch)
                if 'comp_lp_lip' in self.adv_loss_fns:
                    losses['comp_lp_lip'] = self.criterion_comp_lp(pred_imgs, tgt_imgs, mouth_masks_for_batch)

            losses['mse_loss'] = mse_loss
            losses['head_mse_loss'] = head_mse_loss
            
            if self.criterion_lpips:
                losses['lip_lpips_loss'] = lip_lpips_loss / batch_size
            
            if self.criterion_sd:
                losses['lip_sd_loss'] = lip_sd_loss / batch_size
            
            losses['lip_mse_loss'] = lip_mse_loss / batch_size

            # eye blink reg loss
            if i_step % 4 == 0:
                blink_secc_lst1 = []
                blink_secc_lst2 = []
                blink_secc_lst3 = []
                for i in range(len(drv_secc_color)):
                    secc = drv_secc_color[i]
                    blink_percent1 = random.random() * 0.5 # 0~0.5
                    blink_percent3 = 0.5 + random.random() * 0.5 # 0.5~1.0
                    blink_percent2 = (blink_percent1 + blink_percent3)/2
                    try:
                        out_secc1 = blink_eye_for_secc(secc, blink_percent1).to(secc.device)
                        out_secc2 = blink_eye_for_secc(secc, blink_percent2).to(secc.device)
                        out_secc3 = blink_eye_for_secc(secc, blink_percent3).to(secc.device)
                    except:
                        print("blink eye for secc failed, use original secc")
                        out_secc1 = copy.deepcopy(secc)
                        out_secc2 = copy.deepcopy(secc)
                        out_secc3 = copy.deepcopy(secc)
                    blink_secc_lst1.append(out_secc1)
                    blink_secc_lst2.append(out_secc2)
                    blink_secc_lst3.append(out_secc3)
                src_secc_color1 = torch.stack(blink_secc_lst1)
                src_secc_color2 = torch.stack(blink_secc_lst2)
                src_secc_color3 = torch.stack(blink_secc_lst3)
            blink_cond1 = {'cond_cano': cano_secc_color, 'cond_src': src_secc_color, 'cond_tgt': src_secc_color1}
            blink_cond2 = {'cond_cano': cano_secc_color, 'cond_src': src_secc_color, 'cond_tgt': src_secc_color2}
            blink_cond3 = {'cond_cano': cano_secc_color, 'cond_src': src_secc_color, 'cond_tgt': src_secc_color3}
            blink_secc_plane1 = self.model.cal_secc_plane(blink_cond1)
            blink_secc_plane2 = self.model.cal_secc_plane(blink_cond2)
            blink_secc_plane3 = self.model.cal_secc_plane(blink_cond3)
            interpolate_blink_secc_plane = (blink_secc_plane1 + blink_secc_plane3) / 2
            blink_reg_loss = torch.nn.functional.l1_loss(blink_secc_plane2, interpolate_blink_secc_plane)
            losses['blink_reg_loss'] = blink_reg_loss

            # Triplane Reg loss
            triplane_reg_loss = (self.learnable_triplane - init_plane).abs().mean()
            losses['triplane_reg_loss'] = triplane_reg_loss

            ref_id = self.ds['id'][0:1]
            secc_pertube_randn_scale = hparams['secc_pertube_randn_scale']
            perturbed_id = ref_id + torch.randn_like(ref_id) * secc_pertube_randn_scale
            drv_exp = self.ds['exps'][drv_idx]
            perturbed_exp = drv_exp + torch.randn_like(drv_exp) * secc_pertube_randn_scale
            zero_euler = torch.zeros([len(drv_idx), 3], device=ref_id.device, dtype=ref_id.dtype)
            zero_trans = torch.zeros([len(drv_idx), 3], device=ref_id.device, dtype=ref_id.dtype)
            perturbed_secc = self.secc_renderer(perturbed_id, perturbed_exp, zero_euler, zero_trans)[1]
            secc_reg_loss = torch.nn.functional.l1_loss(drv_secc_color, perturbed_secc)
            losses['secc_reg_loss'] = secc_reg_loss

            ############################################
            # GAN and Feature Matching Loss Calculation
            ############################################
            if self.use_adv_loss:
                # Generator loss
                # 1. Full image GAN loss
                if 'gan' in self.adv_loss_fns:
                    losses['gan'] = self.criterion_gan(self.D_full(pred_imgs), target_is_real=True)

                # 2. Lip region GAN and FSM loss
                if 'gan_lip' in self.adv_loss_fns or 'fsm_lip' in self.adv_loss_fns:
                    # Using mask instead of rect for GAN and FSM on lips
                    lip_pred_region = pred_imgs * mouth_masks_for_batch - (1 - mouth_masks_for_batch) # Black background
                    lip_gt_region = tgt_imgs * mouth_masks_for_batch - (1 - mouth_masks_for_batch)

                    if 'gan_lip' in self.adv_loss_fns:
                        losses['gan_lip'] = self.criterion_gan(self.D_lip(lip_pred_region), target_is_real=True)
                    if 'fsm_lip' in self.adv_loss_fns:
                        pred_lip_feats = self.D_lip.get_features(lip_pred_region)
                        gt_lip_feats = self.D_lip.get_features(lip_gt_region.detach())
                        losses['fsm_lip'] = self.criterion_fsm(pred_lip_feats, gt_lip_feats)

                # 3. Eye region GAN and FSM loss
                if 'gan_eye' in self.adv_loss_fns or 'fsm_eye' in self.adv_loss_fns:
                    (l_xmin, l_xmax, l_ymin, l_ymax), (r_xmin, r_xmax, r_ymin, r_ymax) = drv_eye_rects[0]
                    pred_eye_region = pred_imgs[:, :, l_ymin:l_ymax, l_xmin:l_xmax]
                    gt_eye_region = tgt_imgs[:, :, l_ymin:l_ymax, l_xmin:l_xmax]
                    
                    if 'gan_eye' in self.adv_loss_fns:
                        losses['gan_eye'] = self.criterion_gan(self.D_eye(pred_eye_region), target_is_real=True)
                    if 'fsm_eye' in self.adv_loss_fns:
                        pred_eye_feats = self.D_eye.get_features(pred_eye_region)
                        gt_eye_feats = self.D_eye.get_features(gt_eye_region.detach())
                        losses['fsm_eye'] = self.criterion_fsm(pred_eye_feats, gt_eye_feats)

            total_loss = sum([loss_weights.get(k, 1.0) * v for k, v in losses.items() if isinstance(v, torch.Tensor) and v.requires_grad])
            
            # Update Generator
            self.optimizer.zero_grad()
            total_loss.backward()
            if self.learnable_triplane.grad is not None:
                self.learnable_triplane.grad.data = self.learnable_triplane.grad.data * self.learnable_triplane.numel()
            self.optimizer.step()

            # Update Discriminators
            if self.use_adv_loss:
                # Full Image Discriminator
                if self.D_full:
                    self.optimizer_D_full.zero_grad()
                    loss_d_full_real = self.criterion_gan(self.D_full(tgt_imgs.detach()), target_is_real=True)
                    loss_d_full_fake = self.criterion_gan(self.D_full(pred_imgs.detach()), target_is_real=False)
                    loss_d_full = (loss_d_full_real + loss_d_full_fake) * 0.5
                    loss_d_full.backward()
                    self.optimizer_D_full.step()
                
                # Lip Discriminator
                if self.D_lip:
                    self.optimizer_D_lip.zero_grad()
                    # Re-slice lip region for discriminator update
                    pred_lip_region = (pred_imgs * mouth_masks_for_batch - (1-mouth_masks_for_batch)).detach()
                    gt_lip_region = (tgt_imgs * mouth_masks_for_batch - (1-mouth_masks_for_batch)).detach()
                    loss_d_lip_real = self.criterion_gan(self.D_lip(gt_lip_region), target_is_real=True)
                    loss_d_lip_fake = self.criterion_gan(self.D_lip(pred_lip_region), target_is_real=False)
                    loss_d_lip = (loss_d_lip_real + loss_d_lip_fake) * 0.5
                    loss_d_lip.backward()
                    self.optimizer_D_lip.step()

                # Eye Discriminator
                if self.D_eye:
                    self.optimizer_D_eye.zero_grad()
                    # Re-slice eye region for discriminator update
                    (l_xmin, l_xmax, l_ymin, l_ymax), _ = drv_eye_rects[0]
                    pred_eye_region = pred_imgs[:, :, l_ymin:l_ymax, l_xmin:l_xmax].detach()
                    gt_eye_region = tgt_imgs[:, :, l_ymin:l_ymax, l_xmin:l_xmax].detach()
                    loss_d_eye_real = self.criterion_gan(self.D_eye(gt_eye_region), target_is_real=True)
                    loss_d_eye_fake = self.criterion_gan(self.D_eye(pred_eye_region), target_is_real=False)
                    loss_d_eye = (loss_d_eye_real + loss_d_eye_fake) * 0.5
                    loss_d_eye.backward()
                    self.optimizer_D_eye.step()

            meter.update(total_loss.item())
            if i_step % 10 == 0:
                log_line = f"Iter {i_step+1}: total_loss={meter.avg:.3f} "
                
                raw_losses_tb = {}
                weighted_losses_tb = {}

                for k, v in losses.items():
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    log_line = log_line + f" {k}={val:.3f} "
                    self.logger.add_scalar(f'train/{k}', val, i_step)

                    if isinstance(v, torch.Tensor):
                        raw_val = v.item()
                        weight = loss_weights.get(k, 1.0)
                        weighted_val = raw_val * weight
                        
                        raw_losses_tb[k] = raw_val
                        self.logger.add_scalars(f'train/raw_losses', raw_losses_tb, i_step)

                        if v.requires_grad: # Only log weighted losses that contribute to total_loss
                            weighted_losses_tb[k] = weighted_val
                        
                        self.logger.add_scalars(f'train/weighted_losses', weighted_losses_tb, i_step)
                        
                        log_line += f" {k}={raw_val:.4f} "

                self.logger.add_scalar('train/total_loss', total_loss.item(), i_step)
                

                print(log_line)
                meter.reset()
                
        # Mouth-focused tuning phase
        if inp['mouth_tuning_steps'] > 0:
            print(f"| Starting mouth-focused tuning for {inp['mouth_tuning_steps']} steps...")
            mouth_meter = AvgrageMeter()

            for i_step_mouth in tqdm.trange(inp['mouth_tuning_steps'], desc="mouth tuning..."):
                # --- Batch Preparation (copied from main loop) ---
                drv_idx = [random.randint(0, num_samples-1) for _ in range(batch_size)]
                drv_secc_colors = []
                gt_imgs = []
                head_imgs = []
                segmaps_0 = []
                segmaps = []
                torso_imgs = []
                drv_eye_rects = []
                kp_src = []
                kp_drv = []
                mouth_masks_for_batch = []
                eye_masks_for_batch = []
                for di in drv_idx:
                    # 读取target image
                    if self.torso_mode:
                        if self.ds['com_imgs'][di] is None:
                            img_name = f'data/processed/videos/{video_id}/com_imgs/{format(di, "08d")}.jpg'
                            img = torch.tensor(cv2.imread(img_name)[..., ::-1] / 127.5 - 1).permute(2,0,1).float() # [3, H, W]
                            self.ds['com_imgs'][di] = img
                        gt_imgs.append(self.ds['com_imgs'][di])
                    else:
                        if self.ds['head_imgs'][di] is None:
                            img_name = f'data/processed/videos/{video_id}/head_imgs/{format(di, "08d")}.png'
                            img = torch.tensor(cv2.imread(img_name)[..., ::-1] / 127.5 - 1).permute(2,0,1).float() # [3, H, W]
                            self.ds['head_imgs'][di] = img
                        gt_imgs.append(self.ds['head_imgs'][di])
                    
                    if self.ds['head_imgs'][di] is None:
                        img_name = f'data/processed/videos/{video_id}/head_imgs/{format(di, "08d")}.png'
                        img = torch.tensor(cv2.imread(img_name)[..., ::-1] / 127.5 - 1).permute(2,0,1).float()
                        self.ds['head_imgs'][di] = img
                    head_imgs.append(self.ds['head_imgs'][di])

                    if self.ds['torso_imgs'][0] is None:
                        img_name = f'data/processed/videos/{video_id}/inpaint_torso_imgs/{format(0, "08d")}.png'
                        img = torch.tensor(cv2.imread(img_name)[..., ::-1] / 127.5 - 1).permute(2,0,1).float()
                        self.ds['torso_imgs'][0] = img
                    torso_imgs.append(self.ds['torso_imgs'][0])
                    if self.ds['segmaps'][0] is None:
                        img_name = f'data/processed/videos/{video_id}/segmaps/{format(0, "08d")}.png'
                        seg_img = cv2.imread(img_name)[:,:, ::-1]
                        segmap = torch.from_numpy(decode_segmap_mask_from_image(seg_img))
                        self.ds['segmaps'][0] = segmap
                    segmaps_0.append(self.ds['segmaps'][0])
                    if self.ds['segmaps'][di] is None:
                        img_name = f'data/processed/videos/{video_id}/segmaps/{format(di, "08d")}.png'
                        seg_img = cv2.imread(img_name)[:,:, ::-1]
                        segmap = torch.from_numpy(decode_segmap_mask_from_image(seg_img))
                        self.ds['segmaps'][di] = segmap
                    segmaps.append(self.ds['segmaps'][di])
                    
                    mouth_masks_for_batch.append(self.ds['lip_masks'][di])
                    eye_masks_for_batch.append(self.ds['eye_masks'][di])
                    
                    _, secc_color = self.secc_renderer(ids[0:1], exps[di:di+1], zero_eulers[0:1], zero_trans[0:1])
                    drv_secc_colors.append(secc_color)
                    drv_eye_rects.append(self.ds['eye_rects'][di])
                    kp_src.append(self.ds['kps'][0])
                    kp_drv.append(self.ds['kps'][di])

                bg_img = self.ds['bg_img'].unsqueeze(0).repeat([batch_size, 1, 1, 1]).cuda()
                ref_torso_imgs = torch.stack(torso_imgs).float().cuda()
                kp_src = torch.stack(kp_src).float().cuda()
                kp_drv = torch.stack(kp_drv).float().cuda()
                segmaps = torch.stack(segmaps).float().cuda()
                segmaps_0 = torch.stack(segmaps_0).float().cuda()
                tgt_imgs = torch.stack(gt_imgs).float().cuda()
                head_imgs = torch.stack(head_imgs).float().cuda()
                drv_secc_color = torch.cat(drv_secc_colors)
                cano_secc_color = self.ds['cano_secc_color'].repeat([batch_size, 1, 1, 1])
                src_secc_color = self.ds['src_secc_color'].repeat([batch_size, 1, 1, 1])
                mouth_masks_for_batch = torch.stack(mouth_masks_for_batch).float().cuda()

                cond = {'cond_cano': cano_secc_color,'cond_src': src_secc_color, 'cond_tgt': drv_secc_color,
                        'ref_torso_img': ref_torso_imgs, 'bg_img': bg_img, 
                        'segmap': segmaps_0, 'kp_s': kp_src, 'kp_d': kp_drv, 'mouth_mask': mouth_masks_for_batch}
                if self.inp.get('mouth_encode_mode', 'none') != 'none' and self.ds['mouth_ref_img'] is not None:
                    cond['mouth_ref_img'] = self.ds['mouth_ref_img'].unsqueeze(0).repeat([batch_size, 1, 1, 1])
                
                # --- Forward pass ---
                camera = self.ds['cameras'][drv_idx]
                gen_output = self.secc2video_model.forward(img=None, camera=camera, cond=cond, ret={}, cache_backbone=False, use_cached_backbone=True)
                pred_imgs = gen_output['image']
                pred_imgs_raw = gen_output['image_raw']
                neural_rendering_reso = self.neural_rendering_resolution


                # --- Loss Calculation (All losses for logging) ---
                losses_for_logging = {}
                
                # Standard losses from main loop
                if inp.get('l1', False):
                    losses_for_logging['mse_loss'] = (pred_imgs - tgt_imgs).abs().mean()
                    losses_for_logging['head_mse_loss'] = (pred_imgs_raw - F.interpolate(head_imgs, size=(neural_rendering_reso,neural_rendering_reso), mode='bilinear', antialias=True)).abs().mean()
                else:
                    losses_for_logging['mse_loss'] = (pred_imgs - tgt_imgs).pow(2).mean()
                    losses_for_logging['head_mse_loss'] = (pred_imgs_raw - F.interpolate(head_imgs, size=(neural_rendering_reso,neural_rendering_reso), mode='bilinear', antialias=True)).pow(2).mean()
                if self.criterion_lpips:
                    losses_for_logging['lpips_loss'] = self.criterion_lpips(pred_imgs, tgt_imgs).mean()
                    losses_for_logging['head_lpips_loss'] = self.criterion_lpips(pred_imgs_raw, F.interpolate(head_imgs, size=(neural_rendering_reso,neural_rendering_reso), mode='bilinear', antialias=True)).mean()
                if self.criterion_sd:
                    losses_for_logging['sd_loss'] = self.criterion_sd(pred_imgs, tgt_imgs)
                    losses_for_logging['head_sd_loss'] = self.criterion_sd(pred_imgs_raw, F.interpolate(head_imgs, size=(neural_rendering_reso,neural_rendering_reso), mode='bilinear', antialias=True))
                
                # Lip losses (used for optimization)
                lip_mse_loss = 0
                lip_lpips_loss = 0
                lip_sd_loss = 0

                lip_mask_for_loss = mouth_masks_for_batch.float()
                lip_pixel_count = lip_mask_for_loss.sum()
                if lip_pixel_count > 0:
                    if inp.get('l1', False):
                        lip_mse_loss = ((pred_imgs - tgt_imgs).abs() * lip_mask_for_loss).sum() / lip_pixel_count
                    else:
                        lip_mse_loss = ((pred_imgs - tgt_imgs).pow(2) * lip_mask_for_loss).sum() / lip_pixel_count
                    if self.criterion_lpips:
                        lip_pred_masked = pred_imgs * lip_mask_for_loss - (1 - lip_mask_for_loss)
                        lip_tgt_masked = tgt_imgs * lip_mask_for_loss - (1 - lip_mask_for_loss)
                        lip_lpips_loss = self.criterion_lpips(lip_pred_masked, lip_tgt_masked).mean()
                    if self.criterion_sd:
                        lip_pred_masked = pred_imgs * lip_mask_for_loss - (1 - lip_mask_for_loss)
                        lip_tgt_masked = tgt_imgs * lip_mask_for_loss - (1 - lip_mask_for_loss)
                        lip_sd_loss = self.criterion_sd(lip_pred_masked, lip_tgt_masked)

                losses_for_logging['lip_mse_loss'] = lip_mse_loss
                if self.criterion_lpips: losses_for_logging['lip_lpips_loss'] = lip_lpips_loss
                if self.criterion_sd: losses_for_logging['lip_sd_loss'] = lip_sd_loss

                # --- Backward Pass (Mouth losses only) ---
                mouth_losses_for_optim = {}
                if isinstance(lip_mse_loss, torch.Tensor) and lip_mse_loss.requires_grad:
                    mouth_losses_for_optim['lip_mse_loss'] = lip_mse_loss
                if self.criterion_lpips and isinstance(lip_lpips_loss, torch.Tensor) and lip_lpips_loss.requires_grad:
                    mouth_losses_for_optim['lip_lpips_loss'] = lip_lpips_loss
                
                total_mouth_loss = sum([loss_weights.get(k, 1.0) * v for k, v in mouth_losses_for_optim.items()])

                self.optimizer.zero_grad()
                total_mouth_loss.backward()
                self.optimizer.step()

                # --- Logging ---
                mouth_meter.update(total_mouth_loss.item())
                if i_step_mouth % 10 == 0:
                    log_line = f"Mouth Iter {i_step_mouth+1}: total_loss={mouth_meter.avg:.3f} "
                    for k, v in losses_for_logging.items():
                        val = v.item() if isinstance(v, torch.Tensor) else v
                        log_line += f" {k}={val:.4f} "
                        self.logger.add_scalar(f'train_mouth/{k}', val, i_step_mouth)
                    self.logger.add_scalar('train_mouth/total_loss', total_mouth_loss.item(), i_step_mouth)
                    print(log_line)
                    mouth_meter.reset()
                if i_step_mouth % 100 == 99:
                    self.test_loop(self.inp, f"{inp['max_updates']}_{i_step_mouth}")
    @torch.no_grad()
    def test_loop(self, inp, step=''):
        self.model.eval()
        # coeff_dict = np.load('data/processed/videos/Lieu/coeff_fit_mp_for_lora.npy', allow_pickle=True).tolist()
        # drv_exps = torch.tensor(coeff_dict['exp']).cuda().float()
        drv_exps = self.ds['exps']
        zero_eulers = self.ds['eulers']*0
        zero_trans = self.ds['trans']*0
        batch_size = 1
        num_samples = len(self.ds['cameras'])
        video_writer = imageio.get_writer(os.path.join(inp['work_dir'], f'val_step{step}.mp4'), fps=25)
        
        masked_comp_video_writer = imageio.get_writer(os.path.join(inp['work_dir'], f'val_step{step}_masked_components.mp4'), fps=25)
        lip_mask_video_writer = imageio.get_writer(os.path.join(inp['work_dir'], f'val_step{step}_lip_mask.mp4'), fps=25)
        eye_mask_video_writer = imageio.get_writer(os.path.join(inp['work_dir'], f'val_step{step}_eye_mask.mp4'), fps=25)

        total_iters = min(num_samples, 250)
        video_id = inp['video_id']
        for i in tqdm.trange(total_iters,desc="testing lora..."):
            drv_idx = [i]
            drv_secc_colors = []
            gt_imgs = []
            segmaps = []
            torso_imgs = []
            drv_eye_rects = []
            kp_src = []
            kp_drv = []
            mouth_masks_for_batch = []
            eye_masks_for_batch = []
            for di in drv_idx:
                # 读取target image
                if self.torso_mode:
                    if self.ds['com_imgs'][di] is None:
                        img_name = f'data/processed/videos/{video_id}/com_imgs/{format(di, "08d")}.jpg'
                        img = torch.tensor(cv2.imread(img_name)[..., ::-1] / 127.5 - 1).permute(2,0,1).float() # [3, H, W]
                        self.ds['com_imgs'][di] = img
                    gt_imgs.append(self.ds['com_imgs'][di])
                else:
                    if self.ds['head_imgs'][di] is None:
                        img_name = f'data/processed/videos/{video_id}/head_imgs/{format(di, "08d")}.png'
                        img = torch.tensor(cv2.imread(img_name)[..., ::-1] / 127.5 - 1).permute(2,0,1).float() # [3, H, W]
                        self.ds['head_imgs'][di] = img
                    gt_imgs.append(self.ds['head_imgs'][di])
                # 使用第一帧的torso作为face v2v的输入 
                if self.ds['torso_imgs'][0] is None:
                    img_name = f'data/processed/videos/{video_id}/inpaint_torso_imgs/{format(0, "08d")}.png'
                    img = torch.tensor(cv2.imread(img_name)[..., ::-1] / 127.5 - 1).permute(2,0,1).float() # [3, H, W]
                    self.ds['torso_imgs'][0] = img
                torso_imgs.append(self.ds['torso_imgs'][0])
                # 所以segmap也用第一帧的了
                if self.ds['segmaps'][0] is None:
                    img_name = f'data/processed/videos/{video_id}/segmaps/{format(0, "08d")}.png'
                    seg_img = cv2.imread(img_name)[:,:, ::-1]
                    segmap = torch.from_numpy(decode_segmap_mask_from_image(seg_img)) # [6, H, W]
                    self.ds['segmaps'][0] = segmap
                segmaps.append(self.ds['segmaps'][0])
                
                # Get the pre-computed mouth mask for the current frame
                mouth_masks_for_batch.append(self.ds['lip_masks'][di])
                eye_masks_for_batch.append(self.ds['eye_masks'][di])

                drv_eye_rects.append(self.ds['eye_rects'][di])
                kp_src.append(self.ds['kps'][0])
                kp_drv.append(self.ds['kps'][di])
            bg_img = self.ds['bg_img'].unsqueeze(0).repeat([batch_size, 1, 1, 1]).cuda()
            ref_torso_imgs = torch.stack(torso_imgs).float().cuda()
            kp_src = torch.stack(kp_src).float().cuda()
            kp_drv = torch.stack(kp_drv).float().cuda()
            segmaps = torch.stack(segmaps).float().cuda()
            tgt_imgs = torch.stack(gt_imgs).float().cuda()
            mouth_masks_for_batch = torch.stack(mouth_masks_for_batch).float().cuda()
            eye_masks_for_batch = torch.stack(eye_masks_for_batch).float().cuda()
            for di in drv_idx:
                _, secc_color = self.secc_renderer(self.ds['id'][0:1], drv_exps[di:di+1], zero_eulers[0:1], zero_trans[0:1])
                drv_secc_colors.append(secc_color)
            drv_secc_color = torch.cat(drv_secc_colors)
            cano_secc_color = self.ds['cano_secc_color'].repeat([batch_size, 1, 1, 1])
            src_secc_color = self.ds['src_secc_color'].repeat([batch_size, 1, 1, 1])
            cond = {'cond_cano': cano_secc_color,'cond_src': src_secc_color, 'cond_tgt': drv_secc_color,
                    'ref_torso_img': ref_torso_imgs, 'bg_img': bg_img, 'segmap': segmaps,
                    'kp_s': kp_src, 'kp_d': kp_drv, 'mouth_mask': mouth_masks_for_batch}
            if self.inp.get('mouth_encode_mode', 'none') != 'none' and self.ds['mouth_ref_img'] is not None:
                cond['mouth_ref_img'] = self.ds['mouth_ref_img'].unsqueeze(0).repeat([batch_size, 1, 1, 1])
            camera = self.ds['cameras'][drv_idx]
            gen_output = self.secc2video_model.forward(img=None, camera=camera, cond=cond, ret={}, cache_backbone=False, use_cached_backbone=True)
            pred_img = gen_output['image']
            
            # Save mask videos by applying them to the ground truth image
            lip_mask_applied = tgt_imgs.clone()
            lip_mask_applied[self.ds['lip_masks'][i:i+1].cuda().repeat(1,3,1,1) == 0] = -1
            lip_mask_frame = ((lip_mask_applied[0].permute(1, 2, 0) + 1) / 2 * 255).int().cpu().numpy().astype(np.uint8)

            eye_mask_applied = tgt_imgs.clone()
            eye_mask_applied[self.ds['eye_masks'][i:i+1].cuda().repeat(1,3,1,1) == 0] = -1
            eye_mask_frame = ((eye_mask_applied[0].permute(1, 2, 0) + 1) / 2 * 255).int().cpu().numpy().astype(np.uint8)

            lip_mask_video_writer.append_data(lip_mask_frame)
            eye_mask_video_writer.append_data(eye_mask_frame)

            # Create and write masked components debug frame
            combined_mask = (self.ds['lip_masks'][i:i+1] + self.ds['eye_masks'][i:i+1]).clamp(0, 1).cuda()
            gt_masked = tgt_imgs * combined_mask
            pred_masked = pred_img * combined_mask
            
            # Make unmasked area black for visualization
            expanded_mask = combined_mask.repeat(1, 3, 1, 1)
            gt_masked[expanded_mask == 0] = -1
            pred_masked[expanded_mask == 0] = -1

            combined_masked_frame = torch.cat([gt_masked, pred_masked], dim=-1)
            combined_masked_frame_np = ((combined_masked_frame[0].permute(1, 2, 0) + 1) / 2 * 255).int().cpu().numpy().astype(np.uint8)
            masked_comp_video_writer.append_data(combined_masked_frame_np)

            pred_img = ((pred_img.permute(0, 2, 3, 1) + 1)/2 * 255).int().cpu().numpy().astype(np.uint8)
            video_writer.append_data(pred_img[0])
        video_writer.close()
        masked_comp_video_writer.close()
        lip_mask_video_writer.close()
        eye_mask_video_writer.close()
        self.model.train()
        # Optional unseen-audio validation
        self.audio_validation(inp, step)
        print("| audio validation done")

    def masked_error_loss(self, img_pred, img_gt, mask, unmasked_weight=0.1, mode='l1'):
        # 对raw图像，因为deform的原因背景没法全黑，导致这部分mse过高，我们将其mask掉，只计算人脸部分
        masked_weight = 1.0
        weight_mask = mask.float() * masked_weight + (~mask).float() * unmasked_weight
        if mode == 'l1':
            error = (img_pred - img_gt).abs().sum(dim=1) * weight_mask
        else:
            error = (img_pred - img_gt).pow(2).sum(dim=1) * weight_mask
        error.clamp_(0, max(0.5, error.quantile(0.8).item())) # clamp掉较高loss的pixel，避免姿态没对齐的pixel导致的异常值占主导影响训练
        loss = error.mean()
        return loss
    
    def dilate(self, bin_img, ksize=5, mode='max_pool'):
        """
        mode: max_pool or avg_pool
        """
        # bin_img, [1, h, w]
        pad = (ksize-1)//2
        bin_img = F.pad(bin_img, pad=[pad,pad,pad,pad], mode='reflect')
        if mode == 'max_pool':
            out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
        else:
            out = F.avg_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
        return out
        
    def dilate_mask(self, mask, ksize=21):
        mask = self.dilate(mask, ksize=ksize, mode='max_pool')
        return mask
    
    def set_unmasked_to_black(self, img, mask):
        out_img = img * mask.float() - (~mask).float() # -1 denotes black
        return out_img
    
    def dump_checkpoint(self, inp):
        checkpoint = {}
        # save optimizers
        optimizer_states = []
        self.optimizers = [self.optimizer]
        for i, optimizer in enumerate(self.optimizers):
            if optimizer is not None:
                state_dict = optimizer.state_dict()
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                optimizer_states.append(state_dict)
        checkpoint['optimizer_states'] = optimizer_states
        state_dict = {
            'model': self.model.state_dict(),
            'learnable_triplane': self.model.state_dict()['_last_cano_planes'],
        }
        del state_dict['model']['_last_cano_planes']
        
        # Add discriminators to checkpoint if they exist
        if self.use_adv_loss:
            if self.D_full: state_dict['D_full'] = self.D_full.state_dict()
            if self.D_lip: state_dict['D_lip'] = self.D_lip.state_dict()
            if self.D_eye: state_dict['D_eye'] = self.D_eye.state_dict()
            if self.optimizer_D_full: optimizer_states.append(self.optimizer_D_full.state_dict())
            if self.optimizer_D_lip: optimizer_states.append(self.optimizer_D_lip.state_dict())
            if self.optimizer_D_eye: optimizer_states.append(self.optimizer_D_eye.state_dict())

        checkpoint['state_dict'] = state_dict
        checkpoint['lora_args'] = self.lora_args
        person_ds = {}
        video_id = inp['video_id']
        img_name = f'data/processed/videos/{video_id}/gt_imgs/{format(0, "08d")}.jpg'
        gt_img = torch.tensor(cv2.resize(cv2.imread(img_name), (512, 512))[..., ::-1] / 127.5 - 1).permute(2,0,1).float() # [3, H, W]
        person_ds['gt_img'] = gt_img.reshape([1, 3, 512, 512])
        person_ds['id'] = self.ds['id'][0:1].cpu()
        person_ds['src_kp'] = self.ds['kps'][0].cpu()
        person_ds['video_id'] = inp['video_id']
        checkpoint['person_ds'] = person_ds
        
        # Save person_ds to a separate file for inference pipeline
        person_ds_path = os.path.join(inp['work_dir'], 'person_ds.npy')
        np.save(person_ds_path, person_ds)
        print(f"| Saved person_ds to {person_ds_path}")

        return checkpoint

    def audio_validation(self, inp, step=''):
        """Run an optional unseen-audio validation using the AdaptGeneFace2Infer pipeline.
        This simply shells out to `mimictalk_infer.py` reusing the currently trained
        checkpoint as `--torso_ckpt` and saving the output into work_dir.
        """
        a2m_ckpt = inp.get('a2m_ckpt', '')
        val_audio = inp.get('val_audio', '')
        if a2m_ckpt == '' or val_audio == '':
            print("| Skip audio validation – a2m_ckpt or val_audio not provided.")
            return
            
        # Find the latest checkpoint in the current work directory
        latest_ckpt, _ = get_last_checkpoint(inp['work_dir'])
        if latest_ckpt is None:
            print(f"| Skip audio validation – no checkpoint found in {inp['work_dir']}.")
            return

        # Compose command for quick validation
        print(f"video_path: {inp['video_path']}")
        out_name = os.path.join(inp['work_dir'], f"val_audio_step{step}.mp4")
        cmd = (
            f"python inference/mimictalk_infer.py "
            f"--a2m_ckpt {a2m_ckpt} "
            f"--torso_ckpt \"{inp['work_dir']}\" " # Pass the specific checkpoint file
            f"--drv_aud \"{val_audio}\" "
            f"--drv_pose \"{inp['video_path'].replace('examples', 'videos')}\" "
            f"--drv_style \"{inp['video_path'].replace('examples', 'videos')}\" "
            f"--out_mode concat_debug "
            f"--out_name \"{out_name}\" "
            f"--map_to_init_pose True --temperature 0.3 --denoising_steps 20"
        )
        print(f"| Running audio validation: {cmd}")
        os.system(cmd)

        cmd = (
            f"python inference/mimictalk_infer.py "
            f"--a2m_ckpt {a2m_ckpt} "
            f"--torso_ckpt \"{inp['work_dir']}\" " # Pass the specific checkpoint file
            f"--drv_aud \"{val_audio}\" "
            f"--drv_pose \"{inp['video_path'].replace('examples', 'videos')}\" "
            f"--drv_style \"{inp['video_path'].replace('examples', 'videos')}\" "
            f"--out_mode final "
            f"--out_name \"{out_name.replace('.mp4','_full.mp4')}\" "
            f"--map_to_init_pose True --temperature 0.3 --denoising_steps 20"
        )
        print(f"| Running audio validation: {cmd}")
        os.system(cmd)

if __name__ == '__main__':
    import argparse, glob, tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument("--head_ckpt", default='') # checkpoints/0729_th1kh/secc_img2plane checkpoints/0720_img2planes/secc_img2plane_two_stage
    # parser.add_argument("--torso_ckpt", default='checkpoints/240210_real3dportrait_orig/secc2plane_torso_orig') # checkpoints/0729_th1kh/secc_img2plane checkpoints/0720_img2planes/secc_img2plane_two_stage
    parser.add_argument("--torso_ckpt", default='checkpoints/mimictalk_orig/os_secc2plane_torso') # checkpoints/0729_th1kh/secc_img2plane checkpoints/0720_img2planes/secc_img2plane_two_stage
    parser.add_argument("--video_id", default='data/raw/examples/id2.mp4', help="identity source, we support (1) already processed <video_id> of GeneFace, (2) video path, (3) image path")
    parser.add_argument("--work_dir", default=None) 
    parser.add_argument("--max_updates", default=4001, type=int, help="for video, 2000 is good; for an image, 3~10 is good") 
    parser.add_argument("--test", action='store_true') 
    parser.add_argument("--batch_size", default=1, type=int, help="batch size during training, 1 needs 8GB, 2 needs 15GB") 
    parser.add_argument("--lr", default=0.001, type=float) 
    parser.add_argument("--lr_triplane", default=0.005, type=float, help="for video, 0.1; for an image, 0.001; for ablation with_triplane, 0.") 
    parser.add_argument("--lora_r", default=64, type=int, help="width of lora unit") 
    parser.add_argument("--lora_mode", default='secc2plane_sr', help='for video, full; for an image, none')
    parser.add_argument("--offset_x", default=424, type=int, help="x offset for face cropping")
    parser.add_argument("--offset_y", default=60, type=int, help="y offset for face cropping")
    # New arguments for compatibility
    parser.add_argument("--blink_reg_loss_w", default=0.003, type=float, help="weight for blink regularization loss")
    parser.add_argument("--secc_reg_loss_w", default=0.01, type=float, help="weight for secc regularization loss")
    parser.add_argument("--lip_loss_w_mult", default=1.0, type=float, help="multiplier for lip losses")
    parser.add_argument("--head_loss_w_mult", default=1.0, type=float, help="multiplier for head losses")
    parser.add_argument("--random_seed", default=None, type=int, help="random seed for reproducibility")
    parser.add_argument("--preprocess_only", action='store_true', help="run preprocessing only and exit")
    parser.add_argument("--mouth_encode_mode", default='none', choices=['none', 'concat', 'add', 'style_latent', 'adain', 'gated', 'film'], help="choose the mouth feature injection mode for the SR module")
    parser.add_argument("--a2m_ckpt", default='checkpoints/audio2motion_vae', help="checkpoint directory of audio2secc model, if provided test_loop will run unseen audio validation")
    parser.add_argument("--val_audio", default='data/raw/examples/10s.wav', help="path to an unseen audio file for additional validation during training")
    # Advanced Loss arguments - New flexible system
    parser.add_argument("--adv_loss_fns", default="", type=str, help="Space-separated list of advanced loss functions to use (e.g., 'gan gan_lip fsm_lip comp_lp comp_lp_eye comp_lp_lip').")
    parser.add_argument("--lambda_gan", default=1.0, type=float, help="Generic weight for all GAN losses.")
    parser.add_argument("--lambda_fsm", default=10.0, type=float, help="Generic weight for all Feature Style Matching losses.")
    parser.add_argument("--lambda_comp_lp", default=1.0, type=float, help="Generic weight for all Facial Component Perceptual losses (comp_lp, comp_lp_eye, comp_lp_lip).")
    parser.add_argument("--lr_d", default=0.0001, type=float, help="learning rate for discriminators")
    parser.add_argument("--lambda_lpips", default=0.5, type=float, help="Weight for LPIPS (VGG) perceptual loss. Set to 0 to disable.")
    parser.add_argument("--lambda_sd", default=0.0, type=float, help="Weight for Stable Diffusion perceptual loss. Set to 0 to disable.")
    parser.add_argument("--generator", default='stylegan2', choices=['stylegan2', 'stylegan3'], help="Super-resolution generator architecture.")
    parser.add_argument("--large_sr", action='store_true', help="Use the larger super-resolution model.")
    parser.add_argument("--mouth_tuning_steps", default=0, type=int, help="Number of additional steps for mouth-focused tuning.")
    parser.add_argument("--l1", action='store_true', help="Use L1 loss instead of MSE for all pixel losses (head, lip, etc)")

    args = parser.parse_args()
    
    # Set random seed if provided
    if args.random_seed is not None:
        print(f"| Using random seed: {args.random_seed}")
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_seed)

    inp = {
            'head_ckpt': args.head_ckpt,
            'torso_ckpt': args.torso_ckpt,
            'video_id': args.video_id,
            'work_dir': args.work_dir,
            'max_updates': args.max_updates,
            'batch_size': args.batch_size,
            'test': args.test,
            'lr': float(args.lr),
            'lr_triplane': float(args.lr_triplane),
            'lora_mode': args.lora_mode,
            'lora_r': args.lora_r,
            'offset_x': args.offset_x,
            'offset_y': args.offset_y,
            # New arguments for compatibility
            'blink_reg_loss_w': args.blink_reg_loss_w,
            'secc_reg_loss_w': args.secc_reg_loss_w,
            'lip_loss_w_mult': args.lip_loss_w_mult,
            'head_loss_w_mult': args.head_loss_w_mult,
            'random_seed': args.random_seed,
            'preprocess_only': args.preprocess_only,
            'mouth_encode_mode': args.mouth_encode_mode,
            # Audio validation options
            'a2m_ckpt': args.a2m_ckpt,
            'val_audio': args.val_audio,
            # Advanced Loss arguments
            'adv_loss_fns': args.adv_loss_fns,
            'lambda_gan': args.lambda_gan,
            'lambda_fsm': args.lambda_fsm,
            'lambda_comp_lp': args.lambda_comp_lp,
            'lr_d': args.lr_d,
            'lambda_lpips': args.lambda_lpips,
            'lambda_sd': args.lambda_sd,
            'generator': args.generator,
            'large_sr': args.large_sr,
            'video_path': args.video_id,
            'mouth_tuning_steps': args.mouth_tuning_steps,
            'l1': args.l1,
            }
    if inp['work_dir'] == None:
        video_id = os.path.basename(inp['video_id'])[:-4] if inp['video_id'].endswith((".mp4", ".png", ".jpg", ".jpeg")) else inp['video_id']
        
        # Create a descriptive suffix based on enabled losses for automatic work_dir naming
        lora_suffix = f"_lora_{inp['lora_mode']}_rank_{inp['lora_r']}"
        
        mouth_mode_suffix = ""
        if inp['mouth_encode_mode'] != 'none':
            mouth_mode_suffix = f"_mouth_{inp['mouth_encode_mode']}"

        mouth_tuning_suffix = ""
        if inp['mouth_tuning_steps'] > 0:
            mouth_tuning_suffix = f"_mouth_tuning_{inp['mouth_tuning_steps']}"

        sr_suffix = "_largeSR" if inp['large_sr'] else ""

        l1_suffix = "_l1" if inp.get('l1', False) else ""

        loss_suffix_parts = []
        if inp['adv_loss_fns']:
            loss_suffix_parts.extend(sorted(inp['adv_loss_fns'].split()))
        if inp['lambda_sd'] > 0:
            loss_suffix_parts.append('sdloss')
        
        loss_suffix = ('_' + '_'.join(loss_suffix_parts)) if loss_suffix_parts else ''
        
        inp['work_dir'] = f'checkpoints_mimictalk/{video_id}{lora_suffix}{mouth_mode_suffix}{sr_suffix}{l1_suffix}{loss_suffix}{mouth_tuning_suffix}'
    os.makedirs(inp['work_dir'], exist_ok=True)
    trainer = LoRATrainer(inp)

    # Initialize the gating network bias if in 'gated' mode
    if inp['mouth_encode_mode'] == 'gated':
        # Use the first batch of mouth masks from the dataset for initialization
        sample_mask = trainer.ds['lip_masks'][0:1]
        trainer.secc2video_model.superresolution.initialize_gating_bias(sample_mask)

    if inp['preprocess_only']:
        print("| Preprocessing finished. Exiting as --preprocess_only is set.")
        exit(0)

    if inp['test']:
        trainer.test_loop(inp)
    else:
        trainer.training_loop(inp)
        trainer.test_loop(inp)
    print(" ")