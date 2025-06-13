"""
用于推理 inference/train_mimictalk_on_a_video.py 得到的person-specific模型
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# import librosa
import random
import time
import numpy as np
import importlib
import tqdm
import copy
import cv2
import lpips
from utils.commons.meters import AvgrageMeter
meter = AvgrageMeter()
from torch.utils.tensorboard import SummaryWriter
from kornia.color import rgb_to_hsv, hsv_to_rgb
from inference.infer_utils import smooth_features_xd_gaussian

# common utils
from utils.commons.hparams import hparams, set_hparams
from utils.commons.tensor_utils import move_to_cuda, convert_to_tensor
from utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint
# 3DMM-related utils
from deep_3drecon.deep_3drecon_models.bfm import ParametricFaceModel
from data_util.face3d_helper import Face3DHelper
from data_gen.utils.process_image.fit_3dmm_landmark import fit_3dmm_for_a_image
from data_gen.utils.process_video.fit_3dmm_landmark import fit_3dmm_for_a_video
from deep_3drecon.secc_renderer import SECC_Renderer
from data_gen.eg3d.convert_to_eg3d_convention import get_eg3d_convention_camera_pose_intrinsic
# Face Parsing 
from data_gen.utils.mp_feature_extractors.mp_segmenter import MediapipeSegmenter
from data_gen.utils.process_video.extract_segment_imgs import inpaint_torso_job, extract_background
# other inference utils
from inference.infer_utils import mirror_index, load_img_to_512_hwc_array, load_img_to_normalized_512_bchw_tensor
from inference.infer_utils import smooth_camera_sequence, smooth_features_xd
from inference.edit_secc import blink_eye_for_secc, hold_eye_opened_for_secc
from inference.real3d_infer import GeneFace2Infer

def hsv_value_matching(pred_rgbs, region, blurred_mask, mask_threshold=0.1):
    """
    각 이미지별로 마스크 영역의 V 채널 통계를 계산하여 매칭합니다.
    (벡터화된 연산을 사용하여 배치 내 모든 이미지에 대해 동시에 계산)
    """
    # A. region, pred_rgbs를 HSV로 변환
    region_hsv = rgb_to_hsv(region)
    pred_hsv   = rgb_to_hsv(pred_rgbs)
    
    # B. V 채널 분리 (shape: (B, 1, H, W))
    region_v = region_hsv[:, 2:3, :, :]
    pred_v   = pred_hsv[:, 2:3, :, :]
    
    # C. 마스크 생성: (B, 1, H, W), mask_threshold 초과하는 부분만 선택
    mask = (blurred_mask > mask_threshold).float()
    
    # D. 배치 차원별로 마스크 영역의 픽셀 수 계산 (shape: (B, 1, 1, 1)로 맞춰줌)
    eps = 1e-5
    count = mask.sum(dim=(2, 3), keepdim=True)  # (B, 1, 1, 1)
    
    # E. 각 이미지별 마스크 영역에 대한 평균 계산
    region_mean = (region_v * mask).sum(dim=(2, 3), keepdim=True) / (count + eps)
    pred_mean   = (pred_v * mask).sum(dim=(2, 3), keepdim=True)   / (count + eps)
    
    # F. 각 이미지별 마스크 영역에 대한 분산/표준편차 계산
    region_var = ((region_v - region_mean) * mask).pow(2).sum(dim=(2, 3), keepdim=True) / (count + eps)
    pred_var   = ((pred_v   - pred_mean)   * mask).pow(2).sum(dim=(2, 3), keepdim=True) / (count + eps)
    region_std = region_var.sqrt()
    pred_std   = pred_var.sqrt()
    
    # G. 안정성을 위해 표준편차를 epsilon으로 클램핑
    pred_std_clamped   = torch.clamp(pred_std, min=eps)
    region_std_clamped = torch.clamp(region_std, min=eps)
    
    # H. Mean-Std 매칭 공식 적용: (src - src_mean) * (tgt_std / src_std) + tgt_mean
    scale = region_std_clamped / pred_std_clamped
    shift = region_mean - pred_mean * scale
    matched_pred_v = torch.clamp(pred_v * scale + shift, 0.0, 1.0)
    
    # I. 매칭된 V 채널을 원래의 pred_hsv에 반영 (H, S 채널은 그대로)
    pred_hsv_matched = torch.cat([
        pred_hsv[:, 0:1, :, :],  # Hue
        pred_hsv[:, 1:2, :, :],  # Saturation
        matched_pred_v         # 매칭된 Value
    ], dim=1)
    
    # J. HSV -> RGB 변환 후 반환
    pred_rgbs_matched = hsv_to_rgb(pred_hsv_matched)
    return pred_rgbs_matched

def _get_gauss1d(radius: int, device, gauss_cache):
    key = (radius, device)
    if key not in gauss_cache:
        sigma  = radius / 2.0
        x      = torch.arange(-radius, radius + 1, device=device)
        g      = torch.exp(-(x**2) / (2 * sigma**2))
        g      = (g / g.sum()).view(1, 1, 1, -1)
        gauss_cache[key] = (g, g.transpose(2, 3))
    return gauss_cache

def alpha_feathering_optimized(gauss_cache, 
                               mask: torch.Tensor,
                               feather_radius: int = 30,
                               blend_weight: float = 0.7,
                               temperature: float = 10.0):
    device = mask.device
    key = (feather_radius, device)
    gauss_cache = _get_gauss1d(feather_radius, device, gauss_cache)
    k_h, k_v = gauss_cache[key]

    fg, bg = mask, 1.0 - mask

    def _sep_blur(x):
        x = F.conv2d(x, k_v, padding=(feather_radius, 0), groups=x.size(1))
        return F.conv2d(x, k_h, padding=(0, feather_radius), groups=x.size(1))

    fg_dist = _sep_blur(fg)
    bg_dist = _sep_blur(bg)

    fg_w = torch.sigmoid(temperature * (fg_dist - bg_dist))

    return mask * (1.0 - blend_weight) + fg_w * blend_weight, gauss_cache

def create_mouth_mask_gpt(lm2ds, image_size=512):
    b = lm2ds.shape[0]
    device = lm2ds.device
    
    polygon_indices = [123, 147, 213, 138, 172, 136, 150, 149, 176, 148, 152, 
                    377, 400, 378, 379, 365, 397, 367, 433, 376, 352,
                    347, 197, 118]

    mask = torch.zeros((b, 1, image_size, image_size), dtype=torch.float32, device=device)

    for i in range(b):
        poly_pts = lm2ds[i, polygon_indices].detach().cpu().numpy().astype(np.float32)
        poly_pts = poly_pts.reshape((-1, 1, 2)).astype(np.int32)
        mask_np = np.zeros((image_size, image_size), dtype=np.uint8)
        cv2.fillPoly(mask_np, [poly_pts], 255)
        mask[i, 0] = torch.from_numpy(mask_np).to(device) / 255.0

    return mask

def vis_lm2ds_to_coordinate(lm2ds, hw=512):    
    lm2d = (lm2ds * hw).to(torch.float32)
    return lm2d

def combine_images_with_mouth_mask_gpt(gauss_cache, transformed_gt_imgs, pred_rgbs, coordinate_lm2ds, crop_positions=(424, 60)):
    mouth_mask = create_mouth_mask_gpt(coordinate_lm2ds)
    blurred_mask, gauss_cache = alpha_feathering_optimized(gauss_cache, mouth_mask, feather_radius=30, blend_weight=1)
    
    blurred_mask_3ch = blurred_mask.repeat(1, 3, 1, 1)
    start_x, start_y = crop_positions
    region = transformed_gt_imgs[:, :, start_y:start_y+512, start_x:start_x+512]
    
    matched_pred_rgbs = hsv_value_matching(pred_rgbs, region, blurred_mask, mask_threshold=0.1)
    combined_region = region * (1 - blurred_mask_3ch) + matched_pred_rgbs * blurred_mask_3ch

    result_img = transformed_gt_imgs.clone()
    result_img[:, :, start_y:start_y+512, start_x:start_x+512] = combined_region
    
    return result_img, gauss_cache

class AdaptGeneFace2Infer(GeneFace2Infer):
    def __init__(self, audio2secc_dir, head_model_dir, torso_model_dir, device=None, **kwargs):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.inp = kwargs.get('inp')
        self.audio2secc_dir = audio2secc_dir
        self.head_model_dir = head_model_dir
        self.torso_model_dir = torso_model_dir
        self.audio2secc_model = self.load_audio2secc(audio2secc_dir)
        self.secc2video_model = self.load_secc2video(head_model_dir, torso_model_dir)
        self.audio2secc_model.to(device).eval()
        self.secc2video_model.to(device).eval()
        self.seg_model = MediapipeSegmenter()
        self.secc_renderer = SECC_Renderer(512)
        self.face3d_helper = Face3DHelper(use_gpu=True, keypoint_mode='lm68')
        self.mp_face3d_helper = Face3DHelper(use_gpu=True, keypoint_mode='mediapipe')
        self.gauss_cache = {}
        self.previous_exp = None
        self.pose_video_frames = None
        self.gt_video_frames = None
        # self.camera_selector = KNearestCameraSelector()

    def load_secc2video(self, head_model_dir, torso_model_dir):
        if torso_model_dir != '':
            config_dir = torso_model_dir if os.path.isdir(torso_model_dir) else os.path.dirname(torso_model_dir)
            set_hparams(f"{config_dir}/config.yaml", print_hparams=False)
            
            # Manually override mouth_encode_mode if provided in inference args
            if self.inp and self.inp.get('mouth_encode_mode') is not None:
                hparams['mouth_encode_mode'] = self.inp['mouth_encode_mode']
                print(f"| Manually set mouth_encode_mode to: {hparams['mouth_encode_mode']}")

            hparams['htbsr_head_threshold'] = 1.0
            self.secc2video_hparams = copy.deepcopy(hparams)
            ckpt = get_last_checkpoint(torso_model_dir)[0]
            lora_args = ckpt.get("lora_args", None)
            from modules.real3d.secc_img2plane_torso import OSAvatarSECC_Img2plane_Torso
            model = OSAvatarSECC_Img2plane_Torso(self.secc2video_hparams, lora_args=lora_args)
            load_ckpt(model, f"{torso_model_dir}", model_name='model', strict=False)
            self.learnable_triplane = nn.Parameter(torch.zeros([1, 3, model.triplane_hid_dim*model.triplane_depth, 256, 256]).float().cuda(), requires_grad=True)
            load_ckpt(self.learnable_triplane, f"{torso_model_dir}", model_name='learnable_triplane', strict=True)
            model._last_cano_planes = self.learnable_triplane
            if head_model_dir != '':
                print("| Warning: Assigned --torso_ckpt which also contains head, but --head_ckpt is also assigned, skipping the --head_ckpt.")
        else:
            from modules.real3d.secc_img2plane_torso import OSAvatarSECC_Img2plane
            set_hparams(f"{head_model_dir}/config.yaml", print_hparams=False)
            ckpt = get_last_checkpoint(head_model_dir)[0]
            lora_args = ckpt.get("lora_args", None)
            self.secc2video_hparams = copy.deepcopy(hparams)
            model = OSAvatarSECC_Img2plane(self.secc2video_hparams, lora_args=lora_args)
            load_ckpt(model, f"{head_model_dir}", model_name='model', strict=True)
            self.learnable_triplane = nn.Parameter(torch.zeros([1, 3, model.triplane_hid_dim*model.triplane_depth, 256, 256]).float().cuda(), requires_grad=True)
            model._last_cano_planes = self.learnable_triplane
            load_ckpt(model._last_cano_planes, f"{head_model_dir}", model_name='learnable_triplane', strict=True)
        self.person_ds = ckpt['person_ds']
        return model

    def prepare_batch_from_inp(self, inp):
        """
        :param inp: {'audio_source_name': (str)}
        :return: a dict that contains the condition feature of NeRF
        """
        sample = {}
        # Process Driving Motion
        if inp['drv_audio_name'][-4:] in ['.wav', '.mp3']:
            self.save_wav16k(inp['drv_audio_name'])
            if self.audio2secc_hparams['audio_type'] == 'hubert':
                hubert = self.get_hubert(self.wav16k_name)
            elif self.audio2secc_hparams['audio_type'] == 'mfcc':
                hubert = self.get_mfcc(self.wav16k_name) / 100

            f0 = self.get_f0(self.wav16k_name)
            if f0.shape[0] > len(hubert):
                f0 = f0[:len(hubert)]
            else:
                num_to_pad = len(hubert) - len(f0)
                f0 = np.pad(f0, pad_width=((0,num_to_pad), (0,0)))
            t_x = hubert.shape[0]
            x_mask = torch.ones([1, t_x]).float() # mask for audio frames
            y_mask = torch.ones([1, t_x//2]).float() # mask for motion/image frames
            sample.update({
                'hubert': torch.from_numpy(hubert).float().unsqueeze(0).cuda(),
                'f0': torch.from_numpy(f0).float().reshape([1,-1]).cuda(),
                'x_mask': x_mask.cuda(),
                'y_mask': y_mask.cuda(),
                })
            sample['blink'] = torch.zeros([1, t_x, 1]).long().cuda()
            sample['audio'] = sample['hubert']
            sample['eye_amp'] = torch.ones([1, 1]).cuda() * 1.0
        elif inp['drv_audio_name'][-4:] in ['.mp4']:
            drv_motion_coeff_dict = fit_3dmm_for_a_video(inp['drv_audio_name'], save=False)
            drv_motion_coeff_dict = convert_to_tensor(drv_motion_coeff_dict)
            t_x = drv_motion_coeff_dict['exp'].shape[0] * 2
            self.drv_motion_coeff_dict = drv_motion_coeff_dict
        elif inp['drv_audio_name'][-4:] in ['.npy']:
            drv_motion_coeff_dict = np.load(inp['drv_audio_name'], allow_pickle=True).tolist()
            drv_motion_coeff_dict = convert_to_tensor(drv_motion_coeff_dict)
            t_x = drv_motion_coeff_dict['exp'].shape[0] * 2
            self.drv_motion_coeff_dict = drv_motion_coeff_dict

        # Face Parsing
        sample['ref_gt_img'] = self.person_ds['gt_img'].cuda()
        img = self.person_ds['gt_img'].reshape([3, 512, 512]).permute(1, 2, 0)
        img = (img + 1) * 127.5
        img = np.ascontiguousarray(img.int().numpy()).astype(np.uint8)
        segmap = self.seg_model._cal_seg_map(img)
        sample['segmap'] = torch.tensor(segmap).float().unsqueeze(0).cuda()
        head_img = self.seg_model._seg_out_img_with_segmap(img, segmap, mode='head')[0]
        sample['ref_head_img'] = ((torch.tensor(head_img) - 127.5)/127.5).float().unsqueeze(0).permute(0, 3, 1,2).cuda() # [b,c,h,w]
        inpaint_torso_img, _, _, _ = inpaint_torso_job(img, segmap)
        sample['ref_torso_img'] = ((torch.tensor(inpaint_torso_img) - 127.5)/127.5).float().unsqueeze(0).permute(0, 3, 1,2).cuda() # [b,c,h,w]
        
        if inp['bg_image_name'] == '':
            bg_img = extract_background([img], [segmap], 'knn')
        else:
            bg_img = cv2.imread(inp['bg_image_name'])
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            bg_img = cv2.resize(bg_img, (512,512))
        sample['bg_img'] = ((torch.tensor(bg_img) - 127.5)/127.5).float().unsqueeze(0).permute(0, 3, 1,2).cuda() # [b,c,h,w]

        # 3DMM, get identity code and camera pose
        image_name = f"data/raw/val_imgs/{self.person_ds['video_id']}_img.png"
        os.makedirs(os.path.dirname(image_name), exist_ok=True)
        cv2.imwrite(image_name, img[:,:,::-1])
        coeff_dict = fit_3dmm_for_a_image(image_name, save=False)
        coeff_dict['id'] = self.person_ds['id'].reshape([1,80]).numpy()

        assert coeff_dict is not None
        src_id = torch.tensor(coeff_dict['id']).reshape([1,80]).cuda()
        src_exp = torch.tensor(coeff_dict['exp']).reshape([1,64]).cuda()
        src_euler = torch.tensor(coeff_dict['euler']).reshape([1,3]).cuda()
        src_trans = torch.tensor(coeff_dict['trans']).reshape([1,3]).cuda()
        sample['id'] = src_id.repeat([t_x//2,1])

        # get the src_kp for torso model
        sample['src_kp'] = self.person_ds['src_kp'].cuda().reshape([1, 68, 3]).repeat([t_x//2,1,1])[..., :2] # [B, 68, 2]

        # get camera pose file
        random.seed(time.time())
        if inp['drv_pose_name'] in ['nearest', 'topk']:
            camera_ret = get_eg3d_convention_camera_pose_intrinsic({'euler': torch.tensor(coeff_dict['euler']).reshape([1,3]), 'trans': torch.tensor(coeff_dict['trans']).reshape([1,3])})
            c2w, intrinsics = camera_ret['c2w'], camera_ret['intrinsics']
            camera = np.concatenate([c2w.reshape([1,16]), intrinsics.reshape([1,9])], axis=-1)
            coeff_names, distance_matrix = self.camera_selector.find_k_nearest(camera, k=100)
            coeff_names = coeff_names[0] # squeeze
            if inp['drv_pose_name'] == 'nearest':
                inp['drv_pose_name'] = coeff_names[0]
            else:
                inp['drv_pose_name'] = random.choice(coeff_names)
            # inp['drv_pose_name'] = coeff_names[0]
        elif inp['drv_pose_name'] == 'random':
            inp['drv_pose_name'] = self.camera_selector.random_select()
        else:
            inp['drv_pose_name'] = inp['drv_pose_name']

        print(f"| To extract pose from {inp['drv_pose_name']}")

        # extract camera pose 
        if inp['drv_pose_name'] == 'static':
            sample['euler'] = torch.tensor(coeff_dict['euler']).reshape([1,3]).cuda().repeat([t_x//2,1]) # default static pose
            sample['trans'] = torch.tensor(coeff_dict['trans']).reshape([1,3]).cuda().repeat([t_x//2,1])
        else: # from file
            if inp['drv_pose_name'].endswith('.mp4'):
                # extract coeff from video
                drv_pose_coeff_dict = fit_3dmm_for_a_video(inp['drv_pose_name'], save=False)
            else:
                # load from npy
                drv_pose_coeff_dict = np.load(inp['drv_pose_name'], allow_pickle=True).tolist()
            print(f"| Extracted pose from {inp['drv_pose_name']}")
            eulers = convert_to_tensor(drv_pose_coeff_dict['euler']).reshape([-1,3]).cuda()
            trans = convert_to_tensor(drv_pose_coeff_dict['trans']).reshape([-1,3]).cuda()
            len_pose = len(eulers)
            index_lst = [mirror_index(i, len_pose) for i in range(t_x//2)]
            sample['euler'] = eulers[index_lst]
            sample['trans'] = trans[index_lst]

        # fix the z axis
        sample['trans'][:, -1] = sample['trans'][0:1, -1].repeat([sample['trans'].shape[0]])

        # mapping to the init pose
        if inp.get("map_to_init_pose", 'False') == 'True':
            diff_euler = torch.tensor(coeff_dict['euler']).reshape([1,3]).cuda() - sample['euler'][0:1]
            sample['euler'] = sample['euler'] + diff_euler
            diff_trans = torch.tensor(coeff_dict['trans']).reshape([1,3]).cuda() - sample['trans'][0:1]
            sample['trans'] = sample['trans'] + diff_trans

        # prepare camera
        camera_ret = get_eg3d_convention_camera_pose_intrinsic({'euler':sample['euler'].cpu(), 'trans':sample['trans'].cpu()})
        c2w, intrinsics = camera_ret['c2w'], camera_ret['intrinsics']
        # smooth camera
        camera_smo_ksize = 7
        camera = np.concatenate([c2w.reshape([-1,16]), intrinsics.reshape([-1,9])], axis=-1)
        camera = smooth_camera_sequence(camera, kernel_size=camera_smo_ksize) # [T, 25]
        camera = torch.tensor(camera).cuda().float()
        sample['camera'] = camera

        return sample

    def infer_once(self, inp):
        if inp.get('gt_video'):
            print("| Loading ground truth video for debug visualization...")
            cap = cv2.VideoCapture(inp['gt_video'])
            self.gt_video_frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                self.gt_video_frames.append(frame)
            cap.release()
            if not self.gt_video_frames:
                print(f"| WARNING: Could not load frames from {inp['gt_video']}. Will use static ref image.")
                self.gt_video_frames = None

        if inp['blending']:
            print("| Loading pose video for blending background...")
            cap = cv2.VideoCapture(inp['drv_pose_name'])
            self.pose_video_frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                self.pose_video_frames.append(frame)
            cap.release()
            if not self.pose_video_frames:
                print(f"| WARNING: Could not load frames from {inp['drv_pose_name']}. Blending will be disabled.")
                inp['blending'] = False
            else:
                 print(f"| Loaded {len(self.pose_video_frames)} frames from {inp['drv_pose_name']}")
        
        super().infer_once(inp)

    @torch.no_grad()
    def forward_audio2secc(self, batch, inp=None):
        """
        기존 real3d_infer.py의 forward_audio2secc에서 '말하는 스타일 참조' 로직을 제거하고,
        오직 입력 오디오로부터 표정(exp)을 생성하는 핵심 로직만 남긴 오버라이드 메소드.
        """
        # audio-to-exp
        ret = {}
        if self.use_icl_audio2motion and inp['drv_talking_style_name'].endswith(".mp4"):
            print(f"this is called")
            from inference.infer_utils import extract_audio_motion_from_ref_video
            ref_exp, ref_hubert, ref_f0 = extract_audio_motion_from_ref_video(inp['drv_talking_style_name'])
            self.audio2secc_model.add_sample_to_context(ref_exp, ref_hubert, ref_f0)
        
        if self.use_icl_audio2motion:
            # here
            pred = self.audio2secc_model.forward(batch, ret=ret,train=False, denoising_steps=inp['denoising_steps'], cond_scale=inp['cfg_scale'])
        else:
            pred = self.audio2secc_model.forward(batch, ret=ret,train=False, temperature=inp['temperature'], denoising_steps=inp['denoising_steps'], cond_scale=inp['cfg_scale'])
    
        if pred.shape[-1] == 144:
            id = ret['pred'][0][:,:80]
            exp = ret['pred'][0][:,80:]
        else:
            id = batch['id']
            exp = ret['pred'][0]
        
        if inp['smoothing']:
            if self.previous_exp is None:
                exp_start_to_smooth = exp[:10]
                exp_start_smoothed = smooth_features_xd_gaussian(exp_start_to_smooth, kernel_size=5, strength=0.6)
                exp[:10] = exp_start_smoothed
            else:
                to_smooth = torch.cat([self.previous_exp, exp[:10]], dim=0)
                smoothed = smooth_features_xd_gaussian(to_smooth, kernel_size=5, strength=0.6)
                exp[:5] = smoothed[5:10]

            self.previous_exp = exp[-5:].detach().clone()
        else:
            print(f"no smoothing")

        if inp.get('drv_motion_coeff_dict', None) is not None:
            pass

        batch['id'] = id
        batch['exp'] = exp
        batch = self.get_driving_motion(batch['id'], batch['exp'], batch['euler'], batch['trans'], batch, inp)
        return batch

    @torch.no_grad()
    def forward_secc2video(self, batch, inp=None):
        num_frames = len(batch['drv_secc'])
        camera = batch['camera']
        src_kps = batch['src_kp']
        drv_kps = batch['drv_kp']
        cano_secc_color = batch['cano_secc']
        src_secc_color = batch['src_secc']
        drv_secc_colors = batch['drv_secc']
        ref_img_gt = batch['ref_gt_img']
        ref_img_head = batch['ref_head_img']
        ref_torso_img = batch['ref_torso_img']
        bg_img = batch['bg_img']
        segmap = batch['segmap']
        
        # smooth torso drv_kp
        torso_smo_ksize = 7
        drv_kps = smooth_features_xd(drv_kps.reshape([-1, 68*2]), kernel_size=torso_smo_ksize).reshape([-1, 68, 2])

        # Prepare video writers
        import imageio
        import uuid
        temp_video_path = f'infer_out/tmp/{uuid.uuid1()}.mp4'
        os.makedirs(os.path.dirname(temp_video_path), exist_ok=True)
        video_writer = imageio.get_writer(temp_video_path, fps=25, format='FFMPEG', codec='h264')
        
        secc_writer = None
        if inp.get('save_secc_video', False):
            if inp['out_name'] != '':
                base, ext = os.path.splitext(inp['out_name'])
                secc_out_fname = base + '_secc.mp4'
            else:
                drv_name = os.path.basename(inp['drv_audio_name']).split('.')[0]
                secc_out_fname = os.path.join('infer_out', drv_name + '_secc.mp4')
            os.makedirs(os.path.dirname(secc_out_fname), exist_ok=True)
            secc_writer = imageio.get_writer(secc_out_fname, fps=25, format='FFMPEG', codec='h264')

        # Prepare for blending if enabled
        bg_frames_tensor = None
        if inp['blending'] and self.pose_video_frames:
            print("| Applying advanced blending with pose video background...")
            num_frames_to_render = num_frames
            len_pose_video = len(self.pose_video_frames)
            
            bg_frames_batch = []
            for i in range(num_frames_to_render):
                src_idx = mirror_index(i, len_pose_video)
                frame_bgr = self.pose_video_frames[src_idx]
                
                frame_resized = cv2.resize(frame_bgr, (1360, 764))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                bg_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().to(self.device) / 255.0
                bg_frames_batch.append(bg_tensor)
            
            bg_frames_tensor = torch.stack(bg_frames_batch)

        with torch.no_grad():
            for i in tqdm.trange(num_frames, desc="MimicTalk is rendering frames"):
                kp_src = torch.cat([src_kps[i:i+1].reshape([1, 68, 2]), torch.zeros([1, 68,1]).to(src_kps.device)],dim=-1)
                kp_drv = torch.cat([drv_kps[i:i+1].reshape([1, 68, 2]), torch.zeros([1, 68,1]).to(drv_kps.device)],dim=-1)
                cond={'cond_cano': cano_secc_color,'cond_src': src_secc_color, 'cond_tgt': drv_secc_colors[i:i+1].cuda(),
                        'ref_torso_img': ref_torso_img, 'bg_img': bg_img, 'segmap': segmap,
                        'kp_s': kp_src, 'kp_d': kp_drv}
                
                gen_output = self.secc2video_model.forward(img=None, camera=camera[i:i+1], cond=cond, ret={}, cache_backbone=False, use_cached_backbone=True)
                
                # Process and write frame by frame
                img = gen_output['image']

                if secc_writer:
                    secc_img = F.interpolate(drv_secc_colors[i:i+1], (512,512))
                    secc_img_to_save = ((secc_img.cpu() + 1) / 2).clamp(0, 1)
                    secc_out_img = (secc_img_to_save.permute(0, 2, 3, 1) * 255).int().cpu().numpy().astype(np.uint8)
                    secc_writer.append_data(secc_out_img[0])
                
                if inp['out_mode'] == 'concat_debug':
                    img_raw = gen_output['image_raw']
                    depth_img = gen_output['image_depth']
                    secc_img_debug = F.interpolate(drv_secc_colors[i:i+1], (512,512))
                    
                    # Prepare GT frame
                    if self.gt_video_frames:
                        len_gt_video = len(self.gt_video_frames)
                        src_idx = mirror_index(i, len_gt_video)
                        frame_bgr = self.gt_video_frames[src_idx]
                        frame_resized = cv2.resize(frame_bgr, (512, 512))
                        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                        gt_frame = (torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 127.5 - 1).unsqueeze(0)
                    else:
                        gt_frame = ref_img_gt.cpu() # Fallback
                    
                    # Prepare other debug columns
                    secc_img_for_debug = ((secc_img_debug.cpu() + 1) / 2).clamp(0, 1) * 2 - 1
                    
                    depth_img_for_debug = F.interpolate(depth_img, (512,512)).cpu()
                    depth_img_for_debug = depth_img_for_debug.repeat([1,3,1,1])
                    # Normalize and invert the depth map
                    normalized_depth = (depth_img_for_debug - depth_img_for_debug.min()) / (depth_img_for_debug.max() - depth_img_for_debug.min() + 1e-6)
                    inverted_depth = 1.0 - normalized_depth
                    depth_img_for_debug = (inverted_depth * 2 - 1).clamp(-1,1)

                    # Assemble in new order: [GT, SECC, Depth, RawHead, Final]
                    processed_frame = torch.cat([
                        gt_frame,
                        secc_img_for_debug,
                        depth_img_for_debug,
                        F.interpolate(img_raw, (512,512)).cpu(),
                        img.cpu()
                    ], dim=-1)
                elif inp['out_mode'] == 'final':
                    processed_frame = img.cpu()

                if bg_frames_tensor is not None:
                    coordinate_lm2ds = vis_lm2ds_to_coordinate(batch['drv_kp'][i:i+1], hw=512)
                    crop_positions = (inp['offset_x'], inp['offset_y'])
                    blended_frame, self.gauss_cache = combine_images_with_mouth_mask_gpt(
                self.gauss_cache, 
                        bg_frames_tensor[i:i+1],
                        (processed_frame + 1) / 2,
                coordinate_lm2ds,
                crop_positions=crop_positions
            )
                    processed_frame = (blended_frame * 2) - 1

                out_frame_np = ((processed_frame.clamp(-1,1) + 1) / 2 * 255).permute(0, 2, 3, 1).int().cpu().numpy().astype(np.uint8)
                video_writer.append_data(out_frame_np[0])

        video_writer.close()
        if secc_writer:
            secc_writer.close()
            print(f"| Saved driving SECC video to {secc_out_fname}")
        
        if inp['out_name'] == '':
            torso_ckpt_name = os.path.basename(os.path.normpath(inp['torso_ckpt']))
            audio_name = os.path.basename(inp['drv_audio_name']).split('.')[0]
            
            # Add suffix for debug mode to make it explicit
            suffix = '_debug' if inp['out_mode'] == 'concat_debug' else ''

            out_fname = os.path.join('infer_out', audio_name, f"{torso_ckpt_name}{suffix}.mp4")
        else:
            out_fname = inp['out_name']

        try:
            os.makedirs(os.path.dirname(out_fname), exist_ok=True)
        except: pass
        if inp['drv_audio_name'][-4:] in ['.wav', '.mp3']:
            cmd = f"/usr/bin/ffmpeg -i {temp_video_path} -i {self.wav16k_name} -y -r 25 -ar 16000 -c:v copy -c:a libmp3lame -pix_fmt yuv420p -b:v 2000k  -strict experimental -shortest {out_fname}"
            os.system(cmd)
            os.system(f"rm {temp_video_path}")
        else:
            ret = os.system(f"ffmpeg -i {temp_video_path} -i {inp['drv_audio_name']} -map 0:v -map 1:a -y -v quiet -shortest {out_fname}")
            if ret != 0: 
                os.system(f"mv {temp_video_path} {out_fname}")
        print(f"Saved at {out_fname}")
        return out_fname

if __name__ == '__main__':
    import argparse, glob, tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2m_ckpt", default='checkpoints/240112_icl_audio2secc_vox2_cmlr') # checkpoints/0727_audio2secc/audio2secc_withlm2d100_randomframe
    parser.add_argument("--head_ckpt", default='') # checkpoints/0729_th1kh/secc_img2plane checkpoints/0720_img2planes/secc_img2plane_two_stage
    parser.add_argument("--torso_ckpt", default='checkpoints_mimictalk/German_20s') 
    parser.add_argument("--bg_img", default='') # data/raw/val_imgs/bg3.png
    parser.add_argument("--drv_aud", default='data/raw/examples/80_vs_60_10s.wav')
    parser.add_argument("--drv_pose", default='data/raw/examples/id2_512.mp4') #this is used for add_sample_to_context, nearest | topk | random | static | vid_name
    parser.add_argument("--drv_style", default='data/raw/examples/id2_512.mp4') # nearest | topk | random | static | vid_name
    parser.add_argument("--blink_mode", default='period') # none | period
    parser.add_argument("--temperature", default=0.3, type=float) # nearest | random
    parser.add_argument("--denoising_steps", default=20, type=int) # nearest | random
    parser.add_argument("--cfg_scale", default=1.5, type=float) # nearest | random
    parser.add_argument("--out_name", default='') # nearest | random
    parser.add_argument("--out_mode", default='concat_debug') # concat_debug | debug | final 
    parser.add_argument("--hold_eye_opened", default='False') # concat_debug | debug | final 
    parser.add_argument("--map_to_init_pose", default='True') # concat_debug | debug | final 
    parser.add_argument("--seed", default=None, type=int) # random seed, default None to use time.time()
    parser.add_argument("--smoothing", action='store_true', help="Enable stateful smoothing between chunks.")
    parser.add_argument("--blending", action='store_true', help="Enable advanced blending with color correction.")
    parser.add_argument("--mouth_encode_mode", default=None, type=str, choices=['none', 'concat', 'add', 'style_latent', 'adain', 'gated', 'film'], help="Manually specify the mouth feature injection mode if it's missing from the config")
    parser.add_argument("--save_secc_video", action='store_true', help="Save a separate video of the driving SECC for debugging.")
    parser.add_argument("--gt_video", default=None, type=str, help="Optional ground truth video for the first column in debug mode.")
 
    args = parser.parse_args()

    inp = {
            'a2m_ckpt': args.a2m_ckpt,
            'head_ckpt': args.head_ckpt,
            'torso_ckpt': args.torso_ckpt,
            'bg_image_name': args.bg_img,
            'drv_audio_name': args.drv_aud,
            'drv_pose_name': args.drv_pose,
            'drv_talking_style_name': args.drv_style,
            'blink_mode': args.blink_mode,
            'temperature': args.temperature,
            'denoising_steps': args.denoising_steps,
            'cfg_scale': args.cfg_scale,
            'out_name': args.out_name,
            'out_mode': args.out_mode,
            'map_to_init_pose': args.map_to_init_pose,
            'hold_eye_opened': args.hold_eye_opened,
            'seed': args.seed,
            'smoothing': args.smoothing,
            'blending': args.blending,
            'mouth_encode_mode': args.mouth_encode_mode,
            'save_secc_video': args.save_secc_video,
            'gt_video': args.gt_video,
            }
    print(f"args.smoothing: {args.smoothing}")
    print(f"args.blending: {args.blending}")
    AdaptGeneFace2Infer.example_run(inp)