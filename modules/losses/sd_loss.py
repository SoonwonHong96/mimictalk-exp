import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

class SDPerceptualLoss(nn.Module):
    """
    Stable Diffusion as a Learned Perceptual a.k.a. LPIPS Loss.

    Reference: https://arxiv.org/abs/2304.02086 "The Surprising Effectiveness of Diffusion Models for Image Net-to-Net Translation"
    """
    def __init__(self, device='cuda', model_id="stabilityai/stable-diffusion-2-1-base", loss_weight=1.0):
        super().__init__()
        self.device = device
        self.loss_weight = loss_weight
        
        # Load the U-Net and VAE models
        print("| Loading Stable Diffusion for perceptual loss...")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(self.device)
        
        # Freeze the models
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        
        # Scheduler for adding noise
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        
        print("| Stable Diffusion perceptual loss loaded.")

    def get_features(self, x, t, text_embeds):
        """
        Get intermediate features from the U-Net.
        x: latents (B, 4, H//8, W//8)
        t: timestep
        """
        # We are interested in the features from the down-sampling blocks
        features = []
        for module in self.unet.down_blocks:
            x, res_samples = module(hidden_states=x, temb=t, encoder_hidden_states=text_embeds)
            features.extend(res_samples)
        return features

    def forward(self, pred_img, target_img):
        """
        pred_img, target_img: Images in range [-1, 1], shape (B, 3, H, W)
        """
        # We need a dummy text embedding for the conditional U-Net
        # An empty prompt is sufficient as we only care about the visual features.
        tokenizer = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
        text_encoder = CLIPTextModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K").to(self.device)
        text_input = tokenizer("", padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_embeds = text_encoder(text_input.input_ids.to(self.device))[0]
        
        # Encode images to latent space using the VAE
        # The VAE expects input in [0, 1] range, so we scale it.
        pred_latents = self.vae.encode((pred_img + 1) / 2).latent_dist.sample() * self.vae.config.scaling_factor
        target_latents = self.vae.encode((target_img + 1) / 2).latent_dist.sample() * self.vae.config.scaling_factor
        
        # Add noise to latents
        # We use a fixed, moderate timestep for consistency
        t = torch.tensor([self.num_train_timesteps // 2] * pred_latents.shape[0], device=self.device)
        noise = torch.randn_like(pred_latents)
        pred_latents_noisy = self.scheduler.add_noise(pred_latents, noise, t)
        target_latents_noisy = self.scheduler.add_noise(target_latents, noise, t)

        # Get U-Net features
        # Timestep needs to be embedded
        t_emb = self.unet.time_proj(t)
        t_emb = self.unet.time_embedding(t_emb)

        pred_features = self.get_features(pred_latents_noisy, t_emb, text_embeds)
        target_features = self.get_features(target_latents_noisy, t_emb, text_embeds)

        # Calculate L1 loss on features
        loss = 0.0
        for feat_p, feat_t in zip(pred_features, target_features):
            loss += F.l1_loss(feat_p, feat_t)
            
        return loss * self.loss_weight 