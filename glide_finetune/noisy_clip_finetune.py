from typing import Tuple

import torch
from glide_text2im.clip.model_creation import CLIPModel
from glide_text2im.respace import SpacedDiffusion
from torch.nn.functional import cross_entropy
# from torchvision import transforms


def train_step(
        clip_model: CLIPModel,
        clip_diffusion: SpacedDiffusion,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        device: str,
):
    prompts, x_start = batch
    x_start.to(device)
    x_start.permute(0, 3, 1, 2)
    with torch.no_grad():
        timesteps = torch.randint(
            0, len(clip_diffusion.betas) - 1, (x_start.shape[0],), device=device
        )
        t_noise = torch.randn_like(x_start, device=device)
        x_t = clip_diffusion.q_sample(
            x_start.to(device), timesteps.to(device), noise=t_noise.to(device)
        ).to(device)
        text_features = clip_model.text_embeddings(prompts)
    image_features = clip_model.image_embeddings(x_t, timesteps)
    labels = torch.arange(x_start.shape[0], device=device)
    text_loss = cross_entropy(image_features, labels)
    image_loss = cross_entropy(text_features, labels) / 2
    loss = text_loss + image_loss
    return loss
