import os
from typing import Tuple
# from wandb import wandb

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from glide_text2im.respace import SpacedDiffusion
from glide_text2im.text2im_model import Text2ImUNet
from glide_finetune import glide_util, train_util


def base_train_step(
        glide_model: Text2ImUNet,
        glide_diffusion: SpacedDiffusion,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        device: str,
):
    """
    Perform a single training step.

        Args:
            glide_model: The model to train.
            glide_diffusion: The diffusion to use.
            batch: A tuple of (tokens, masks, reals) where
                - tokens is a tensor of shape (batch_size, seq_len),
                - masks is a tensor of shape (batch_size, seq_len) and
                - reals is a tensor of shape (batch_size, 3, side_x, side_y) normalized to [-1, 1]
            device: The device to use for getting model outputs and computing loss.
        Returns:
            The loss.
    """
    tokens, masks, reals = [x.to(device) for x in batch]
    timesteps = torch.randint(
        0, len(glide_diffusion.betas) - 1, (reals.shape[0],), device=device
    )
    noise = torch.randn_like(reals, device=device)
    x_t = glide_diffusion.q_sample(reals, timesteps, noise=noise).to(device)
    _, C = x_t.shape[:2]
    model_output = glide_model(
        x_t.to(device),
        timesteps.to(device),
        tokens=tokens.to(device),
        mask=masks.to(device),
    )
    epsilon, _ = torch.split(model_output, C, dim=1)
    return torch.nn.functional.mse_loss(epsilon, noise.to(device).detach())


def upsample_train_step(
        glide_model: Text2ImUNet,
        glide_diffusion: SpacedDiffusion,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        device: str,
):
    """
    Perform a single training step.

        Args:
            glide_model: The model to train.
            glide_diffusion: The diffusion to use.
            batch: A tuple of (tokens, masks, low_res, high_res) where 
                - tokens is a tensor of shape (batch_size, seq_len), 
                - masks is a tensor of shape (batch_size, seq_len) with dtype torch.bool
                - low_res is a tensor of shape (batch_size, 3, base_x, base_y), normalized to [-1, 1]
                - high_res is a tensor of shape (batch_size, 3, base_x*4, base_y*4), normalized to [-1, 1]
            device: The device to use for getting model outputs and computing loss.
        Returns:
            The loss.
    """
    tokens, masks, low_res_image, high_res_image = [x.to(device) for x in batch]
    timesteps = torch.randint(0, len(glide_diffusion.betas) - 1, (low_res_image.shape[0],), device=device)
    noise = torch.randn_like(high_res_image, device=device)  # Noise should be shape of output i think
    noised_high_res_image = glide_diffusion.q_sample(high_res_image, timesteps, noise=noise).to(device)
    _, C = noised_high_res_image.shape[:2]
    model_output = glide_model(
        noised_high_res_image.to(device),
        timesteps.to(device),
        low_res=low_res_image.to(device),
        tokens=tokens.to(device),
        mask=masks.to(device))
    epsilon, _ = torch.split(model_output, C, dim=1)
    return torch.nn.functional.mse_loss(epsilon, noise.to(device).detach())


def run_glide_finetune_epoch(
        glide_model: Text2ImUNet,
        glide_diffusion: SpacedDiffusion,
        glide_options: dict,
        dataloader: DataLoader,
        optimizer: Optimizer,
        sample_bs: int = 1,  # batch size for inference
        sample_gs: float = 4.0,  # guidance scale for inference
        sample_respacing: str = "100",  # respacing for inference
        prompt: str = "",  # prompt for inference, not training
        side_x: int = 64,
        side_y: int = 64,
        outputs_dir: str = "./outputs/glide_finetune_test/",
        checkpoints_dir: str = "./ckpt/checkpoint_glide_finetune/",
        device: str = "cpu",
        log_frequency: int = 100,
        wandb_run=None,
        gradient_accumulation_steps=1,
        epoch: int = 0,
        train_upsample: bool = False,
        upsample_factor=4,
        image_to_upsample='low_res_face.png',
):
    if train_upsample:
        train_step = upsample_train_step
    else:
        train_step = base_train_step

    glide_model.to(device)
    glide_model.train()
    log = {}
    train_idx_last = 0
    for train_idx, batch in enumerate(dataloader):
        train_idx_last = train_idx
        accumulated_loss = train_step(
            glide_model=glide_model,
            glide_diffusion=glide_diffusion,
            batch=batch,
            device=device,
        )
        accumulated_loss.backward()
        optimizer.step()
        glide_model.zero_grad()
        log = {**log, "iter": train_idx, "loss": accumulated_loss.item() / gradient_accumulation_steps}
        # Sample from the model
        if train_idx > 0 and train_idx % log_frequency == 0:
            print(f"loss: {accumulated_loss.item():.4f}")
            print(f"Sampling from model at iteration {train_idx}")
            samples = glide_util.sample(
                glide_model=glide_model,
                glide_options=glide_options,
                side_x=side_x,
                side_y=side_y,
                prompt=prompt,
                batch_size=sample_bs,
                guidance_scale=sample_gs,
                device=device,
                prediction_respacing=sample_respacing,
                image_to_upsample=image_to_upsample,
            )
            sample_save_path = os.path.join(outputs_dir, f"{train_idx}.png")
            train_util.pred_to_pil(samples).save(sample_save_path)
            # wandb_run.log(
            #     {
            #         **log,
            #         "iter": train_idx,
            #         "samples": wandb.Image(sample_save_path, caption=prompt),
            #     }
            # )
            # print(f"Saved sample {sample_save_path}")
            print(f"iter: {train_idx}; saved samples: {sample_save_path}; caption: {prompt}")
        # if train_idx % 10000 == 0 and train_idx > 0:
        #     # train_util.save_model(glide_model, checkpoints_dir, train_idx, epoch)
        #     save_path = os.path.join(checkpoints_dir, f"glide-finetune-epoch{epoch}_batch{train_idx}.pt")
        #     torch.save(glide_model.state_dict(), save_path)  # about 1.54 GB per checkpoint
        #     print(f"Saved checkpoint after epoch {epoch} batch {train_idx}: {save_path}")
        # wandb_run.log(log)
        print(log)

    print(f"Finished training of epoch {epoch}")
    # train_util.save_model(glide_model, checkpoints_dir, train_idx_last, epoch)
    save_path = os.path.join(checkpoints_dir, f"glide-finetune-epoch{epoch}_batch{train_idx_last}.pt")
    torch.save(glide_model.state_dict(), save_path)
    print(f"Saved final checkpoint after epoch {epoch} batch {train_idx_last}: {save_path}")
