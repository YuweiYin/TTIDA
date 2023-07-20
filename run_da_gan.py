#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

from __future__ import print_function

import os
import sys
import time
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from img_clf.utils import set_seed

"""
[DCGAN](https://arxiv.org/abs/1511.06434)
Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
"""


class Generator(nn.Module):

    def __init__(self, args):
        super(Generator, self).__init__()

        self.ngpu = args.ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(args.nz, args.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(args.ngf * 8, args.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(args.ngf * 4, args.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(args.ngf * 2, args.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(args.ngf, args.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):

    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.ngpu = args.ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(args.nc, args.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(args.ndf, args.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(args.ndf * 2, args.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(args.ndf * 4, args.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(args.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


def weights_init(m):
    # custom weights initialization called on netG and netD
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def run_train_gan(args, category: str) -> None:
    set_seed(int(args.seed))

    args.category_root = os.path.join(args.dataroot, category)
    args.output_img_dir = os.path.join(args.output_dir, args.data, category, "img_train/")
    args.output_ckpt_dir = os.path.join(args.output_dir, args.data, category, "ckpt/")
    os.makedirs(args.output_img_dir, exist_ok=True)
    os.makedirs(args.output_ckpt_dir, exist_ok=True)

    logger.info(f">>> run_train_gan: {args.category_root}")

    # ------------------------------
    # Create the dataset
    # ------------------------------
    dataset = dset.ImageFolder(
        root=args.category_root,
        transform=transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    )
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # Decide which device we want to run on
    # device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    has_cuda = torch.cuda.is_available()
    device = torch.device("cpu" if not has_cuda else "cuda")

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig(os.path.join(args.output_img_dir, "real_images.png"))

    # ------------------------------
    # Create the generator
    # ------------------------------
    netG = Generator(args).to(device)
    if device.type == "cuda" and args.ngpu > 1:  # Handle multi-gpu if desired
        netG = nn.DataParallel(netG, list(range(args.ngpu)))
    netG.apply(weights_init)  # weights_init: to randomly initialize all weights to mean=0, stdev=0.02.
    logger.info(netG)  # Print the model

    # ------------------------------
    # Create the Discriminator
    # ------------------------------
    netD = Discriminator(args).to(device)
    if device.type == "cuda" and args.ngpu > 1:  # Handle multi-gpu if desired
        netD = nn.DataParallel(netD, list(range(args.ngpu)))
    netD.apply(weights_init)  # weights_init: to randomly initialize all weights to mean=0, stdev=0.2.
    logger.info(netD)  # Print the model

    # ------------------------------
    # Loss Functions and Optimizers
    # ------------------------------
    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # ------------------------------
    # Training Loop
    # ------------------------------

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    # only save the GAN model with the smallest (G_loss + D_loss)
    smallest_G_loss = float("inf")
    smallest_D_loss = float("inf")
    save_gap = 50
    gen_gap = 500

    def plot_fake_images(fake_img_batch, batch_idx: str, iter_idx: str):
        # Plot the fake images from the last epoch
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(fake_img_batch, (1, 2, 0)))
        # plt.show()
        plt.savefig(os.path.join(args.output_img_dir, f"fake_images_training_epoch_{batch_idx}_{iter_idx}.png"))

    logger.info("Starting DCGAN Training Loop...")
    for epoch in range(args.num_epochs):  # For each epoch
        for i, data in enumerate(dataloader, 0):  # For each batch in the dataloader

            # ------------------------------------------------------------
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # ------------------------------------------------------------
            # # Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            # ------------------------------------------------------------
            # (2) Update G network: maximize log(D(G(z)))
            # ------------------------------------------------------------
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                logger.info("[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                            % (epoch, args.num_epochs, i, len(dataloader),
                               errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            cur_G_loss = errG.item()
            cur_D_loss = errD.item()
            G_losses.append(cur_G_loss)
            D_losses.append(cur_D_loss)

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % gen_gap == 0) or ((epoch == args.num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                cur_fake_img_batch = vutils.make_grid(fake, padding=2, normalize=True)
                img_list.append(cur_fake_img_batch)
                # Plot the fake images in the current epoch
                plot_fake_images(cur_fake_img_batch, str(epoch), str(iters))

            if (iters % save_gap == 0) and (cur_G_loss + cur_D_loss) < (smallest_G_loss + smallest_D_loss):
                smallest_G_loss = cur_G_loss
                smallest_D_loss = cur_D_loss
                logger.info(f">>> save ckpt: epoch = {epoch}; iters = {iters}; "
                            f"cur_G_loss = {cur_G_loss}; cur_D_loss = {cur_D_loss}")
                # torch.save(netD.state_dict(), os.path.join(args.output_ckpt_dir, f"netD_smallest_GD_loss.pt"))
                torch.save(netG.state_dict(), os.path.join(args.output_ckpt_dir, f"netG_smallest_GD_loss.pt"))

                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                cur_fake_img_batch = vutils.make_grid(fake, padding=2, normalize=True)
                img_list.append(cur_fake_img_batch)
                # Plot the fake images in the current epoch
                plot_fake_images(cur_fake_img_batch, "best", "best")

            iters += 1

        # save model checkpoints (generator & discriminator) at the end of each epoch
        # torch.save(netD.state_dict(), os.path.join(args.output_ckpt_dir, f"netD_{epoch}.pt"))
        # torch.save(netG.state_dict(), os.path.join(args.output_ckpt_dir, f"netG_{epoch}.pt"))

    # save the last model checkpoints (generator & discriminator)
    # torch.save(netD.state_dict(), os.path.join(args.output_ckpt_dir, "netD_last.pt"))
    torch.save(netG.state_dict(), os.path.join(args.output_ckpt_dir, "netG_last.pt"))

    # show training losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(args.output_img_dir, "G_D_training_loss.png"))

    # ------------------------------
    # Visualization of G's progression
    # ------------------------------
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    HTML(ani.to_jshtml())

    # ------------------------------
    # Real Images vs. Fake Images
    # ------------------------------
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig(os.path.join(args.output_img_dir, "real_images.png"))

    # Plot the fake images from the last epoch
    plot_fake_images(img_list[-1], "last", "last")


def run_generate_gan(args, category: str) -> None:
    # set_seed(int(args.seed))  # enable randomness

    # Decide which device we want to run on
    # device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    has_cuda = torch.cuda.is_available()
    device = torch.device("cpu" if not has_cuda else "cuda")

    args.category_root = os.path.join(args.dataroot, category)
    args.output_img_dir = os.path.join(args.output_dir, args.data, category, "img_gen/")
    args.output_ckpt_dir = os.path.join(args.output_dir, args.data, category, "ckpt/")
    os.makedirs(args.output_img_dir, exist_ok=True)
    # os.makedirs(args.output_ckpt_dir, exist_ok=True)

    logger.info(f">>> run_generate_gan: {args.category_root}")

    # ------------------------------
    # Load the trained generator
    # ------------------------------
    netG = Generator(args)
    # weights_path_G = os.path.join(args.output_ckpt_dir, f"netG_{ckpt_epoch}.pt")
    # weights_path_G = os.path.join(args.output_ckpt_dir, "netG_last.pt")
    weights_path_G = os.path.join(args.output_ckpt_dir, "netG_smallest_GD_loss.pt")
    assert os.path.isfile(weights_path_G)
    netG.load_state_dict(torch.load(weights_path_G))
    netG = netG.to(device)
    if device.type == "cuda" and args.ngpu > 1:  # Handle multi-gpu if desired
        netG = nn.DataParallel(netG, list(range(args.ngpu)))
    logger.info(netG)  # Print the model

    # ------------------------------
    # Load the trained Discriminator
    # ------------------------------
    # netD = Discriminator(args).to(device)
    # # weights_path_D = os.path.join(args.output_ckpt_dir, f"netD_{ckpt_epoch}.pt")
    # weights_path_D = os.path.join(args.output_ckpt_dir, "netD_smallest_GD_loss.pt")
    # assert os.path.isfile(weights_path_D)
    # netD.load_state_dict(torch.load(weights_path_D))
    # netD = netD.to(device)
    # if device.type == "cuda" and args.ngpu > 1:  # Handle multi-gpu if desired
    #     netD = nn.DataParallel(netD, list(range(args.ngpu)))
    # logger.info(netD)  # Print the model

    # Create batch of latent vectors that we will use to visualize the progression of the generator
    # fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    # ------------------------------
    # Generating Process
    # ------------------------------
    # img_list = []

    def plot_fake_images(fake_img_batch, batch_idx: str):
        # Plot the fake images from the last epoch
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(fake_img_batch, (1, 2, 0)))
        # plt.show()
        plt.savefig(os.path.join(args.output_img_dir, f"fake_images_generating_{batch_idx}.png"))

    if hasattr(args, "n_gen_batch") and isinstance(args.n_gen_batch, int) and args.n_gen_batch > 0:
        n_gen_batch = args.n_gen_batch
    else:
        n_gen_batch = 1

    logger.info(f"Starting DCGAN Generating... {category}")
    with torch.no_grad():
        save_img_idx = 0
        save_img_batch_idx = 0

        for gen_idx in range(n_gen_batch):  # 64 images per batch
            cur_noise = torch.randn(64, args.nz, 1, 1, device=device)
            fake = netG(cur_noise).detach().cpu()  # (64, 3, 64, 64)

            cur_fake_img_batch = vutils.make_grid(fake, padding=2, normalize=True)
            plot_fake_images(cur_fake_img_batch, str(save_img_batch_idx))
            save_img_batch_idx += 1

            im_batch = fake.numpy()
            # img_dim = int(args.image_size)
            for im in im_batch:
                im = ((im - im.min()) / (im.max() - im.min()) * 255).astype("uint8")  # normalize
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im, mode="RGB")
                im.save(os.path.join(args.output_img_dir, f"{save_img_idx}.png"))
                save_img_idx += 1


def main() -> bool:
    timer_start = time.process_time()

    parser = argparse.ArgumentParser()

    parser.add_argument("--cuda", type=str, default="0", help="Specify which device to use")
    parser.add_argument("--ngpu", type=int, default=1, help="Number of GPUs available. Use 0 for CPU mode.")
    parser.add_argument("--seed", type=int, default=42, help="The seed for all random modules")
    parser.add_argument("--data", type=str, default="cifar100", help="Dataset Root directory")
    parser.add_argument("--output_dir", type=str, default="./outputs/gan/",
                        help="The directory to store the GAN outputs")

    parser.add_argument("--workers", type=int, default=2, help="Number of workers for dataloader")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size during training")
    parser.add_argument("--image_size", type=int, default=64,
                        help="Spatial size of training images. "
                             "All images will be resized to this size using a transformer.")
    parser.add_argument("--nc", type=int, default=3, help="Number of channels in the training images. Color images: 3")
    parser.add_argument("--nz", type=int, default=100, help="Size of z latent vector (i.e. size of generator input)")
    parser.add_argument("--ngf", type=int, default=64, help="Size of feature maps in generator")
    parser.add_argument("--ndf", type=int, default=64, help="Size of feature maps in discriminator")
    # parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizers")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 hyperparameter for Adam optimizers")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 hyperparameter for Adam optimizers")

    parser.add_argument("--generate", action="store_true", default=False)
    parser.add_argument("--n_gen_batch", type=int, default=1, help="64 imager per batch")

    args = parser.parse_args()
    logger.info(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    # apply GAN on images of each category
    args.dataroot = os.path.join("./data", f"{args.data}_gan")
    assert os.path.isdir(args.dataroot)
    category_list = os.listdir(args.dataroot)
    category_list.sort()
    # args.category_root_list = [os.path.join(args.dataroot, category) for category in category_list]

    # for category in category_list:  # move all images into a deeper directory level
    #     category_dir = os.path.join(args.dataroot, category)
    #     category_img_dir = os.path.join(args.dataroot, category, category)
    #     os.makedirs(category_img_dir, exist_ok=True)
    #     todo_command = f"mv {category_dir}/*.png {category_img_dir}/"
    #     logger.info(todo_command)
    #     os.system(todo_command)

    # os.makedirs(args.output_dir, exist_ok=True)

    for category in category_list:
        if hasattr(args, "generate") and isinstance(args.generate, bool) and args.generate:
            run_generate_gan(args, category)
        else:
            run_train_gan(args, category)

    # dataset = "cifar100"
    # source_dir = os.path.join(f"./outputs/gan/{dataset}/")
    # category_list = os.listdir(source_dir)
    # for category in category_list:
    #     target_dir = os.path.join(f"./data/syn_data/gan_base/{dataset}/{category}/")
    #     os.makedirs(target_dir, exist_ok=True)
    #     command = f"cp {os.path.join(source_dir, category, 'img_gen')}/* {target_dir}/"
    #     logger.info(command)
    #     os.system(command)

    timer_end = time.process_time()
    logger.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))

    return True


if __name__ == "__main__":
    """bash
    nohup python run_da_gan.py --cuda 0 --data cifar100 --num_epochs 100 2>&1 > run_gan_cifar100.log &
    """
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    sys.exit(main())
