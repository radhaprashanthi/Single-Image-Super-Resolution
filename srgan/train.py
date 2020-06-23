# -*- coding: utf-8 -*-
"""01_srgan_srresnet_v0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1J1lpbAXgiKQgVUdf0E5h-uI8G24xVG1z

SRGAN

https://towardsdatascience.com/build-a-super-simple-gan-in-pytorch-54ba349920e4
"""
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# from google.colab import drive
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from srgan.checkpoint import *
from srgan.dataset import *
from srgan.loss import *
from srgan.models import *
from srgan.utils import *

# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('runs/srgan')

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def train_model(generator, optimizer_g,
                discriminator, optimizer_d,
                vgg_loss, device,
                train_dataloader, filepath,
                epochs=100, save_freq=5):
    iterations = epochs * len(train_dataloader)
    progress = tqdm(total=iterations)
    content_loss_criterion = nn.MSELoss()
    adversarial_loss_criterion = nn.BCEWithLogitsLoss()

    beta = 1e-3
    vgg_loss_meter = AverageMeter("VGG loss")
    g_add_loss_meter = AverageMeter("Adversarial loss(G)")
    d_add_loss_meter = AverageMeter("Adversarial loss(D)")

    psnr_meter = AverageMeter("PSNR")
    ssim_meter = AverageMeter("SSIM")
    grad_clip = 2

    best_loss = float("inf")

    for epoch in range(epochs):

        if epoch == epochs / 2:
            adjust_learning_rate(optimizer_g, 0.1)
            adjust_learning_rate(optimizer_d, 0.1)

        generator.train()
        for lr, hr in train_dataloader:
            lr, hr = lr.to(device), hr.to(device)
            batch_size = lr.shape[0]

            ###########################################################################################################
            # Train generator
            sr = generator(lr)
            sr = output_to_imagenet(sr)
            sr_vgg = vgg_loss(sr)
            hr_vgg = vgg_loss(hr).detach()
            fake = discriminator(sr)

            # how close is sr to hr?
            # does vgg19 think sr is similar to hr?
            content_loss = content_loss_criterion(sr_vgg, hr_vgg)

            # make-believe
            # can sr be faked as hr?
            adversarial_loss = adversarial_loss_criterion(fake,
                                                          torch.ones_like(fake))
            # use the above losses
            perceptual_loss = beta * adversarial_loss

            # Back-prop loss on Generator
            optimizer_g.zero_grad()
            perceptual_loss.backward()

            # Update generator
            optimizer_g.step()

            vgg_loss_meter.update(content_loss.item(), batch_size)
            g_add_loss_meter.update(adversarial_loss.item(), batch_size)

            ###########################################################################################################
            # Train discriminator
            real = discriminator(hr)
            fake = discriminator(sr.detach())

            # can discriminator identify fake and real?
            # x is image and y (is_real) flag
            adversarial_loss = (adversarial_loss_criterion(fake, torch.zeros_like(fake)) +
                                adversarial_loss_criterion(real, torch.ones_like(real)))

            # Back-prop loss on Discriminator
            optimizer_d.zero_grad()
            adversarial_loss.backward()
	    
            if grad_clip is not None:
                clip_gradient(optimizer_d, grad_clip)

            # Update discriminator
            optimizer_d.step()

            d_add_loss_meter.update(adversarial_loss.item(), batch_size)

            ###########################################################################################################

            del lr, hr, sr
            progress.update()
        print("\n======================\n")
        print(vgg_loss_meter)
        print(g_add_loss_meter)
        print(d_add_loss_meter)
        print("\n======================\n")
        save_checkpoint(generator, discriminator, filepath)
        best_path = Path(str(filepath).replace("v0", "best"))
        if perceptual_loss.item() < best_loss:
            print("better loss, saving the gan")
            best_loss = perceptual_loss.item()
            save_checkpoint(generator, discriminator, best_path)

    return generator


def train(root_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    upscale_factor = 4
    image_channels = 3
    residual_block_channels = 64
    num_residual_blocks = 16
    num_middle_blocks = 7

    generator = Generator(upscale_factor=upscale_factor,
                            image_channels=image_channels,
                            residual_block_channels=residual_block_channels,
                            num_residual_blocks=num_residual_blocks).to(device)
        
    discriminator = Discriminator(
            image_channels=image_channels,
            num_middle_blocks=num_middle_blocks).to(device)

    filepath = root_path / "model_checkpoint/aishu/srgan_v0.pth"
    best_path = Path(str(filepath).replace("v0", "best"))
    print(filepath, best_path)

    if best_path.exists():
        print("checkpoint exists... loading...")
        generator, discriminator = load_checkpoint(generator, discriminator, best_path, device)
    else:
        (root_path / "model_checkpoint/aishu/").mkdir(parents=True, exist_ok=True)
    vgg_loss = VGG19Loss(i=5, j=4).to(device).eval() 

    print(f"Generator: {sum(param.numel() for param in generator.parameters())}")
    print(f"Discriminator: {sum(param.numel() for param in discriminator.parameters())}")

    train_dataset = SRImageDataset(
        dataset_dir=root_path / "training_data",
        crop_size=96
    )

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=200,
        num_workers=4
    )

    print(f"Length of train loader: {len(train_dataloader)}")

    
    lr = 1e-4
    optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, generator.parameters()),
                                   lr=lr)
    optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, discriminator.parameters()),
                                   lr=lr)

    generator = train_model(generator=generator,
                            optimizer_g=optimizer_g,
                            discriminator=discriminator,
                            optimizer_d=optimizer_d,
                            device=device,
                            vgg_loss=vgg_loss,
                            train_dataloader=train_dataloader,
                            epochs=500, save_freq=5,
                            filepath=filepath)
