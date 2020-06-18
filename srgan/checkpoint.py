import torch


def save_checkpoint(generator, discriminator, filepath):
    checkpoint = {
        'upscale_factor': generator.upscale_factor,
        'image_channels': generator.image_channels,
        'residual_block_channels': generator.residual_block_channels,
        'num_residual_blocks': generator.num_residual_blocks,
        'generator': generator,
        'discriminator': discriminator,
    }

    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, device):
    checkpoint = torch.load(filepath, map_location=device)
    generator = checkpoint['generator']
    discriminator = checkpoint['discriminator']

    return generator, discriminator
