import torch


def save_checkpoint(generator, discriminator, filepath):
    checkpoint = {
        'upscale_factor': generator.upscale_factor,
        'image_channels': generator.image_channels,
        'residual_block_channels': generator.residual_block_channels,
        'num_residual_blocks': generator.num_residual_blocks,
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict()
    }

    torch.save(checkpoint, filepath)


def load_checkpoint(generator, discriminator, filepath, device):
    checkpoint = torch.load(filepath, map_location=device)
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])

    return generator, discriminator
