import torch
from srresnet.models import *

def save_checkpoint(model, filepath):
    
    checkpoint = {'state_dict': model.state_dict()}

    torch.save(checkpoint, filepath)



def load_checkpoint(filepath, device):
    scaling_factor = 4 
    large_kernel_size = 9  
    small_kernel_size = 3  
    n_channels = 64  
    n_blocks = 16
    
    checkpoint = torch.load(filepath)
    model = SRResNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
                  n_channels=n_channels, n_blocks=n_blocks, scaling_factor=scaling_factor)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    return model