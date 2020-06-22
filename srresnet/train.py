from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from srresnet.checkpoint import *
from srresnet.dataset import *
from srresnet.loss import *
from srresnet.models import *
from srresnet.utils import AverageMeter

def train_model(model, optimizer,
                device,
                train_dataloader, filepath,
                epochs=100, save_freq=5):
    iterations = epochs * len(train_dataloader)
    progress = tqdm(total=iterations)
    content_loss_criterion = nn.MSELoss()
    
    losses_meter = AverageMeter("MSE loss")
    
    #psnr_meter = AverageMeter("PSNR")
    #ssim_meter = AverageMeter("SSIM")

    for epoch in range(epochs):
        model.train()
        for lr, hr in train_dataloader:
            lr, hr = lr.to(device), hr.to(device)
            batch_size = lr.shape[0]

            ###########################################################################################################
            # Train srresnet
            sr = model(lr)
            
            content_loss = content_loss_criterion(sr, hr)

            # Back-prop loss on Resnet
            optimizer.zero_grad()
            content_loss.backward()

            # Update resnet
            optimizer.step()

            losses_meter.update(content_loss.item(), batch_size)
            
            ###########################################################################################################

            del lr, hr, sr
            progress.update()
            
        print(epoch, losses_meter)
        if epoch % save_freq == 0:
            save_checkpoint(model, filepath)

    return model

def train(root_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    scaling_factor = 4 
    large_kernel_size = 9  
    small_kernel_size = 3  
    n_channels = 64  
    n_blocks = 16
    
    filepath = root_path / "checkpoints/resnet_checkpoint.pth"
    print(filepath)
    if filepath.exists():
        model= load_checkpoint(filepath, device)
    else:
        (root_path / "checkpoints/").mkdir(exist_ok=True)
        model = SRResNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
                  n_channels=n_channels, n_blocks=n_blocks, scaling_factor=scaling_factor).to(device)
        
    print(f"Resnet: {sum(param.numel() for param in model.parameters())}")
    
    train_dataset = SRImageDataset(
        dataset_dir= "data/train_data/",
        crop_size=96
    )

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=200,
        num_workers=4
    )

    print(f"Length of train loader: {len(train_dataloader)}")

    
    lr = 1e-4
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                              lr=lr)


    model = model.to(device)
    model = train_model(model, optimizer,
                        device=device,
                        train_dataloader=train_dataloader,
                        epochs=500, save_freq=5,
                        filepath=filepath)
