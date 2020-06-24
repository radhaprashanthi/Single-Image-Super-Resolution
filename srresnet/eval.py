import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from srresnet.checkpoint import *
from srresnet.dataset import *
from srresnet.utils import AverageMeter

def convert_image(img, source, target):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
    
    if source == '[-1, 1]':
        img = (img + 1.0) / 2.0

    if target == 'y-channel':
        img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights) / 255. + 16.0

    return img


def evaluate(root_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Model checkpoints
    filepath = root_path / "checkpoints/resnet_checkpoint.pth"

    # Load model SRResNet
    srresnet = torch.load(filepath, device).to(device)
    srresnet.eval()
    model = srresnet
    
    test_dataset = SRImageDataset(
        dataset_dir= root_path/"sr/DIV2K_valid_HR",
        crop_size=0,is_valid=True
    )

    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=1,
        num_workers=4
    )

    print(f"Length of train loader: {len(test_dataloader)}")

    # Keep track of the PSNRs and the SSIMs across batches
    PSNRs = AverageMeter("PSNR")
    SSIMs = AverageMeter("SSIM")

    # Batches
    for i, (lr, hr, lr_real) in enumerate(test_loader):
        # Move to default device
        lr = lr.to(device)  
        hr = hr.to(device) 

        # Forward prop.
        sr = model(lr) 

        sr_y = convert_image(sr.unsqueeze(0), source='[-1, 1]', target='y-channel').squeeze(0)
        hr_y = convert_image(hr.unsqueeze(0), source='[-1, 1]', target='y-channel').squeeze(0)

        hr_y = hr_y.detach().cpu().numpy()
        sr_y = sr_y.detach().cpu().numpy()
        
        # Calculate PSNR and SSIM
        psnr = peak_signal_noise_ratio(hr_y, sr_y, data_range=255.)
        ssim = structural_similarity(hr_y, sr_y,data_range=255.)
        
        PSNRs.update(psnr, lr.size(0))
        SSIMs.update(ssim, lr.size(0))

    # Print average PSNR and SSIM
    print('PSNR - {psnrs.avg:.3f}'.format(psnrs=PSNRs))
    print('SSIM - {ssims.avg:.3f}'.format(ssims=SSIMs))

    print("\n")
