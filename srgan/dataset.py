from glob import glob
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import (
    Compose, RandomCrop, ToTensor,
    ToPILImage, Resize,
    RandomHorizontalFlip, RandomVerticalFlip)
from torchvision.transforms import (Lambda)


def max_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


mean = torch.FloatTensor([0.485, 0.456, 0.406])
std = torch.FloatTensor([0.229, 0.224, 0.225])
rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966])

img_mean = mean.view([3, 1, 1])
img_std = std.view([3, 1, 1])
batch_mean = mean.view([1, 3, 1, 1])
batch_std = std.view([1, 3, 1, 1])


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
        Lambda(imagenet_normalise),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor(),
        # normalize
    ])


def valid_hr_transform():
    return Compose([
        ToTensor(),
    ])


def valid_lr_transform(crop_size):
    return Compose([
        ToPILImage(),
        Resize(crop_size, interpolation=Image.BICUBIC),
        ToTensor(),
        Lambda(imagenet_normalise),
    ])


output_to_pil_image = Compose(
    [
        # add 1          => [0, 2]
        # divide by 2    => [0, 1]
        Lambda(lambda img: (img + 1.0) / 2.0),
        ToPILImage()
    ]
)


def imagenet_normalise(img):
    input_device = img.device
    if img.ndimension() == 3:
        img = (img - img_mean.to(input_device)) / img_std.to(input_device)
    elif img.ndimension() == 4:
        img = (img - batch_mean.to(input_device)) / batch_std.to(input_device)
    return img


output_to_imagenet = Compose(
    [
        # add 1          => [0, 2]
        # divide by 2    => [0, 1]
        Lambda(lambda img: (img + 1.0) / 2.0),
        Lambda(imagenet_normalise),
    ]
)


def rgb_to_y_channel(img):
    input_device = img.device
    x = torch.matmul(
        255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :],
        rgb_weights.to(input_device)
    )
    return (x / 255.0) + 16.0


output_to_y_channel = Compose(
    [
        # add 1          => [0, 2]
        # divide by 2    => [0, 1]
        Lambda(lambda img: (img + 1.0) / 2.0),
        Lambda(lambda img: rgb_to_y_channel)
    ]
)


class SRImageDataset(Dataset):
    def __init__(self, dataset_dir,
                 crop_size=100,
                 upscale_factor=4, is_valid=False):
        super().__init__()
        glob_path = str(Path(dataset_dir) / "*[.jpg, .png]")
        self.crop_size = max_crop_size(crop_size, upscale_factor)
        self.upscale_factor = upscale_factor

        self.image_filenames = glob(glob_path)
        self.is_valid = is_valid

    def __getitem__(self, index):
        pil_hr_image = Image.open(self.image_filenames[index]).convert("RGB")
        if self.is_valid:
            hr_image = valid_hr_transform()(pil_hr_image)
            crop_size = (int(pil_hr_image.height / 4),
                         int(pil_hr_image.width / 4))
            lr_image = valid_lr_transform(crop_size)(hr_image)
            lr_real_image = train_lr_transform(self.crop_size, self.upscale_factor)(hr_image)
            
            return lr_image, hr_image, lr_real_image
        else:
            hr_image = train_hr_transform(self.crop_size)(pil_hr_image)
            lr_image = train_lr_transform(self.crop_size, self.upscale_factor)(hr_image)

            return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)
