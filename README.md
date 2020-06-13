# Single-Image-Super-Resolution
Single Image Super-Resolution Using Deep Neural Networks in PyTorch

### Background and motivation
During videos calls over low-bandwidth connections or poor receptions, the video appears to be pixelated and or reduced in resolution. We wanted to explore ways using Deep Learning to make clearer images from low-resolution images. This has numerous applications like satellite and aerial image analysis, medical image processing, compressed image/video enhancement etc.

### Project Objectives
The aim of our project is to recover or restore a high resolution (HR) image from a low resolution (LR) image such that texture detail in the reconstructed Super Resolution (SR) image is not lost. 

### Requirements
1. Python 3.6
2. Google Colab/AWS cloud
3. PyTorch

### Datasets
1. DIV2k data set - https://data.vision.ee.ethz.ch/cvl/DIV2K/. The data set has 1000 2K resolution images divided into: 800 images for training, 100 images for validation, 100 images for testing. The data set provides both high and low resolution images for 2, 3, and 4 down-scaling factors.
2. 3000 2K resolution images from random wallpapers

### Models
1. SRResNet
2. SRGAN

### Evaluation
Evalute SRResNet and SRGAN based on multiple loss criteria, time taken to train each model, average time taken to generate output images.
