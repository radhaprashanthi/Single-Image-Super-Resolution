# def test():
#     # pretrained_model =
#     # load_checkpoint(filepath="/content/drive/My Drive/ML/sr/model_checkpoint/pretrained_to_new.pth")
#     # lr, hr = valid_dataset[9]
#     lr, hr = train_dataset[100]
#
#     generator.eval()
#     sr = (
#         generator(
#             lr.unsqueeze(0).to(device)
#         ).to("cpu")
#             .float()
#             .squeeze(0)
#             .detach()
#     )
#
#     ToPILImage()(lr)
#
#     ToPILImage()(hr)
#
#     ToPILImage()(output_to_imagenet(sr))
