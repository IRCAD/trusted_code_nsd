# """
# Run the trained VNet models on our 5 folds to produce the automatic segmentation of our US data

# Parameters:
# ----------
#     data_root : string
#                 images folder
#     models_folder : string
#                     trained models folder
#     out_root: string
#                 where to save the predicted masks
# """
# import os
# from os.path import join
# from sys import maxsize

# import monai
# import nibabel as nib
# import numpy as np
# import torch
# import yaml
# from monai.data import DataLoader
# from monai.networks.nets import UNet, VNet
# from monai.transforms import (
#     Activations,
#     AsDiscrete,
#     Compose,
#     EnsureChannelFirstd,
#     LoadImaged,
#     NormalizeIntensityd,
#     Resize,
#     Resized,
#     ToTensord,
# )
# from numpy import set_printoptions

# from trusted_datapaper_ds.utils import makedir, parse_args

# set_printoptions(threshold=maxsize)
# device = torch.device("cuda:0")


# val_transform = Compose(
#     [
#         LoadImaged(keys=["image"], image_only=False),
#         EnsureChannelFirstd(keys=["image"]),
#         Resized(keys=["image"], spatial_size=(128, 128, 128), mode="trilinear"),
#         NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
#         ToTensord(keys=["image"]),
#     ]
# )

# post_trans1 = Compose(
#     [Activations(sigmoid=True), AsDiscrete(threshold_values=True, logit_thresh=0.5)]
# )


# def segmentor(
#     model_name, weight_file, img_files, output_folder=None, maskinterpolmode="trilinear"
# ):
#     assert ("unet" in model_name) or (
#         "vnet" in model_name
#     ), " Cannot identify the name of the model. "

#     if output_folder is not None:
#         makedir(output_folder)

#     if "unet" in model_name:
#         model = UNet(
#             dimensions=3,
#             in_channels=1,
#             out_channels=1,
#             act="PRELU",
#             channels=(16, 32, 64, 128, 256),
#             strides=(2, 2, 2, 2),
#             dropout=0.0,
#         ).to(device)

#     if "vnet" in model_name:
#         model = VNet(
#             spatial_dims=3,
#             in_channels=1,
#             out_channels=1,
#             act=("elu", {"inplace": True}),
#             dropout_prob=0.5,
#             # dropout_prob_down=0.5,
#             # dropout_prob_up=(0.5, 0.5),
#             dropout_dim=3,
#         ).to(device)

#     model.load_state_dict(torch.load(weight_file))

#     dict_img_paths = [{"image": img_files[i]} for i in range(len(img_files))]
#     ds = monai.data.Dataset(dict_img_paths, transform=val_transform)
#     loader = DataLoader(ds, batch_size=1, shuffle=False)

#     model.eval()

#     with torch.no_grad():
#         for image_data in loader:
#             file_name = image_data["image_meta_dict"]["filename_or_obj"][0]
#             print("Segmenting the image file: ")
#             print(file_name)

#             affine = image_data["image_meta_dict"]["affine"][0]
#             original_shape = np.ndarray.tolist(
#                 np.array(image_data["image_meta_dict"]["dim"][0, 1:4])
#             )
#             img_basename = os.path.basename(file_name)

#             if maskinterpolmode == "trilinear":
#                 post_trans2 = Compose(
#                     [
#                         Resize(
#                             spatial_size=original_shape,
#                             mode=maskinterpolmode,
#                             align_corners=True,
#                         ),
#                         AsDiscrete(threshold_values=True, logit_thresh=0.5),
#                     ]
#                 )
#             else:
#                 post_trans2 = Compose(
#                     [
#                         Resize(
#                             spatial_size=original_shape,
#                             mode=maskinterpolmode,
#                         ),
#                         AsDiscrete(threshold_values=True, logit_thresh=0.5),
#                     ]
#                 )

#             val_input = np.array(image_data["image"])
#             val_input = torch.as_tensor(val_input, dtype=None, device=device)
#             val_output = model(val_input)
#             val_output = post_trans1(val_output)
#             np_val_output = np.asarray(val_output.detach().cpu()).squeeze(0).squeeze(0)
#             print(np_val_output.shape)

#             # np_val_output = post_trans2(np_val_output)
#             # np_val_output = np.asarray(np_val_output.detach().cpu()).squeeze(0)

#             nib_val_output = nib.Nifti1Image(np_val_output, affine)

#             if output_folder is not None:
#                 mask_basename = img_basename.replace("img", "mask")
#                 out_file_name = join(output_folder, mask_basename)
#                 nib.save(nib_val_output, out_file_name)
#                 print("The segmentation prediction saved as: ", out_file_name)

#     return nib_val_output


# if __name__ == "__main__":
#     args = parse_args()
#     with open(args.config_path, "r") as yaml_file:
#         config = yaml.safe_load(yaml_file)

#     modality = config["modality"]
#     # for cv in ["cv1", "cv2", "cv3", "cv4", "cv5"]:
#     for cv in ["cv10"]:
#         model_name = config["segmodel"]
#         output_folder = join(
#             config["seg128location"], config["segmodel"], config["training_target"]
#         )  # Here the outputs are already upsampled

#         weight_file = join(
#             config["trained_models_location"],
#             model_name,
#             cv,
#             config["training_target"],
#             "best_metric_model.pth",
#         )

#         img_folder = join(config["img_location"])
#         img_files = [
#             join(img_folder, i + config[modality + "img_end"]) for i in config[cv]
#         ]
#         print(len(img_files))

#         nib_val_output = segmentor(
#             model_name,
#             weight_file,
#             img_files,
#             output_folder,
#             maskinterpolmode="trilinear",
#         )
