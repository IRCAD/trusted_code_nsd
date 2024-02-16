"""
    Copyright (C) 2022-2024 IRCAD France - All rights reserved. *
    This file is part of Disrumpere. *
    Disrumpere can not be copied, modified and/or distributed without
    the express permission of IRCAD France.
"""

import os
from os.path import join
from sys import maxsize

import monai
import nibabel as nib
import numpy as np
import torch
import yaml
from monai.data import DataLoader
from monai.networks.nets import UNet, VNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    EnsureType,
    LoadImaged,
    NormalizeIntensityd,
    Resize,
    Resized,
    ToTensord,
)
from monai.utils import set_determinism
from numpy import set_printoptions

from trusted_datapaper_ds.utils import makedir, parse_args

"""## Set deterministic training for reproducibility"""
set_determinism(seed=0)

set_printoptions(threshold=maxsize)
device = torch.device("cuda:0")

val_transform = Compose(
    [
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(128, 128, 128), mode="trilinear"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ToTensord(keys=["image"]),
    ]
)

post_trans1 = Compose(
    [Activations(sigmoid=True), AsDiscrete(threshold_values=True, logit_thresh=0.5)]
)
binarizing = Compose(
    [EnsureType(data_type="tensor"), AsDiscrete(threshold_values=True)]
)


def segmentor(
    model_name, weight_file, img_files, output_folder=None, maskinterpolmode="trilinear"
):
    """
    This function loads a UNet or VNet model, applies it to a list of input images,
    and optionally saves the resulting segmentation masks.

    Args:
        model_name (str): Name of the model to use, either "unet" or "vnet".
        weight_file (str): Path to the saved model weights file.
        img_files (list): List of paths to input images to be segmented.
        output_folder (str, optional): Path to a folder where segmentation masks will be saved.
            Defaults to None, which means masks will not be saved.
        maskinterpolmode (str, optional): Interpolation mode for resizing segmentation masks.
            Defaults to "trilinear".

    Returns:
        nib.Nifti1Image: The predicted segmentation mask.

    Raises:
        AssertionError: If the provided model_name is not "unet" or "vnet".
    """

    assert ("unet" in model_name) or (
        "vnet" in model_name
    ), " Cannot identify the name of the model. "

    if output_folder is not None:
        makedir(output_folder)

    if "unet" in model_name:
        model = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=1,
            act="PRELU",
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            dropout=0.0,
        ).to(device)

    if "vnet" in model_name:
        model = VNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            act=("elu", {"inplace": True}),
            dropout_prob=0.5,
            dropout_dim=3,
        ).to(device)

    model.load_state_dict(torch.load(weight_file))

    dict_img_paths = [{"image": img_files[i]} for i in range(len(img_files))]
    ds = monai.data.Dataset(dict_img_paths, transform=val_transform)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model.eval()

    with torch.no_grad():
        for image_data in loader:
            file_name = image_data["image_meta_dict"]["filename_or_obj"][0]
            print("Segmenting the image file: ")
            print(file_name)

            affine = image_data["image_meta_dict"]["affine"][0]
            original_shape = np.ndarray.tolist(
                np.array(image_data["image_meta_dict"]["dim"][0, 1:4])
            )
            img_basename = os.path.basename(file_name)

            if maskinterpolmode == "trilinear":
                post_trans2 = Resize(
                    spatial_size=original_shape,
                    mode=maskinterpolmode,
                    align_corners=True,
                )
            else:
                post_trans2 = Resize(
                    spatial_size=original_shape,
                    mode=maskinterpolmode,
                )

            val_input = np.array(image_data["image"])
            val_input = torch.as_tensor(val_input, dtype=None, device=device)
            val_output = model(val_input)
            val_output = post_trans1(val_output)
            val_output = val_output.detach().cpu()
            val_output = torch.squeeze(val_output, 0)

            val_output = post_trans2(val_output)
            val_output = binarizing(val_output)

            np_val_output = np.asarray(val_output).squeeze(0)

            nib_val_output = nib.Nifti1Image(np_val_output, affine)

            if output_folder is not None:
                mask_basename = img_basename.replace("img", "mask")
                out_file_name = join(output_folder, mask_basename)
                nib.save(nib_val_output, out_file_name)
                print("The segmentation prediction saved as: ", out_file_name)

    return nib_val_output


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    modality = config["modality"]

    list_segmodel = config["list_UVnetmodels"]
    list_training_target = config["list_training_target"]

    for segmodel in list_segmodel:
        for training_target in list_training_target:
            print(segmodel)
            print(training_target)

            for cv in ["cv1", "cv2", "cv3", "cv4", "cv5"]:
                model_name = segmodel
                output_folder = join(
                    config["mask_seglocation"],
                    segmodel,
                    training_target,
                )  # Here the outputs are already upsampled

                weight_file = join(
                    config["trained_models_location"],
                    model_name,
                    cv,
                    training_target,
                    "best_metric_model.pth",
                )

                img_folder = join(config["img_location"])
                img_files = [
                    join(img_folder, i + config[modality + "img_end"])
                    for i in config[cv]
                ]

                nib_val_output = segmentor(
                    model_name,
                    weight_file,
                    img_files,
                    output_folder,
                    maskinterpolmode="trilinear",
                )
