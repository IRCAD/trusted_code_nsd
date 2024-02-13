# Setup imports"""
import os
import sys
import time
import warnings
from glob import glob
from os.path import isfile, join

import matplotlib
import matplotlib.pyplot as plt
import monai
import numpy as np
import torch
import yaml
from monai.data import DataLoader
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet, VNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
    ToTensord,
)
from monai.utils import set_determinism
from natsort import natsorted

from trusted_datapaper_ds.utils import parse_args

warnings.filterwarnings("ignore")  # remove some warnings
matplotlib.use("Agg")


def training(
    config,
    modality,
    cv,
    segmodel,
    data_location,
    output_location,
    lr,
    wdecay,
    nb_epochs,
    double_targets_training,
):
    """## out.txt is the text file containing the informations concerning the training events like
    the training loss value, the validation metric value, ...
    """

    print(data_location)

    NumberOfEpochs = nb_epochs

    # input_folder = join(data_location, modality + "_DATA", modality + "img128")
    input_folder = join(data_location, config[modality + "128imgfol"])
    target_folder1 = join(data_location, config[modality + "128ma1fol"])
    target_folder2 = join(data_location, config[modality + "128ma2fol"])
    target_folder12 = join(data_location, config[modality + "128magtfol"])

    if double_targets_training:
        print("###")
        print("TWO TRAINING TARGETS SEGMENTATION WITH " + segmodel)
        print("###")
        output_folder = join(
            output_location,
            modality + "_DATA",
            modality + "seg_trained_models",
            segmodel + "_lr" + str(lr) + "_epoch" + str(nb_epochs),
            cv,
            "double_targets_training",
        )
    if not double_targets_training:
        print("###")
        print("SINGLE FUSED TRAINING TARGETS SEGMENTATION WITH " + segmodel)
        print("###")
        output_folder = join(
            output_location,
            modality + "_DATA",
            modality + "seg_trained_models",
            segmodel + "_lr" + str(lr) + "_epoch" + str(nb_epochs),
            cv,
            "single_target_training",
        )

    try:
        # Create ouput Directory
        os.makedirs(output_folder)
        print("Directory ", output_folder, " Created ")
    except FileExistsError:
        print("Directory ", output_folder, " already exists")

    indiv = config[cv]
    print("indiv = ", indiv)

    all_img_paths = natsorted(glob(join(input_folder, "*.nii.gz")))
    valid_img_paths = [
        join(input_folder, i + config[modality + "img_end"])
        for i in indiv
        if isfile(join(input_folder, i + config[modality + "img_end"]))
    ]
    all_seg12_paths = natsorted(glob(join(target_folder12, "*.nii.gz")))
    valid_seg12_paths = [
        join(target_folder12, i + config[modality + "ma_end"])
        for i in indiv
        if isfile(join(target_folder12, i + config[modality + "ma_end"]))
    ]

    if modality == "CT":
        middle = "_"
    if modality == "US":
        middle = ""

    all_seg1_paths = natsorted(glob(join(target_folder1, "*.nii.gz")))
    valid_seg1_paths = [
        join(target_folder1, i + middle + "1" + config[modality + "ma_end"])
        for i in indiv
        if isfile(join(target_folder1, i + middle + "1" + config[modality + "ma_end"]))
    ]
    all_seg2_paths = natsorted(glob(join(target_folder2, "*.nii.gz")))
    valid_seg2_paths = [
        join(target_folder2, i + middle + "2" + config[modality + "ma_end"])
        for i in indiv
        if isfile(join(target_folder2, i + middle + "2" + config[modality + "ma_end"]))
    ]

    print("len valid_img_paths", len(valid_img_paths))
    print("len valid_seg1_paths", len(valid_seg1_paths))
    print("len valid_seg2_paths", len(valid_seg2_paths))
    print("len valid_seg12_paths", len(valid_seg12_paths))

    if len(valid_img_paths) != len(indiv) or (len(valid_seg1_paths) != len(indiv)):
        sys.exit(
            "STOP! Please check if the valid img paths or valid seg1 paths are all correct"
        )
    if len(valid_img_paths) != len(indiv) or (len(valid_seg2_paths) != len(indiv)):
        sys.exit(
            "STOP! Please check if the valid img paths or valid seg2 paths are all correct"
        )

    train_img_paths = natsorted([i for i in all_img_paths if i not in valid_img_paths])
    train_seg1_paths = natsorted(
        [i for i in all_seg1_paths if i not in valid_seg1_paths]
    )
    train_seg2_paths = natsorted(
        [i for i in all_seg2_paths if i not in valid_seg2_paths]
    )
    train_seg12_paths = natsorted(
        [i for i in all_seg12_paths if i not in valid_seg12_paths]
    )

    """## Set deterministic training for reproducibility"""
    set_determinism(seed=0)

    train_transform = Compose(
        [
            LoadImaged(keys=["image", "label1", "label2", "label12"]),
            EnsureChannelFirstd(keys=["image", "label1", "label2", "label12"]),
            RandFlipd(
                keys=["image", "label1", "label2", "label12"], prob=0.25, spatial_axis=0
            ),
            RandFlipd(
                keys=["image", "label1", "label2", "label12"], prob=0.25, spatial_axis=1
            ),
            RandFlipd(
                keys=["image", "label1", "label2", "label12"], prob=0.25, spatial_axis=2
            ),
            RandAffined(
                keys=["image", "label1", "label2", "label12"],
                rotate_range=(np.pi, np.pi, np.pi),
                translate_range=(50, 50, 50),
                padding_mode="border",
                scale_range=(0.25, 0.25, 0.25),
                mode=("bilinear", "nearest", "nearest", "nearest"),
                prob=1.0,
            ),
            Resized(
                keys=["image", "label1", "label2", "label12"],
                spatial_size=(128, 128, 128),
                mode="trilinear",
                align_corners=True,
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            RandGaussianNoised(keys="image", prob=0.1, mean=0.0, std=0.1),
            RandGaussianSmoothd(
                keys="image",
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
                prob=0.2,
            ),
            RandAdjustContrastd(keys="image", prob=0.15),
            ToTensord(keys=["image", "label1", "label2", "label12"]),
        ]
    )

    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label12"]),
            EnsureChannelFirstd(keys=["image", "label12"]),
            Resized(
                keys=["image", "label12"],
                spatial_size=(128, 128, 128),
                mode="trilinear",
                align_corners=True,
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "label12"]),
        ]
    )
    binarizing = AsDiscrete(threshold_values=True, logit_thresh=0.5)

    dict_train_data_paths = [
        {
            "image": train_img_paths[i],
            "label1": train_seg1_paths[i],
            "label2": train_seg2_paths[i],
            "label12": train_seg12_paths[i],
        }
        for i in range(len(train_img_paths))
    ]
    dict_valid_data_paths = [
        {"image": valid_img_paths[i], "label12": valid_seg12_paths[i]}
        for i in range(len(valid_img_paths))
    ]

    train_ds = monai.data.CacheDataset(
        dict_train_data_paths,
        transform=train_transform,
        cache_num=24,
        cache_rate=1.0,
        num_workers=1,
    )
    val_ds = monai.data.CacheDataset(
        dict_valid_data_paths,
        transform=val_transform,
        cache_num=6,
        cache_rate=1.0,
        num_workers=1,
    )
    train_loader = DataLoader(
        train_ds, batch_size=2, shuffle=True, num_workers=1, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )

    """## Create Model, Loss, Optimizer"""
    device = torch.device("cuda:0")

    if segmodel == "unet_test":
        model = UNet(
            dimensions=3,
            in_channels=1,
            out_channels=1,
            act="PRELU",
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            dropout=0.0,
        ).to(device)

        loss_function = DiceCELoss(
            to_onehot_y=False,
            sigmoid=True,
            softmax=False,
            other_act=None,
            squared_pred=False,
            jaccard=False,
            reduction="mean",
            smooth_nr=1e-05,
            smooth_dr=1e-05,
            batch=False,
            ce_weight=None,
            lambda_dice=1.0,
            lambda_ce=1.0,
        )
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wdecay)

    if segmodel == "vnet_test":
        model = VNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            act=("elu", {"inplace": True}),
            dropout_prob=0.5,
            dropout_dim=3,
        ).to(device)

        loss_function = DiceCELoss(
            to_onehot_y=False,
            sigmoid=True,
            softmax=False,
            other_act=None,
            squared_pred=False,
            jaccard=False,
            reduction="mean",
            smooth_nr=1e-05,
            smooth_dr=1e-05,
            batch=False,
            ce_weight=None,
            lambda_dice=1.0,
            lambda_ce=1.0,
        )
        torch.backends.cudnn.benchmark = True
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wdecay)

    tic = time.perf_counter()

    """## Execute a typical PyTorch training process"""
    max_epochs = NumberOfEpochs
    val_interval = 2

    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels1, labels2, labels12 = (
                batch_data["image"].to(device),
                batch_data["label1"].to(device),
                batch_data["label2"].to(device),
                batch_data["label12"].to(device),
            )
            labels1 = binarizing(labels1)
            labels2 = binarizing(labels2)
            labels12 = binarizing(labels12)
            optimizer.zero_grad()
            outputs = model(inputs)
            if double_targets_training:
                loss = (
                    loss_function(outputs, labels1) + loss_function(outputs, labels2)
                ) * 0.5
            if not double_targets_training:
                loss = loss_function(outputs, labels12)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}"
                f", train_loss: {loss.item():.4f}"
            )
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                dice_metric12 = DiceMetric(include_background=False, reduction="mean")
                post_trans = Compose(
                    [
                        Activations(sigmoid=True),
                        AsDiscrete(threshold_values=True, logit_thresh=0.5),
                    ]
                )
                epoch_metric_vector = list()
                for val_data in val_loader:
                    val_inputs, val_labels12 = (
                        val_data["image"].to(device),
                        val_data["label12"].to(device),
                    )
                    val_labels12 = binarizing(val_labels12)
                    val_outputs = model(val_inputs)
                    val_outputs = post_trans(val_outputs)

                    dice_metric12(y_pred=val_outputs, y=val_labels12)
                    dice12 = dice_metric12.aggregate().item()
                    dice = dice12
                    epoch_metric_vector.append(dice)

                dice_metric12.reset()

            metric = np.mean(np.asarray(epoch_metric_vector))

            metric_values.append(metric)

            if metric > best_metric:
                best_metric = metric
                best_epoch_metric_vector = np.asarray(epoch_metric_vector)
                best_metric_epoch = epoch + 1
                torch.save(
                    model.state_dict(),
                    os.path.join(output_folder, "best_metric_model.pth"),
                )
                np.savetxt(
                    join(output_folder, "best_epoch_metric_vector.txt"),
                    best_epoch_metric_vector,
                )
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )

    print(
        f"train completed, best_metric: {best_metric:.4f}"
        f" at epoch: {best_metric_epoch}"
    )

    """## Plot the loss and metric"""
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y, color="red")
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y, color="green")
    plt.savefig(join(output_folder, "loss_metric" + ".png"))
    plt.close("train")

    """## Save the loss and metric"""
    np.savetxt(join(output_folder, "loss.txt"), epoch_loss_values)
    np.savetxt(join(output_folder, "metric.txt"), metric_values)

    toc = time.perf_counter()
    print(f"Trained in {toc - tic:0.4f} seconds")

    #################################
    print("###")
    print("FINISH")

    return


def main(config, modality, segmodel, lr, wdecay, nb_epochs):
    data_location = config["data_location"]
    output_location = config["output_location"]
    double_targets_training = config["training_target"] == "double_targets_training"

    for cv in config["cv_list"]:
        training(
            config,
            modality,
            cv,
            segmodel,
            data_location,
            output_location,
            lr,
            wdecay,
            nb_epochs,
            double_targets_training=double_targets_training,
        )


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    modality = config["modality"]
    segmodel = config["trainsegmodel"]

    lr = 1e-0
    wdecay = 1e-5
    nb_epochs = 1001

    # Please respect the case
    main(
        config=config,
        modality=modality,
        segmodel=segmodel,
        lr=lr,
        wdecay=wdecay,
        nb_epochs=nb_epochs,
    )
