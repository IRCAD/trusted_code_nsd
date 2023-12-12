import glob
import json
import os
import random
import shutil
import warnings

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import yaml
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstD,
    EnsureTypeD,
    LoadImaged,
    LoadImageD,
    RandFlipD,
    RandRotateD,
    RandZoomD,
    Resized,
    ScaleIntensityD,
)
from tqdm import trange

warnings.filterwarnings("ignore")


def monai_nnunet_data_preprocessing(root_dir, msd_task, num_crossfold, dataset_id):
    """
    Perform data preprocessing for MONAI nnUNet.

    This function prepares the necessary data for training and testing nnUNet models using MONAI.
    It creates the data list file for training and testing images, shuffles and divides training data
    into multiple folds, and writes the necessary information to an input.yaml file.

    Args:
        root_dir (str): Root directory where the data preprocessing will be performed.
        msd_task (str): Name of the MSD task.
        num_crossfold: Number of cross-fold for training.
        dataset_name: Dataset name used for training MONAI nnunet.
    """
    nnunet_preprocessed_path = os.path.join(root_dir, "nnUNet_preprocessed")
    if not os.path.exists(nnunet_preprocessed_path):
        os.makedirs(nnunet_preprocessed_path)
    nnunet_raw_path = os.path.join(root_dir, "nnUNet_raw_data_base")
    if not os.path.exists(nnunet_raw_path):
        os.makedirs(nnunet_raw_path)
    nnunet_results = os.path.join(root_dir, "nnUNet_trained_models")
    if not os.path.exists(nnunet_results):
        os.makedirs(nnunet_results)
    input_path_yaml = os.path.join(root_dir, "input.yaml")
    dataroot = os.path.join(root_dir, msd_task)
    test_dir = os.path.join(dataroot, "imagesTs/")
    train_dir = os.path.join(dataroot, "imagesTr/")
    datalist_json = {"testing": [], "training": []}
    datalist_json["testing"] = [
        {"image": "./imagesTs/" + file}
        for file in os.listdir(test_dir)
        if (".nii.gz" in file) and ("._" not in file)
    ]
    datalist_json["training"] = [
        {"image": "./imagesTr/" + file, "label": "./labelsTr/" + file, "fold": 0}
        for file in os.listdir(train_dir)
        if (".nii.gz" in file) and ("._" not in file)
    ]
    random.seed(42)
    random.shuffle(datalist_json["training"])
    num_folds = num_crossfold
    fold_size = len(datalist_json["training"]) // num_folds
    for i in range(num_folds):
        for j in range(fold_size):
            datalist_json["training"][i * fold_size + j]["fold"] = i
    datalist_file = "msd_" + msd_task.lower() + "_folds.json"
    datalist_file_path = os.path.join(dataroot, datalist_file)
    with open(datalist_file_path, "w", encoding="utf-8") as f:
        json.dump(datalist_json, f, ensure_ascii=False, indent=4)
    print(f"Datalist is saved to {datalist_file_path}")

    data = {
        "modality": "US",
        "datalist": datalist_file_path,
        "dataroot": dataroot,
        "dataset_name_or_id": dataset_id,  # task-specific integer index (optional)
        "nnunet_preprocessed": nnunet_preprocessed_path,  # directory for storing pre-processed data (optional)
        "nnunet_raw": nnunet_raw_path,  # directory for storing formatted raw data (optional)
        "nnunet_results": nnunet_results,  # directory for storing trained model checkpoints (optional)
    }

    # Write data to input.yaml
    with open(input_path_yaml, "w") as f:
        yaml.dump(data, f)


def set_nnunet_dir(root_dir):
    """
    Function that set nnUNet required directories
    Argument:
        -root_dir(str): root directory of these nnUNet directories"
    """
    raw_data_dir = os.path.join(root_dir, "nnUNet_raw")
    preprocessed_data_dir = os.path.join(root_dir, "nnUNet_preprocessed")
    results_dir = os.path.join(root_dir, "nnUNet_results")
    nnUNet_raw = raw_data_dir
    nnUNet_preprocessed = preprocessed_data_dir
    nnUNet_results = results_dir
    os.environ["nnUNet_raw"] = nnUNet_raw
    os.environ["nnUNet_preprocessed"] = nnUNet_preprocessed
    os.environ["nnUNet_results"] = nnUNet_results


def set_env_nnunet():
    """
    Function that sets the environmnet
    nnU-Net requires some environment variables so that it always knows where the raw data,
    preprocessed data and trained models are.

    """
    if (
        os.path.join(os.path.normpath(os.getcwd())).split("/")[-1]
        != "spatiotemporal-us-organ-segmentation"
    ):
        temp_folder = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    else:
        temp_folder = os.path.normpath(os.getcwd())
    nnraw = os.path.join(temp_folder, "data/processed/nnunetv2/nnUNet_raw/")
    if not os.path.exists(nnraw):
        os.makedirs(nnraw)
    nnpreprocessed = os.path.join(
        temp_folder, "data/processed/nnunetv2/nnUNet_preprocessed/"
    )
    if not os.path.exists(nnpreprocessed):
        os.makedirs(nnpreprocessed)
    nnresults = os.path.join(temp_folder, "data/processed/nnunetv2/nnUNet_results/")
    if not os.path.exists(nnresults):
        os.makedirs(nnresults)
    os.environ["nnUNet_raw"] = nnraw
    os.environ["nnUNet_preprocessed"] = nnpreprocessed
    os.environ["nnUNet_results"] = nnresults


def generate_json(image_folder, label_folder, test_folder, output_path):
    """
    Generate a JSON file in the desired format for a given dataset.

    Args:
        image_folder (str): Path to the folder containing the training image files.
        label_folder (str): Path to the folder containing the trining label files.
        test_folder (str): Path to the folder containing the test image files.
        output_path (str): Path to save the generated JSON file.
    """
    training_samples = []
    test_samples = []

    for filename in os.listdir(image_folder):
        if filename.endswith(".nii.gz"):
            image_path = os.path.join(image_folder, filename)
            label_path = os.path.join(label_folder, filename)

            if os.path.isfile(label_path):
                training_samples.append({"image": image_path, "label": label_path})

    for filename in os.listdir(test_folder):
        if filename.endswith(".nii.gz"):
            image_path = os.path.join(test_folder, filename)
            test_samples.append({"image": image_path})

    num_training = len(training_samples)
    num_test = len(test_samples)

    data = {
        "name": "Kidney",
        "description": "Kidney segmentation",
        "tensorImageSize": "3D",
        "reference": "King’s College London",
        "licence": "CC-BY-SA 4.0",
        "release": "1.0 04/05/2018",
        "modality": {"0": "US"},
        "labels": {"0": "background", "1": "kidney"},
        "numTraining": num_training,
        "numTest": num_test,
        "training": training_samples,
        "test": test_samples,
    }

    with open(output_path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def convert_2_match_nnunet_format(
    image_folder, volume_folder, image_destination_folder, volume_destination_folder
):
    """
    Convert and match files to the desired format for nnU-Net.

    Args:
        image_folder (str): Path to the folder containing the source .nii.gz files.
        volume_folder (str): Path to the folder containing the volume (label) .nii.gz files.
        image_destination_folder (str): Path to the first destination folder
                                        where the source files will be copied/renamed.
        volume_destination_folder (str): Path to the second destination folder
                                         where the volume files will be copied/renamed.
    """
    # Get a list of all .nii.gz files in the source folder
    source_files = [f for f in os.listdir(image_folder) if f.endswith(".nii.gz")]

    # Iterate over the source files and copy/rename them to the destination folders
    for i, source_file in enumerate(source_files):
        base_filename = os.path.basename(source_file)
        base_filename_without_ext = base_filename[: base_filename.rfind(".")]
        numeric_id = base_filename_without_ext[: base_filename_without_ext.rfind("_")]
        label_index = base_filename_without_ext[
            base_filename_without_ext.rfind("_") + 1 :
        ].split(".")[0]
        label_file = os.path.join(
            volume_folder, "{}_{}_Vol.nii.gz".format(numeric_id, label_index)
        )
        print("Source", source_file)
        print(numeric_id)
        print(label_index)
        print("Label", label_file)

        source_path = os.path.join(image_folder, source_file)
        if os.path.exists(label_file):
            print("Label exist")
            volume_path = os.path.join(volume_folder, label_file)

        new_name = f"la_{i+1:03d}.nii.gz"
        destination_path_1 = os.path.join(image_destination_folder, new_name)
        destination_path_2 = os.path.join(volume_destination_folder, new_name)

        if os.path.isfile(volume_path):
            # Copy/rename the source file to the first destination folder
            shutil.copy2(source_path, destination_path_1)

            # Copy/rename the volume file to the second destination folder
            shutil.copy2(volume_path, destination_path_2)

            print(
                f"Copied/Renamed {source_file} and {label_file} to {new_name} in both destination folders"
            )
        else:
            print(
                f"Skipping {source_file} as the corresponding volume file {label_file} is not found"
            )


def load_and_prepare_dataset(
    volume_folder, label_folder, volume_suffix=".nii.gz", test_size=0.2
):
    """
    Code for loading and preparing datasets

    Args:
        volume_folder: Path to the directory that holds the volumes
        label_folder: Path to the directory that holds the labels
        volume_suffix: the suffix for volume
        test_size: percentage of test data

    Returns:
        train_datadict: the dictionary of all the training path files
        test_datadict: the dictionary of all the test path files

    """
    files_ = sorted(glob.glob(os.path.join(volume_folder, "*" + volume_suffix)))

    # Split the image filenames into training and test sets
    if label_folder != "":
        num_test = int(len(files_) * test_size)
        test_files_ = files_[:num_test]
        train_files_ = files_[num_test:]
        # Create a list of dictionaries, where each dictionary contains an image filename and a label filename
        train_datadict = []
        test_datadict = []
    else:
        test_files_ = files_
        # Create a list of dictionaries, where each dictionary contains an image filename and a label filename
        test_datadict = []

    if label_folder != "":
        for image_file in test_files_:
            if os.path.exists(image_file):
                base_filename = os.path.basename(image_file)
                base_filename_without_ext = base_filename[: base_filename.rfind(".")]
                numeric_id = base_filename_without_ext[
                    : base_filename_without_ext.rfind("_")
                ]
                label_index = base_filename_without_ext[
                    base_filename_without_ext.rfind("_") + 1 :
                ].split(".")[0]
                label_file = os.path.join(
                    label_folder, "{}_{}_Vol.nii.gz".format(numeric_id, label_index)
                )

                if os.path.exists(label_file):
                    test_datadict.append({"image": image_file, "label": label_file})
                else:
                    print(f"Label file not found: {label_file}")

        for image_file in train_files_:
            if os.path.exists(image_file):
                base_filename = os.path.basename(image_file)
                base_filename_without_ext = base_filename[: base_filename.rfind(".")]
                numeric_id = base_filename_without_ext[
                    : base_filename_without_ext.rfind("_")
                ]
                label_index = base_filename_without_ext[
                    base_filename_without_ext.rfind("_") + 1 :
                ].split(".")[0]
                label_file = os.path.join(
                    label_folder, "{}_{}_Vol.nii.gz".format(numeric_id, label_index)
                )

                if os.path.exists(label_file):
                    train_datadict.append({"image": image_file, "label": label_file})
                else:
                    print(f"Label file not found: {label_file}")

        # Print the number of training and test samples
        print("Number of training samples:", len(train_datadict))
        print("Number of test samples:", len(test_datadict))

        return train_datadict, test_datadict
    else:
        for image_file in test_files_:
            if os.path.exists(image_file):
                test_datadict.append({"image": image_file})
        return None, test_datadict


def apply_transforms(
    train_datadict, test_datadict, inference, batch_size=4, size=128, num_workers=10
):
    """
    Based upon the setting this function will apply transformation
    for training and validation dataset

    Args:
        train_datadict: the dictionary of all the training path files
        test_datadict: the dictionary of all the test path files
        inference: if inference is True or False. If True then we evaluate only the
                   inference, If False the script for training
        batch_size: batch size for the data loader
        size: spatial size
        num_workers: number of workers

    Returns:
        train_loader: train data loader
        test_loader: test data loader
    """

    if inference:
        test_transforms = Compose(
            [
                LoadImageD(keys=["image"]),
                EnsureChannelFirstD(keys=["image"]),
                ScaleIntensityD(keys=["image"]),
                Resized(keys=["image"], spatial_size=(size, size, size)),
                EnsureTypeD(keys=["image"]),
            ]
        )
        test_ds = CacheDataset(test_datadict, test_transforms, num_workers=num_workers)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        return None, test_loader

    else:
        train_transforms = Compose(
            [
                LoadImageD(keys=["image", "label"]),
                EnsureChannelFirstD(keys=["image", "label"]),
                ScaleIntensityD(keys=["image", "label"]),
                RandRotateD(
                    keys=["image", "label"],
                    range_x=np.pi / 12,
                    prob=0.5,
                    keep_size=True,
                ),
                RandFlipD(keys=["image", "label"], spatial_axis=0, prob=0.5),
                RandZoomD(
                    keys=["image", "label"], min_zoom=0.9, max_zoom=1.1, prob=0.5
                ),
                Resized(keys=["image", "label"], spatial_size=(size, size, size)),
                EnsureTypeD(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImageD(keys=["image", "label"]),
                EnsureChannelFirstD(keys=["image", "label"]),
                ScaleIntensityD(keys=["image", "label"]),
                Resized(keys=["image", "label"], spatial_size=(size, size, size)),
                EnsureTypeD(keys=["image", "label"]),
            ]
        )

        test_ds = CacheDataset(test_datadict, test_transforms, num_workers=num_workers)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        train_ds = CacheDataset(
            train_datadict, train_transforms, num_workers=num_workers
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        return train_loader, test_loader


def infer_and_save_results(model, test_loader, output_folder, device, size, inference):
    """
    Function that will do the inference and saving the results

    Args:
            model: best model saved after training
            test_loader: test dataloader
            output_folder: Path to the directory where the model is going to be predict
            device: to use GPU or CPU
            size: spatial size
            inference: if inference is True or False. If True then we evaluate only the
                       inference, If False the script for training

    """
    model.eval()
    with torch.no_grad():
        for i, val_data in enumerate(test_loader):
            roi_size = (size, size, size)
            name = (
                val_data["image_meta_dict"]["filename_or_obj"][0]
                .split("/")[-1]
                .split(".")[0]
            )
            val_outputs = sliding_window_inference(
                val_data["image"].to(device), roi_size, 1, model
            )
            if not inference:
                output_path_3D_volume = os.path.join(
                    output_folder, str(name) + "_GTlabel.nii.gz"
                )
                result_vol = sitk.GetImageFromArray(val_data["label"].detach().cpu())
                sitk.WriteImage(result_vol, output_path_3D_volume)
            output_path_3D_volume_gt = os.path.join(
                output_folder, str(name) + "_GTimage.nii.gz"
            )
            result_vol_gt = sitk.GetImageFromArray(val_data["image"].detach().cpu())
            sitk.WriteImage(result_vol_gt, output_path_3D_volume_gt)
            output_path_3D_volume_pr = os.path.join(
                output_folder, str(name) + "_PR.nii.gz"
            )
            result_vol_pr = sitk.GetImageFromArray(
                np.argmax(val_outputs.detach().cpu()[0, :, :, :], axis=0)
            )
            sitk.WriteImage(result_vol_pr, output_path_3D_volume_pr)


def train_model(
    model, loss_function, optimizer, train_loader, device, step, epoch_loss
):
    """
    Function that will do the training

    Args:
            model: MONAI model
            loss_function: a loss function
            optimizer: an optimizer function
            train_loader: train loader
            device: to use GPU or CPU
            step: Step value
            epoch_loss : Epoch loss value

        Returns:
            model: trained model
            epoch_loss : epoch loss value
    """

    for batch_data in train_loader:
        step += 1
        inputs = batch_data["image"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, batch_data["label"].to(device))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return model, step


def validate_and_checkpoint(
    model,
    metric,
    best_metric,
    best_metric_epoch,
    output_folder,
    device,
    test_loader,
    post_pred,
    post_label,
    epoch,
    size,
    batch,
):
    """
    Function that will do the validation and saving the checkpoint model

    Args:
            model: MONAI model to be valuated
            metric: Loss Metric
            best_metric: Best metric value for current epoch
            best_metric_epoch: best_metric_epoch until now saved
            output_folder: Path to the directory where the model is going to be saved
            test_loader: test loader used for comparing
            device: to use GPU or CPU
            post_pred: post processing for prediction
            post_label: post processing for label
            epoch: epoch value
            size: spatial size
            batch: Batch size

        Returns:
            model: training model
            best_metric: best_metric value
            best_metric_epoch: best_metric_epoch value per epoch
    """

    model.eval()
    with torch.no_grad():
        for val_data in test_loader:
            val_inputs, val_labels = (
                val_data["image"].to(device),
                val_data["label"].to(device),
            )
            roi_size = (size, size, size)
            sw_batch_size = batch
            val_outputs = sliding_window_inference(
                val_inputs, roi_size, sw_batch_size, model
            )
            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]
            # compute metric for current iteration
            metric(y_pred=val_outputs, y=val_labels)

        # aggregate the final mean dice result
        metric_result = metric.aggregate().item()
        # reset the status for next validation round
        metric.reset()

        if metric_result > best_metric:
            best_metric = metric_result
            best_metric_epoch = epoch + 1
            best_model = model
            try:
                # Create output Directory
                os.makedirs(os.path.join(output_folder))
                print("Directory ", os.path.join(output_folder), " Created ")
            except FileExistsError:
                print("Directory ", os.path.join(output_folder), " already exists")
            torch.save(
                model.state_dict(), os.path.join(output_folder, "best_metric_model.pth")
            )
            print("saved new best metric model")
        print(
            f"current epoch: {epoch + 1} current mean dice: {metric_result:.4f}"
            f"\nbest mean dice: {best_metric:.4f} "
            f"at epoch: {best_metric_epoch}"
        )
    return model, best_metric, best_metric_epoch, best_model


def unet_training(
    volume_folder, label_folder, output_folder, epoch, size, batch_size, number_workers
):
    """
    Function that does the training of a UNET model using MONAI

    Args:
            volume_folder: Path to the directory that holds the volumes
            label_folder: Path to the directory that holds the labels
            output_folder: Path to the directory where the model is going to be saved
            epoch: number of epoch
            size: spatial size
            batch_size: batch size
            number_workers: number of workers

    """
    device = torch.device("cuda:0")
    train_datadict, test_datadict = load_and_prepare_dataset(
        volume_folder, label_folder, volume_suffix=".nii.gz", test_size=0.2
    )
    train_loader, test_loader = apply_transforms(
        train_datadict,
        test_datadict,
        inference=False,
        batch_size=batch_size,
        size=size,
        num_workers=number_workers,
    )
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    metric = DiceMetric(include_background=True, reduction="mean")
    best_metric = -1
    best_metric_epoch = -1
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    epoch_loss_values = []
    val_interval = 1

    # 3. Training Loop

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    model.to(device)
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    t = trange(epoch, leave=True)
    for epoch in t:
        model.train()
        epoch_loss = 0
        step = 0
        model, step = train_model(
            model, loss_function, optimizer, train_loader, device, step, epoch_loss
        )
        epoch_loss /= step
        t.set_description(
            f" -- epoch {epoch + 1}" + f", average loss: {epoch_loss:.4f}"
        )
        if (epoch + 1) % val_interval == 0:
            model, best_metric, best_metric_epoch, best_model = validate_and_checkpoint(
                model,
                metric,
                best_metric,
                best_metric_epoch,
                output_folder,
                device,
                test_loader,
                post_pred,
                post_label,
                epoch,
                size,
                batch=1,
            )
        epoch_loss_values.append(epoch_loss)
    print(
        f"train completed for, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )

    # Infer and save results
    infer_and_save_results(
        best_model, test_loader, output_folder, device, size, inference=False
    )


def unet_inference(
    volume_folder, model_path, output_folder, size, batch_size, number_workers
):
    """
    Function that does the inference of a UNET model using MONAI

    Args:
            volume_folder: Path to the directory that holds the volumes
            model_path: Path to the model
            output_folder: Path to the directory where the model is going to be saved
            size: spatial size
            batch_size: batch size
            number_workers: number of workers

    """
    device = torch.device("cuda:0")
    _, test_datadict = load_and_prepare_dataset(
        volume_folder, label_folder="", volume_suffix=".nii.gz"
    )
    _, test_loader = apply_transforms(
        train_datadict=None,
        test_datadict=test_datadict,
        inference=True,
        batch_size=batch_size,
        size=size,
        num_workers=number_workers,
    )

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(model_path, "best_metric_model.pth")))
    # Infer and save results
    infer_and_save_results(
        model, test_loader, output_folder, device, size, inference=True
    )


def split_segments(image_path, mask_path, output_dir):
    """
    Generates a set of noisy masks / target mask pairs

    """

    keys = ["mask", "image"]
    data = {"mask": mask_path, "image": image_path}
    name_file = image_path.split("/")[-1].split(".")[0]
    print(image_path.split("/")[-1].split(".")[0])

    load_transform = LoadImaged(keys)
    image_transform = load_transform(data)

    print(image_transform["mask"].shape)
    print(image_transform["image"].shape)

    masks = image_transform["mask"]
    images = image_transform["image"]
    threshold = 0.5  # Adjust this value based on your specific problem

    # Automatically identify the segment time ranges
    segment_time_ranges = []
    end_time = None
    for t in range(masks.shape[0]):
        mask = masks[t]
        mask_indices = np.where(mask > threshold)
        if len(mask_indices[0]) > 0:
            if end_time is None:
                end_time = t
        elif end_time is not None:
            segment_time_ranges.append((end_time, t - 1))
            end_time = None
    if end_time is not None:
        segment_time_ranges.append((end_time, masks.shape[0] - 1))

    # Calculate the number of segments
    num_segments = len(segment_time_ranges)

    # Calculate the start and end times for each segment
    segment_time_ranges_adjusted = []
    for i, (start_time, end_time) in enumerate(segment_time_ranges):
        if i == 0:
            start = 0
            end = end_time + (segment_time_ranges[i + 1][0] - end_time) // 2
        elif i == num_segments - 1:
            start = (segment_time_ranges[i - 1][1] + start_time) // 2 + 1
            end = masks.shape[0] - 1
        else:
            start = (segment_time_ranges[i - 1][1] + start_time) // 2 + 1
            end = end_time + (segment_time_ranges[i + 1][0] - end_time) // 2

        segment_time_ranges_adjusted.append((start, end))

    for segment_idx, (start_time, end_time) in enumerate(segment_time_ranges_adjusted):
        mask_segment = masks[start_time : end_time + 1]
        image_segment = images[start_time : end_time + 1]
        mask_volume = np.stack(mask_segment, axis=0)
        image_volume = np.stack(image_segment, axis=0)
        affine = image_transform["mask_meta_dict"][
            "affine"
        ]  # Use the affine information from the original image

        file_path_mask = os.path.join(
            output_dir, f"{name_file}_{segment_idx}_Vol.nii.gz"
        )
        file_path_image = os.path.join(output_dir, f"{name_file}_{segment_idx}.nii.gz")
        print("Saving Files")
        nib.save(nib.Nifti1Image(mask_volume, affine), file_path_mask)
        nib.save(nib.Nifti1Image(image_volume, affine), file_path_image)
