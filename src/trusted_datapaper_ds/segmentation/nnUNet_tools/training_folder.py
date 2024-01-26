import os
import shutil
from glob import glob
from itertools import zip_longest
from os.path import join

from natsort import natsorted

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.utils import makedir


def create(config):
    modality = config["modality"]

    # Folders to copy files from
    source_folder = config["data_location"]
    img_src_folder = join(source_folder, config[modality + "128imgfol"])
    if config["ntarget"] == "1":
        target = "single"
        gt_src_folder = join(source_folder, config[modality + "128magtfol"])
    if config["ntarget"] == "2":
        target = "double"
        gt1_src_folder = join(source_folder, config[modality + "128ma1fol"])
        gt2_src_folder = join(source_folder, config[modality + "128ma2fol"])

    # Folder to copy files to
    dest_folder = config["trained_models_location"]
    img_dest_folder = os.path.join(
        dest_folder, config["nnunet_data"], "raw", "input", target
    )
    makedir(img_dest_folder)
    gt_dest_folder = join(
        dest_folder, config["nnunet_data"], "raw", "ground_truth", target
    )
    makedir(gt_dest_folder)

    # Get list of files to copy
    if config["ntarget"] == "1":
        img_files = natsorted(glob(join(img_src_folder, "*.nii.gz")))
        gt_files = natsorted(glob(join(gt_src_folder, "*.nii.gz")))
        for img_src_file, gt_src_file in zip(img_files, gt_files):
            img_src = dt.Image(img_src_file)
            ID = img_src.individual_name

            img_dest_path = join(img_dest_folder, ID + ".nii.gz")
            gt_dest_path = join(gt_dest_folder, ID + ".nii.gz")
            shutil.copy(img_src_file, img_dest_path)
            shutil.copy(gt_src_file, gt_dest_path)

    if config["ntarget"] == "2":
        img_files = natsorted(glob(join(img_src_folder, "*.nii.gz")))
        gt1_files = natsorted(glob(join(gt1_src_folder, "*.nii.gz")))
        gt2_files = natsorted(glob(join(gt2_src_folder, "*.nii.gz")))
        for img_src_file, gt1_src_file, gt2_src_file in zip_longest(
            img_files, gt1_files, gt2_files
        ):
            img_src = dt.Image(img_src_file)
            ID = img_src.individual_name

            img1_dest_path = join(img_dest_folder, "a" + ID + ".nii.gz")
            img2_dest_path = join(img_dest_folder, "b" + ID + ".nii.gz")
            gt1_dest_path = join(gt_dest_folder, "a" + ID + ".nii.gz")
            gt2_dest_path = join(gt_dest_folder, "b" + ID + ".nii.gz")

            shutil.copy(img_src_file, img1_dest_path)
            shutil.copy(img_src_file, img2_dest_path)
            shutil.copy(gt1_src_file, gt1_dest_path)
            shutil.copy(gt2_src_file, gt2_dest_path)

    return
