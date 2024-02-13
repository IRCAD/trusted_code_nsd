from glob import glob
from os.path import join

import nibabel as nib
import numpy as np
import yaml
from natsort import natsorted

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.utils import makedir, parse_args


def reupsampling_nib(imgnib, prednib, modality, clean=True):
    squared_diff_affine_matrix = (imgnib.affine - prednib.affine) ** 2

    assert (
        np.max(squared_diff_affine_matrix) < 1e-6
    ), "imgnib and prednib seem to be different"
    assert modality.upper() in ["CT", "US"], "modality has to be set in ['CT', 'US'] "

    newsize = list(imgnib.shape)

    resized_prednib = dt.resiz_nib_data(
        prednib, newsize, interpolmode="trilinear", binary=True
    )

    if clean:
        if modality == "US":
            resized_prednib = dt.clean_nibmask(resized_prednib, number_of_kidney=1)
        if modality == "CT":
            resized_prednib = dt.clean_nibmask(resized_prednib, number_of_kidney=2)

    return resized_prednib


def reupsampling_and_save_nii_list(config, predpath_list, output_folder, clean=True):
    annotatorID = config["auto"]
    img_folder = config["img_location"]

    makedir(output_folder)

    for ind in predpath_list:
        maskpath = ind
        mask = dt.Mask(maskpath, annotatorID)
        ID = mask.individual_name

        modality = config["modality"]

        imgpath = join(img_folder, ID + config[modality + "img_end"])
        img = dt.Image(imgpath)

        img_nib = img.nibimg
        pred_nib = mask.nibmask

        resized_prednib = reupsampling_nib(
            imgnib=img_nib, prednib=pred_nib, modality=modality, clean=clean
        )

        reupsampled_data_path = join(output_folder, mask.basename)
        nib.save(resized_prednib, reupsampled_data_path)

        print(
            "resizing and saving: ",
            img.individual_name,
            " to size ",
            resized_prednib.shape,
            " as ",
            reupsampled_data_path,
            " DONE",
        )

    return


def meshing_pcding_and_saving_nii_mask_list(
    annotatorID, maskpath_list, mesh_folder, pcd_folder
):
    makedir(mesh_folder)
    makedir(pcd_folder)

    for path in maskpath_list:
        maskpath = path
        mask = dt.Mask(maskpath, annotatorID)
        nib_data = mask.nibmask
        individual_name = mask.individual_name
        modality = mask.modality

        if modality == "CT":
            (
                o3d_meshCT_L,
                o3d_meshCT_R,
                o3d_pcdCT_L,
                o3d_pcdCT_R,
                mask_cleaned_nib,
            ) = dt.convert_to_mesh_and_pcd(
                nib_data,
                modality,
                individual_name,
                annotatorID,
                mesh_dirname=mesh_folder,
                pcd_dirname=pcd_folder,
                mask_cleaning=False,
            )

        if modality == "US":
            (
                o3d_meshUS,
                o3d_pcdUS,
                mask_cleaned_nib,
            ) = dt.convert_to_mesh_and_pcd(
                nib_data,
                modality,
                individual_name,
                annotatorID,
                mesh_dirname=mesh_folder,
                pcd_dirname=pcd_folder,
                mask_cleaning=False,
            )


def main1(config, predpath_list, output_folder):
    reupsampling_and_save_nii_list(config, predpath_list, output_folder, clean=False)
    return


def main2(config, maskpath_list, mesh_folder, pcd_folder):
    annotatorID = config["auto"]
    meshing_pcding_and_saving_nii_mask_list(
        annotatorID, maskpath_list, mesh_folder, pcd_folder
    )
    return


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    upsampling = bool(config["upsampling"])  # upsampling
    meshing = bool(config["meshing"])  # meshing
    splitCT = False

    list_training_target = config["list_training_target"]

    if config["modality"] == "CT":
        splitCT = bool(config["splitCT"])  # split CT

    if upsampling:
        print("UPSAMPLING")
        for segmodel in config["list_othermodels"]:
            for training_target in list_training_target:
                print(segmodel)
                print(training_target)
                img_folder = config["img_location"]
                input_folder = join(config["seg128location"], segmodel, training_target)
                predpath_list = natsorted(glob(join(input_folder, "*.nii.gz")))
                output_folder = join(
                    config["mask_seglocation"],
                    segmodel,
                    training_target,
                )
                main1(config, predpath_list, output_folder)

    if meshing:
        print("MESHING")
        segmodels = config["list_othermodels"] + config["list_UVnetmodels"]

        for segmodel in segmodels:
            for training_target in list_training_target:
                print(segmodel)
                print(training_target)
                pred_folder = join(
                    config["mask_seglocation"],
                    segmodel,
                    training_target,
                )
                maskpath_list = natsorted(glob(join(pred_folder, "*.nii.gz")))
                mesh_folder = join(
                    config["mesh_seglocation"],
                    segmodel,
                    training_target,
                )
                pcd_folder = join(config["pcd_seglocation"], segmodel, training_target)
                main2(config, maskpath_list, mesh_folder, pcd_folder)

    if splitCT:
        print("SPLIT CT")
        segmodels = config["list_othermodels"] + config["list_UVnetmodels"]

        for segmodel in segmodels:
            for training_target in list_training_target:
                print(segmodel)
                print(training_target)
                mask_folder = join(
                    config["mask_seglocation"],
                    segmodel,
                    training_target,
                )
                maskpath_list = natsorted(glob(join(mask_folder, "*.nii.gz")))
                split_dirname = join(
                    config["splitmask_seglocation"],
                    segmodel,
                    training_target,
                )
                makedir(split_dirname)

                for maskpath in maskpath_list:
                    ctmask = dt.Mask(maskpath, annotatorID=config["auto"])
                    nibL, nibR = ctmask.split(split_dirname=split_dirname)
