import os
from glob import glob
from os.path import join

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import yaml
from landmarks_registration import ldks_transform
from natsort import natsorted
from registration_utils import resample_itk

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.utils import makedir, parse_args


def imfusion_transform(
    imfusion_workspace_file,
    ID,
    moving_nib,
    fix_nib,
    model: str,
    similarity_metric,
    iteration,
    fix_temp_folder,
    mov0_temp_folder,
    mov1_temp_folder,
    output_folder,
):
    assert model in ["affine", "rigid"]

    moving_affine = moving_nib.affine

    mov0_temp_path = join(mov0_temp_folder, ID + "tempUS0.nii")
    mov1_temp_path = join(mov1_temp_folder, ID + "tempUS1.nii")
    fix_temp_path = join(fix_temp_folder, ID + "tempCT.nii")

    """ Data reading """
    nib.save(moving_nib, mov0_temp_path)
    nib.save(fix_nib, fix_temp_path)

    if model == "rigid":
        os.system(
            "ImFusionSuite"
            + imfusion_workspace_file
            + "fixed_img_path="
            + fix_temp_path
            + "moving_img_path="
            + mov0_temp_path
            + "rigid_transform="
            + str(1)
            + "affine_transform="
            + str(0)
            + "metric="
            + similarity_metric
            + "patchSize="
            + str(3)
            + "moved_img_path="
            + mov1_temp_path
        )
        os.system("taskkill /im ImFusionSuite.exe")

    if model == "affine":
        os.system(
            "ImFusionSuite"
            + imfusion_workspace_file
            + "fixed_img_path="
            + fix_temp_path
            + "moving_img_path="
            + mov0_temp_path
            + "rigid_transform="
            + str(0)
            + "affine_transform="
            + str(1)
            + "metric="
            + similarity_metric
            + "patchSize="
            + str(3)
            + "moved_img_path="
            + mov1_temp_path
        )
        os.system("taskkill /im ImFusionSuite.exe")

    moved_nib = nib.load(mov1_temp_path)
    moved_affine = moved_nib.affine
    del moved_nib

    os.remove(mov0_temp_path)
    os.remove(mov1_temp_path)

    T = moved_affine @ np.linalg.inv(moving_affine)

    # imfusion outputs saving
    if output_folder is not None:
        output_path = join(output_folder, ID + "Tfine" + str(iteration) + ".txt")
        np.savetxt(output_path, T)
        print("Tfine saved successfully !")

    return T


if __name__ == "__main__":
    np.random.seed(0)

    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    imfusion_workspace_file = join(
        config["regpack_dir"], "imfusion_workspace", "model.iws"
    )

    ldks_model = config["ldks_model"]
    movldk_location = config["USldks_location"]
    fixldk_location = config["CTldks_location"]
    movldk_files = natsorted(glob(join(movldk_location, "*_ldkUS.txt")))
    movldk_noise_std = 0
    number_of_iterations = 1

    ldkfolder_suffix = "std" + str(movldk_noise_std) + ".0"

    similarity_metric = config["similarity_metric"]

    assert number_of_iterations > 0, "You should set at least one iteration"

    refine_model = config["refine_model"]

    movimg_location = config["USimg_location"]
    fiximg_location = config["CTimg_location"]
    movimg_files = natsorted(glob(join(movimg_location, "*_imgUS.nii.gz")))

    ldkreg_output_folder = config["initreg_location"]
    if ldkreg_output_folder is not None:
        ldkregspecific_output_folder = join(
            ldkreg_output_folder, "ldks_transforms_" + ldkfolder_suffix
        )
        makedir(ldkregspecific_output_folder)
    else:
        ldkregspecific_output_folder = None

    imfreg_output_folder = config["intensityreg_location"]
    if imfreg_output_folder is not None:
        imfspecific_output_folder = join(
            imfreg_output_folder,
            "imf"
            + similarity_metric
            + "_transforms_"
            + refine_model
            + "_"
            + ldkfolder_suffix,
        )
        makedir(imfreg_output_folder)
    else:
        imfspecific_output_folder = None

    fix_temp_folder = join(config["imf_temp_folder"], "imf_fix_temp")
    makedir(fix_temp_folder)
    mov0_temp_folder = join(config["imf_temp_folder"], "imf_mov0_temp")
    makedir(mov0_temp_folder)
    mov1_temp_folder = join(config["imf_temp_folder"], "imf_mov1_temp")
    makedir(mov1_temp_folder)

    for movimg_file in movimg_files:
        dt_movimg = dt.Image(movimg_file)
        ID = dt_movimg.individual_name
        print("processing ID: ", ID)
        IDfix = ID[:-1]
        fiximg_file = join(fiximg_location, IDfix + "_imgCT.nii.gz")
        dt_fiximg = dt.Image(fiximg_file)

        mov0_temp_path = join(mov0_temp_folder, ID + "mov0temp.nii")
        mov1_temp_path = join(mov1_temp_folder, ID + "mov1temp.nii")
        fix_temp_path = join(fix_temp_folder, ID + "fixtemp.nii")

        fiximg0_nib = dt_fiximg.origin(new_origin=[0, 0, 0], shifted_dirname=None)

        movldk_file = join(movldk_location, ID + "_ldkUS.txt")
        dt_movldk = dt.Landmarks(movldk_file, "gt")
        fixldk_file = join(fixldk_location, ID + "_ldkCT.txt")
        dt_fixldk = dt.Landmarks(fixldk_file, "gt")

        for i in range(number_of_iterations):
            T_ldks = ldks_transform(
                dt_movldk,
                dt_fixldk,
                ldks_model,
                use234=True,
                mov_noise_std=movldk_noise_std,
                iteration=i,
                output_folder=ldkregspecific_output_folder,
            )

            movimg_itk = dt_movimg.itkimg
            movimg_itk_Tldks = resample_itk(movimg_itk, T_ldks)
            sitk.WriteImage(movimg_itk_Tldks, mov0_temp_path)
            movimg_nib_Tldks = nib.load(mov0_temp_path)

            Tfine = imfusion_transform(
                imfusion_workspace_file,
                ID=ID,
                moving_nib=movimg_nib_Tldks,
                fix_nib=fiximg0_nib,
                model=config["refine_model"],
                similarity_metric=config["similarity_metric"],
                iteration=i,
                fix_temp_folder=fix_temp_folder,
                mov0_temp_folder=mov0_temp_folder,
                mov1_temp_folder=mov1_temp_folder,
                output_folder=imfspecific_output_folder,
            )

            print(Tfine)
