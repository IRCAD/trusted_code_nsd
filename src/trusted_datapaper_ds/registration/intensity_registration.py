import os
import signal
import subprocess
import time
from glob import glob
from os.path import join

import nibabel as nib
import numpy as np
import psutil
import SimpleITK as sitk
import yaml
from landmarks_registration import ldks_transform
from natsort import natsorted
from registration_utils import resample_itk

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.utils import makedir, parse_args


def single_trials_imfusion_transform(
    imfusion_workspace_file,
    model: str,
    similarity_metric,
    time_sleep,
    fix_temp_path,
    mov0_temp_path,
    mov1_temp_path,
):
    assert model in ["affine", "rigid"]

    if model == "rigid":
        subprocess.Popen(
            " ImFusionSuite "
            + imfusion_workspace_file
            + " fixed_img_path="
            + fix_temp_path
            + " moving_img_path="
            + mov0_temp_path
            + " rigid_transform="
            + str(1)
            + " affine_transform="
            + str(0)
            + " metric="
            + similarity_metric
            + " patchSize="
            + str(3)
            + " moved_img_path="
            + mov1_temp_path,
            shell=True,
        )

    if model == "affine":
        subprocess.Popen(
            " ImFusionSuite "
            + imfusion_workspace_file
            + " fixed_img_path="
            + fix_temp_path
            + " moving_img_path="
            + mov0_temp_path
            + " rigid_transform="
            + str(0)
            + " affine_transform="
            + str(1)
            + " metric="
            + similarity_metric
            + " patchSize="
            + str(3)
            + " moved_img_path="
            + mov1_temp_path,
            shell=True,
        )
    time.sleep(time_sleep)

    # Find the PID of ImFusionSuite process
    found_imfusion = False
    for proc in psutil.process_iter(["pid", "name"]):
        if "ImFusion" in proc.info["name"]:
            found_imfusion = True
            PID = proc.info["pid"]
            os.kill(PID, signal.SIGTERM)
            print(f"ImFusionSuite process with PID {PID} terminated")
            os.system("killall -s HUP ImFusionSuite")

    if not found_imfusion:
        print("ImFusionSuite process not found")

    moving_nib = nib.load(mov0_temp_path)
    moving_affine = moving_nib.affine

    moved_nib = nib.load(mov1_temp_path)
    moved_affine = moved_nib.affine

    T = moved_affine @ np.linalg.inv(moving_affine)

    del moved_nib, moving_nib
    os.remove(mov0_temp_path)
    os.remove(mov1_temp_path)

    return T


def many_trials_imfusion_transform(
    imfusion_workspace_file,
    ID,
    model: str,
    similarity_metric,
    iteration,
    init_time_sleep,
    max_number_of_trials,
    fix_temp_path,
    mov0_temp_path,
    mov1_temp_path,
    output_folder,
):
    time_sleep = init_time_sleep
    Tnocomputed = True
    counter = 0
    while Tnocomputed and counter <= max_number_of_trials:
        try:
            T = single_trials_imfusion_transform(
                imfusion_workspace_file,
                model,
                similarity_metric,
                time_sleep,
                fix_temp_path,
                mov0_temp_path,
                mov1_temp_path,
            )
            Tnocomputed = False
        except Exception as e:
            counter += 1
            time_sleep += init_time_sleep
            print("Exception found: ", e)
            print("......... wait, I'm trying again with more waiting time!")

    if Tnocomputed:
        print(
            "Finnally, I was not able to compute the transform after "
            + str(max_number_of_trials)
            + " trials, and a max time equal to :",
            str(max_number_of_trials * init_time_sleep),
            "seconds",
        )
        print("I returned None.")
        T = None
    else:
        # imfusion outputs saving
        if output_folder is not None:
            output_path = join(output_folder, ID + "Tfine" + str(iteration) + ".txt")
            np.savetxt(output_path, T)
            print("Tfine saved successfully as: ", output_path)

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
    init_time_sleep = 5  # Note about the max_time_sleep (4*time_sleep)
    max_number_of_trials = 4

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
        makedir(imfspecific_output_folder)
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

        fiximg0_itk = dt_fiximg.shift_origin(new_origin=[0, 0, 0], shifted_dirname=None)
        sitk.WriteImage(fiximg0_itk, fix_temp_path)
        fiximg0_nib = nib.load(fix_temp_path)

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
            print("T_ldks computed")
            movimg_itk = dt_movimg.itkimg
            movimg_itk_Tldks = resample_itk(movimg_itk, T_ldks)
            sitk.WriteImage(movimg_itk_Tldks, mov0_temp_path)
            movimg_nib_Tldks = nib.load(mov0_temp_path)
            print("movimg_nib_Tldks loaded")

            Tfine = many_trials_imfusion_transform(
                imfusion_workspace_file=imfusion_workspace_file,
                ID=ID,
                model=config["refine_model"],
                similarity_metric=config["similarity_metric"],
                iteration=i,
                init_time_sleep=init_time_sleep,
                max_number_of_trials=max_number_of_trials,
                fix_temp_path=fix_temp_path,
                mov0_temp_path=mov0_temp_path,
                mov1_temp_path=mov1_temp_path,
                output_folder=imfspecific_output_folder,
            )
