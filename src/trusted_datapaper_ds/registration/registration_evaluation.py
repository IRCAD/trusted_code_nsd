"""
In this file, we analyse the fused manual segmentations versus each annotator.
"""
import copy
import os
import re
from glob import glob
from os.path import join

import numpy as np
import open3d as o3d
import pandas as pd
import SimpleITK as sitk
import yaml
from natsort import natsorted
from registration_utils import array_to_itkmask, voxelization

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.metrics import Dice, HausMesh, MeanNNDistance, MeanTRE
from trusted_datapaper_ds.utils import makedir, parse_args


def loader(
    refimg_files,
    Tinit_files,
    Tfine_files,
    megtfix_files,
    megtmov_files,
    ldmov_files,
    ldfix_files,
):
    """
    This function iteratively loads and yields data for registration evaluation.
    Args:
        refimg_files (list): List of paths to reference image files.
        Tinit_files (list): List of paths to initial transformation files.
        Tfine_files (list): List of paths to fine deformation field files.
        megtfix_files (list): List of paths to fixed mesh files.
        megtmov_files (list): List of paths to moving mesh files.
        ldmov_files (list): List of paths to moving landmark files.
        ldfix_files (list): List of paths to fixed landmark files.

    Yields:
        tuple: A tuple containing:
            - ID: Subject ID.
            - iteration: Registration iteration number.
            - ref_itk: Reference image as ITK image.
            - Tinit: Initial transformation matrix.
            - Tfine: Refinement transformation matrix.
            - megtfix : Fixed mesh object.
            - megtmov : Moving mesh object.
            - ldfix : Fixed landmarks object.
            - ldmov: Moving landmarks object.
    """

    for Tfine_file in Tfine_files:
        Tfine_basename = os.path.basename(Tfine_file)
        a = re.search("Tfine", Tfine_basename).start()
        b = re.search("Tfine", Tfine_basename).end()
        c = re.search(".txt", Tfine_basename).start()
        ID = Tfine_basename[:a]
        iteration = Tfine_basename[b:c]

        refimg_basename = ID[:-1] + "_imgCT.nii.gz"
        refimg_file = next(s for s in refimg_files if refimg_basename in s)
        refimg = dt.Image(refimg_file)
        ref_itk = refimg.itkimg

        Tfine = np.loadtxt(Tfine_file)

        Tinit_basename = ID + "Tinit" + str(iteration) + ".txt"
        Tinit_file = next(s for s in Tinit_files if Tinit_basename in s)
        Tinit = np.loadtxt(Tinit_file)

        megtfix_basename = ID + "meshfaceCT.obj"
        megtfix_file = next(s for s in megtfix_files if megtfix_basename in s)
        megtfix = dt.Mesh(megtfix_file, "gt")

        megtmov_basename = ID + "meshfaceUS.obj"
        megtmov_file = next(s for s in megtmov_files if megtmov_basename in s)
        megtmov = dt.Mesh(megtmov_file, "gt")

        ldfix_basename = ID + "_ldkCT.txt"
        ldfix_file = next(s for s in ldfix_files if ldfix_basename in s)
        ldfix = dt.Landmarks(ldfix_file, "gt")

        ldmov_basename = ID + "_ldkUS.txt"
        ldmov_file = next(s for s in ldmov_files if ldmov_basename in s)
        ldmov = dt.Landmarks(ldmov_file, "gt")

        yield ID, iteration, ref_itk, Tinit, Tfine, megtfix, megtmov, ldfix, ldmov


def regeval(ID, iteration, ref_itk, Tinit, Tfine, megtfix, megtmov, ldfix, ldmov):
    """
    Evaluates registration results and saves metrics.

    This function takes various inputs related to a registration process and performs the following evaluations:

    - Target Registration Error (TRE) between registered moving target point and fixed target point.
    - Dice score between ground truth mask and registered moving mask.
    - Hausdorff Distance (95th percentile) between ground truth and registered meshes.
    - Mean surface-to-surface nearest neighbor distance between ground truth and registered meshes.

    If any errors occur during the evaluation, an error message is printed, and `np.nan` is assigned
    to the corresponding metric.

    Args:
        ID (str): Identifier of the individual being processed.
        iteration (int): Repetition number of the registration process.
        ref_itk (sitk.Image): Reference image used for registration.
        Tinit (np.ndarray): Initial transformation matrix obtained from registration.
        Tfine (np.ndarray): Refinement transformation matrix obtained from registration.
        megtfix: Ground truth fixed mesh.
        megtmov: Ground truth moving mesh.
        ldfix: Ground truth fixed mask.
        ldmov: Ground truth moving mask.

    Returns:
        dict: A dictionary containing the following keys:
            - kidney_id (str): Individual ID.
            - iteration (int): Repetition number.
            - tre (float): Target Registration Error.
            - dice (float): Dice score between registered and ground truth masks.
            - h95mesh (float): Hausdorff distance (95th percentile) between meshes.
            - nndst (float): Mean surface-to-surface nearest neighbor distance between meshes.

    Raises:
        ValueError: If any errors occur during the evaluation process.
    """
    tre1 = np.nan
    dice1 = np.nan
    hmesh1 = np.nan
    nndst1 = np.nan

    print("Processing: ", ID)

    """Select the case where bcpd failed (NaN matrix)"""
    Tfine_flatten = Tfine.flatten()
    prod = np.prod(Tfine_flatten)
    assert not np.isnan(prod), "There is a NaN in this matrix. Abandon this case!"

    ldfix0 = ldfix.nparray[[0], :]
    ldmov0 = ldmov.nparray[[0], :]

    o3dfix0 = o3d.geometry.PointCloud()
    o3dfix0.points = o3d.utility.Vector3dVector(ldfix0)
    o3dmov0 = o3d.geometry.PointCloud()
    o3dmov0.points = o3d.utility.Vector3dVector(ldmov0)

    o3dfixpcd = megtfix.to_o3dpcd()
    o3dmovpcd = megtmov.to_o3dpcd()

    o3dmov0_Tinit = copy.deepcopy(o3dmov0).transform(Tinit)
    o3dmovpcd_Tinit = copy.deepcopy(o3dmovpcd).transform(Tinit)

    o3dmov0_Tfine = copy.deepcopy(o3dmov0_Tinit).transform(Tfine)
    o3dmovpcd_Tfine = copy.deepcopy(o3dmovpcd_Tinit).transform(Tfine)

    arrayfixpcd = voxelization(o3dfixpcd, ref_itk)
    arraymovpcd_Tfine = voxelization(o3dmovpcd_Tfine, ref_itk)

    fixitkmask = array_to_itkmask(arrayfixpcd, ref_itk)
    movitkmask_Tfine = array_to_itkmask(arraymovpcd_Tfine, ref_itk)

    fix_resampled_array = sitk.GetArrayFromImage(fixitkmask)
    mov_resampled_array = sitk.GetArrayFromImage(movitkmask_Tfine)

    # Evaluate TRE:
    try:
        d_TRE = MeanTRE()
        tre1 = d_TRE.evaluate_pcd(o3dmov0_Tfine, o3dfix0)
    except ValueError:
        error_message = (
            "There is an error when computing Dice for individual " + ID + ". \n"
        )
        print(error_message)

    # Evaluate Dice score and Hausdorff95 metrics between masks:
    try:
        dice = Dice(mov_resampled_array, fix_resampled_array)
        dice1 = dice.evaluate_dice()
    except ValueError:
        error_message = (
            "There is an error when computing Dice for individual " + ID + ". \n"
        )
        print(error_message)

    # Evaluate Hausdorf metric between meshes:
    try:
        hausmesh = HausMesh(95)
        hmesh1 = hausmesh.evaluate_pcd(o3dmovpcd_Tfine, o3dfixpcd)
    except ValueError:
        error_message = (
            "There is an error when computing HausdorffMask for individual "
            + ID
            + ". \n"
        )
        print(error_message)

    # Evaluate mean surface-to-surface nearest neighbour distance:
    try:
        d_nn = MeanNNDistance()
        nndst1 = d_nn.evaluate_pcd(o3dmovpcd_Tfine, o3dfixpcd)
    except ValueError:
        error_message = (
            "There is an error when computing nn_dist for individual " + ID + ". \n"
        )
        print(error_message)

    # Results saving
    values = {
        "kidney_id": ID,
        "iteration": iteration,
        "tre": tre1,
        "dice": dice1,
        "h95mesh": hmesh1,
        "nndst": nndst1,
    }

    return values


def main(
    csv_file,
    refimg_files,
    Tinit_files,
    Tfine_files,
    megtfix_files,
    megtmov_files,
    ldmov_files,
    ldfix_files,
):
    df = pd.DataFrame()

    for ID, iteration, ref_itk, Tinit, Tfine, megtfix, megtmov, ldfix, ldmov in loader(
        refimg_files,
        Tinit_files,
        Tfine_files,
        megtfix_files,
        megtmov_files,
        ldmov_files,
        ldfix_files,
    ):
        values = regeval(
            ID, iteration, ref_itk, Tinit, Tfine, megtfix, megtmov, ldfix, ldmov
        )
        df = df.append(values, ignore_index=True)
        # print(df)

    df.to_csv(csv_file, index=False)

    return


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    results_folder = config["regresults"]

    init_cases = ["ldks_transforms_std" + std_case for std_case in config["std_cases"]]

    for refinement_method in config["refinement_methods"]:
        for transform_model in config["transform_models"]:
            for std_case in config["std_cases"]:
                init_case = "ldks_transforms_std" + std_case
                refinement_case = (
                    refinement_method
                    + "_transforms_"
                    + transform_model
                    + "_std"
                    + std_case
                )
                print(refinement_case)
                makedir(join(results_folder, refinement_method, transform_model))

                csv_file = join(
                    results_folder,
                    refinement_method,
                    transform_model,
                    "std" + std_case + "results.csv",
                )

                refimg_folder = config["CTimg_origin0_location"]
                Tinit_folder = join(config["transfo_location"], init_case)
                Tfine_folder = join(config["transfo_location"], refinement_case)
                megtfix_folder = config["CTgtmesh_location"]
                megtmov_folder = config["USgtmesh_location"]
                ldmov_folder = config["USldks_location"]
                ldfix_folder = config["CTldks_location"]

                refimg_files = natsorted(glob(join(refimg_folder, "*.nii.gz")))
                Tinit_files = natsorted(glob(join(Tinit_folder, "*.txt")))
                Tfine_files = natsorted(glob(join(Tfine_folder, "*.txt")))
                megtfix_files = natsorted(glob(join(megtfix_folder, "*.obj")))
                megtmov_files = natsorted(glob(join(megtmov_folder, "*.obj")))
                ldmov_files = natsorted(glob(join(ldmov_folder, "*.txt")))
                ldfix_files = natsorted(glob(join(ldfix_folder, "*.txt")))

                main(
                    csv_file,
                    refimg_files,
                    Tinit_files,
                    Tfine_files,
                    megtfix_files,
                    megtmov_files,
                    ldmov_files,
                    ldfix_files,
                )
