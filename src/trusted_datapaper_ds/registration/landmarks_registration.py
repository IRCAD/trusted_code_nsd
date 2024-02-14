from glob import glob
from os.path import join

import numpy as np
import vtk
import yaml
from natsort import natsorted

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.utils import makedir, parse_args


def compute_landmarks_transform(mov, fix, model):
    """
    Computes 3D transformation matrix between two sets of landmarks.

    This function calculates a transformation matrix (4x4) that aligns one set of
    3D landmarks (`mov`) to another (`fix`) using a specified transformation model.

    Args:
        mov (np.ndarray): Array of moving landmark coordinates (n x 3).
        fix (np.ndarray): Array of fixed landmark coordinates (n x 3).
        model (str): Transformation model ("affine", "similarity", or "rigid").

    Returns:
        np.ndarray: The 4x4 transformation matrix aligning `mov` to `fix`.

    Raises:
        AssertionError: If an invalid `model` is provided.
    """
    assert model in ["affine", "similarity", "rigid"]
    n = fix.shape[0]
    fix_point = vtk.vtkPoints()
    fix_point.SetNumberOfPoints(n)
    mov_point = vtk.vtkPoints()
    mov_point.SetNumberOfPoints(n)

    for i in range(n):
        fix_point.SetPoint(i, fix[i, 0], fix[i, 1], fix[i, 2])
        mov_point.SetPoint(i, mov[i, 0], mov[i, 1], mov[i, 2])

    LandmarkTransform = vtk.vtkLandmarkTransform()

    if model == "affine":
        LandmarkTransform.SetModeToAffine()
    if model == "similarity":
        LandmarkTransform.SetModeToSimilarity()
    if model == "rigid":
        LandmarkTransform.SetModeToRigidBody()

    LandmarkTransform.SetSourceLandmarks(mov_point)
    LandmarkTransform.SetTargetLandmarks(fix_point)
    LandmarkTransform.Update()

    matrix = LandmarkTransform.GetMatrix()
    T = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            T[i, j] = matrix.GetElement(i, j)

    return T


def ldks_transform(
    dt_mov,
    dt_fix,
    model,
    use234=True,
    mov_noise_std=0,
    iteration=0,
    output_folder=None,
):
    """
    Estimates a 3D transformation matrix using landmark-based registration.

    This function performs landmark-based image registration to estimate a
    transformation matrix (4x4) aligning a moving landmark set (`dt_mov`) to a
    fixed landmark set (`dt_fix`). It uses a specified transformation model
    ("affine", "similarity", or "rigid").

    Args:
        dt_mov (dt.Landmarks): Moving landmark object.
        dt_fix (dt.Landmarks): Fixed landmark object.
        model (str): Transformation model used for registration ("similarity").
        use234 (bool, optional): Whether to use only landmarks 2, 3, and 4
            (default: True).
        mov_noise_std (float, optional): Standard deviation of noise added to
            moving landmarks (default: 0.0).
        iteration (int, optional): Repetition number of the registration.
        output_folder (str, optional): Path to save the transformation matrix
            (default: None).

    Returns:
        np.ndarray: The 4x4 transformation matrix aligning `dt_mov` to `dt_fix`.

    Raises:
        AssertionError: If the moving and fixed sets have different numbers of landmarks.
    """
    assert (
        dt_fix.number_of_ldks == dt_mov.number_of_ldks
    ), "mov and fix must have the same number of points."

    if mov_noise_std != 0:
        dt_mov.noising(mov_noise_std)

    if use234:
        mov = dt_mov.nparray[[2, 3, 4], :]
        fix = dt_fix.nparray[[2, 3, 4], :]
    else:
        mov = dt_mov.nparray
        fix = dt_fix.nparray

    T = compute_landmarks_transform(mov, fix, model)

    if output_folder is not None:
        if mov_noise_std == 0:
            "std" + str(0.0)
        else:
            "std" + str(mov_noise_std) + ".0"

        ID = dt_fix.individual_name

        output_path = join(
            output_folder,
            ID + "Tinit" + str(iteration) + ".txt",
        )

        np.savetxt(output_path, T)

    return T


if __name__ == "__main__":
    np.random.seed(0)

    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    ldks_model = config["ldks_model"]

    movldk_location = config["USldks_location"]
    fixldk_location = config["CTldks_location"]

    movldk_files = natsorted(glob(join(movldk_location, "*_ldkUS.txt")))

    ldkreg_output_folder = config["transfo_location"]

    movldk_noise_std = config["noise_std"]
    number_of_iterations = int(config["iternumb"])

    ldkfolder_suffix = "std" + str(movldk_noise_std) + ".0"

    if ldkreg_output_folder is not None:
        ldkregspecific_output_folder = join(
            ldkreg_output_folder, "ldks_transforms_" + ldkfolder_suffix
        )
        makedir(ldkregspecific_output_folder)
    else:
        ldkregspecific_output_folder = None

    for movldk_file in movldk_files:
        dt_movldk = dt.Landmarks(movldk_file, "gt")
        ID = dt_movldk.individual_name
        fixldk_file = join(fixldk_location, ID + "_ldkCT.txt")
        dt_fixldk = dt.Landmarks(fixldk_file, "gt")

        for i in range(number_of_iterations):
            print("processing ID: ", ID)
            ldks_transform(
                dt_movldk,
                dt_fixldk,
                ldks_model,
                use234=True,
                mov_noise_std=movldk_noise_std,
                iteration=i,
                output_folder=ldkregspecific_output_folder,
            )
