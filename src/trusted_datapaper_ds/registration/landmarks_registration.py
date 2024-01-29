from glob import glob
from os.path import join

import numpy as np
import vtk
import yaml
from natsort import natsorted

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.utils import makedir, parse_args


def compute_transform(
    dt_mov,
    dt_fix,
    model,
    use234=True,
    mov_noise_std=None,
    noising_iteration=0,
    output_folder=None,
):  # In real resolution
    assert model in ["affine", "similarity", "rigid"]
    assert (
        dt_fix.number_of_ldks == dt_mov.number_of_ldks
    ), "mov and fix must have the same number of points."

    if mov_noise_std is None or mov_noise_std == 0:
        assert (
            noising_iteration == 0
        ), "If mov_noise_std is None or 0, noising_iteration must be 0"

    if mov_noise_std is not None:
        dt_mov.noising(mov_noise_std)

    if use234:
        mov = dt_mov.nparray[[2, 3, 4], :]
        fix = dt_fix.nparray[[2, 3, 4], :]
    else:
        mov = dt_mov.nparray
        fix = dt_fix.nparray

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

    if output_folder is not None:
        if mov_noise_std is None:
            folder_suffix = "std" + str(0.0)
        else:
            folder_suffix = "std" + str(mov_noise_std) + ".0"

        ID = dt_fix.individual_name

        if noising_iteration == 0:
            output_path = join(
                output_folder, "ldks_transforms_" + folder_suffix, ID + "Tinit.txt"
            )
        else:
            output_path = join(
                output_folder,
                "ldks_transforms_" + folder_suffix,
                ID + "Tinit" + str(noising_iteration) + ".txt",
            )

        np.savetxt(output_path, T)

    return T


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    model = config["ldks_model"]
    mov_location = config["USldks_location"]
    fix_location = config["CTldks_location"]
    mov_files = natsorted(glob(join(mov_location, "*_ldkUS.txt")))

    output_folder = config["initreg_location"]

    mov_noise_std = 0
    number_of_iterations = 1

    if mov_noise_std is None:
        folder_suffix = "std" + str(0.0)
    else:
        folder_suffix = "std" + str(mov_noise_std) + ".0"

    makedir(join(output_folder, "ldks_transforms_" + folder_suffix))

    for mov_file in mov_files:
        dt_mov = dt.Landmarks(mov_file, "gt")
        ID = dt_mov.individual_name
        fix_file = join(fix_location, ID + "_ldkCT.txt")
        dt_fix = dt.Landmarks(fix_file, "gt")

        for i in range(number_of_iterations):
            print("processing ID: ", ID)
            compute_transform(
                dt_mov,
                dt_fix,
                model,
                use234=True,
                mov_noise_std=mov_noise_std,
                noising_iteration=i,
                output_folder=output_folder,
            )
