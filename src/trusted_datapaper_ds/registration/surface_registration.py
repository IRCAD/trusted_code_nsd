import copy
import os
from glob import glob
from os.path import join

import numpy as np
import open3d as o3d
import vtk
import yaml
from landmarks_registration import compute_landmarks_transform, ldks_transform
from natsort import natsorted
from registration_utils import create_pointcloud_polydata, vtkmatrix_to_numpy

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.utils import makedir, parse_args


def icp_transform(
    moving_pcd,
    fix_pcd,
    max_number_pts: int,
    model: str,
    max_iter: int,
    tol: float,
    iteration,
    output_folder,
):
    assert model in ["affine", "rigid"]

    if model == "rigid":
        transform_type = "RigidBody"
    elif model == "affine":
        transform_type = "Affine"

    fix_array = np.asarray(fix_pcd.points)[
        np.random.randint(np.asarray(fix_pcd.points).shape[0], size=max_number_pts), :
    ]
    moving_array = np.asarray(moving_pcd.points)[
        np.random.randint(np.asarray(moving_pcd.points).shape[0], size=max_number_pts),
        :,
    ]

    fix_vtk, fix_vtk_points = create_pointcloud_polydata(fix_array)
    moving_vtk, moving_vtk_points = create_pointcloud_polydata(moving_array)

    print("Running ICP ----------------")
    icp_transform = vtk.vtkIterativeClosestPointTransform()
    icp_transform.SetTarget(fix_vtk)
    icp_transform.SetSource(moving_vtk)

    if transform_type == "RigidBody":
        icp_transform.GetLandmarkTransform().SetModeToRigidBody()
    elif transform_type == "Affine":
        icp_transform.GetLandmarkTransform().SetModeToAffine()

    icp_transform.SetMaximumNumberOfIterations(max_iter)
    icp_transform.StartByMatchingCentroidsOff()
    icp_transform.SetMaximumNumberOfLandmarks(max_number_pts)
    icp_transform.SetMaximumMeanDistance(tol)

    icp_transform.Update()

    matrix = icp_transform.GetMatrix()

    T = vtkmatrix_to_numpy(matrix)

    if output_folder is not None:
        output_path = join(output_folder, ID + "Tfine" + str(iteration) + ".txt")
        np.savetxt(output_path, T)
        print("Tfine saved successfully !")

    return T


def bcpd_transform(
    regpack_dir,
    ID,
    moving_pcd,
    fix_pcd,
    model: str,
    iteration,
    temp_folder,
    output_folder,
):
    assert model in ["affine", "rigid"]

    print("Running BCPD ----------------")

    # temporal saving, because bcpd needs this data format
    temp_moving_pcd_path = join(temp_folder, ID + "_temp_moving.txt")
    temp_fix_pcd_path = join(temp_folder, ID + "_temp_fix.txt")
    moving_pcd_nparray = np.asarray(moving_pcd.points)
    fix_pcd_nparray = np.asarray(fix_pcd.points)
    np.savetxt(temp_moving_pcd_path, moving_pcd_nparray, delimiter=",")
    np.savetxt(temp_fix_pcd_path, fix_pcd_nparray, delimiter=",")

    # bcpd
    if model == "rigid":
        os.system(
            join(regpack_dir, "bcpd-master", "bcpd")
            + " -y "
            + temp_moving_pcd_path
            + " -x "
            + temp_fix_pcd_path
            + " -w0.1 -l1e9 -g0.1 -ux  -K70 -J300 -n50 -N30 -p -d7 -e0.3 -f0.3 -o "
            + ID
            + " -sy"
        )
    if model == "affine":
        os.system(
            join(regpack_dir, "bcpd-master", "bcpd")
            + " -y "
            + temp_moving_pcd_path
            + " -x "
            + temp_fix_pcd_path
            + " -w0.1 -l1e4 -g0.1 -ux  -K70 -J300 -n50 -N30 -p -d7 -e0.3 -f0.3 -o "
            + ID
            + " -sy"
        )

    currentdir = os.getcwd()

    os.rename(
        join(currentdir, ID + "y.txt"),
        join(currentdir, ID + "y" + str(iteration) + "_" + model + ".txt"),
    )
    os.remove(join(currentdir, ID + "comptime.txt"))
    os.remove(join(currentdir, ID + "info.txt"))

    # collect the outputs and check the registration
    y_path = join(currentdir, ID + "y" + str(iteration) + "_" + model + ".txt")
    y = np.genfromtxt(y_path)

    # estimate a 4*4 transform matrix
    if model == "rigid":
        T = compute_landmarks_transform(mov=moving_pcd_nparray, fix=y, model="rigid")

    if model == "affine":
        T = compute_landmarks_transform(mov=moving_pcd_nparray, fix=y, model="affine")

    # clean folder
    os.remove(y_path)
    os.remove(temp_moving_pcd_path)

    # bcpd outputs saving
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

    visualization = False
    icp = config["regmethod"] == "ICP"
    bcpd = config["regmethod"] == "BCPD"

    us_color = np.array([0.0, 1.0, 0.0])
    ct_color = np.array([1.0, 0.0, 0.0])

    ldks_model = config["ldks_model"]
    movldk_location = config["USldks_location"]
    fixldk_location = config["CTldks_location"]
    movldk_noise_std = config["noise_std"]
    number_of_iterations = int(config["iternumb"])
    movldk_files = natsorted(glob(join(movldk_location, "*_ldkUS.txt")))

    ldkfolder_suffix = "std" + str(movldk_noise_std) + ".0"

    assert number_of_iterations > 0, "You should set at least one iteration"

    refine_model = config["refine_model"]
    movmesh_location = join(
        config["USautomesh_location"], config["segmodel"], config["training_target"]
    )
    fixmesh_location = join(
        config["CTautomesh_location"], config["segmodel"], config["training_target"]
    )
    movmesh_files = natsorted(glob(join(movmesh_location, "*meshfaceUS.obj")))

    ldkreg_output_folder = config["transfo_location"]
    if ldkreg_output_folder is not None:
        ldkregspecific_output_folder = join(
            ldkreg_output_folder, "ldks_transforms_" + ldkfolder_suffix
        )
        makedir(ldkregspecific_output_folder)
    else:
        ldkregspecific_output_folder = None

    if icp:
        icpreg_output_folder = config["transfo_location"]
        if icpreg_output_folder is not None:
            icpspecific_output_folder = join(
                icpreg_output_folder,
                "ICP_transforms_" + refine_model + "_" + ldkfolder_suffix,
            )
            makedir(icpspecific_output_folder)
        else:
            icpspecific_output_folder = None

    if bcpd:
        bcpdreg_output_folder = config["transfo_location"]
        if bcpdreg_output_folder is not None:
            bcpdspecific_output_folder = join(
                bcpdreg_output_folder,
                "BCPD_transforms_" + refine_model + "_" + ldkfolder_suffix,
            )
            makedir(bcpdspecific_output_folder)
        else:
            bcpdspecific_output_folder = None

        bcpd_temp_folder = config["BCPD_temp_folder"]
        makedir(bcpd_temp_folder)

    for movmesh_file in movmesh_files:
        dt_movmesh = dt.Mesh(movmesh_file, "gt")
        ID = dt_movmesh.individual_name
        print("processing ID: ", ID)
        fixmesh_file = join(fixmesh_location, ID + "meshfaceCT.obj")
        dt_fixmesh = dt.Mesh(fixmesh_file, "gt")

        movldk_file = join(movldk_location, ID + "_ldkUS.txt")
        dt_movldk = dt.Landmarks(movldk_file, "gt")
        fixldk_file = join(fixldk_location, ID + "_ldkCT.txt")
        dt_fixldk = dt.Landmarks(fixldk_file, "gt")

        o3dpcd_mov = dt_movmesh.to_o3dpcd()
        o3dpcd_mov.paint_uniform_color(color=us_color)
        o3dpcd_fix = dt_fixmesh.to_o3dpcd()
        o3dpcd_fix.paint_uniform_color(color=ct_color)

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

            o3dpcd_mov_Tldks = copy.deepcopy(o3dpcd_mov).transform(T_ldks)
            o3dpcd_mov_Tldks.paint_uniform_color(color=us_color)

            if icp:
                Tfine = icp_transform(
                    o3dpcd_mov_Tldks,
                    o3dpcd_fix,
                    max_number_pts=2500,
                    model=config["refine_model"],
                    max_iter=50,
                    tol=1e-6,
                    iteration=i,
                    output_folder=icpspecific_output_folder,
                )

            if bcpd:
                Tfine = bcpd_transform(
                    regpack_dir=config["regpack_dir"],
                    ID=ID,
                    moving_pcd=o3dpcd_mov_Tldks,
                    fix_pcd=o3dpcd_fix,
                    model=config["refine_model"],
                    iteration=i,
                    temp_folder=bcpd_temp_folder,
                    output_folder=bcpdspecific_output_folder,
                )

            o3dpcd_mov_Tfine = copy.deepcopy(o3dpcd_mov_Tldks).transform(Tfine)
            o3dpcd_mov_Tfine.paint_uniform_color(color=us_color)

            if visualization:
                o3d.visualization.draw_geometries(
                    [o3dpcd_mov, o3dpcd_fix],
                    width=720,
                    height=720,
                    window_name=ID + " Before initialization (fix-red, mov-green)",
                )
                o3d.visualization.draw_geometries(
                    [o3dpcd_mov_Tldks, o3dpcd_fix],
                    width=720,
                    height=720,
                    window_name=ID + " After initialization (fix-red, mov-green)",
                )
                o3d.visualization.draw_geometries(
                    [o3dpcd_mov_Tfine, o3dpcd_fix],
                    width=720,
                    height=720,
                    window_name=ID + " After refinement (fix-red, mov-green)",
                )
