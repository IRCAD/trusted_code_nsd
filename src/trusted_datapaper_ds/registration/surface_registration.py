import copy
from glob import glob
from os.path import join

import numpy as np
import vtk
import yaml
from landmarks_registration import ldks_transform
from natsort import natsorted
from registration_utils import create_pointcloud_polydata, vtkmatrix_to_numpy

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.utils import makedir, parse_args


def icp_without_distance_threshold(
    moving_pcd,
    fixed_pcd,
    max_number_pts: int,
    model: str,
    max_iter: int,
    tol: float,
    mov_noise_std=0,
    noising_iteration=0,
    output_folder=None,
):
    assert model in ["affine", "similarity", "rigid"]

    if model == "rigid":
        transform_type = "RigidBody"
    elif model == "similarity":
        transform_type = "Similarity"
    elif model == "affine":
        transform_type = "Affine"

    fixed_array = np.asarray(fixed_pcd.points)[
        np.random.randint(np.asarray(fixed_pcd.points).shape[0], size=max_number_pts), :
    ]
    moving_array = np.asarray(moving_pcd.points)[
        np.random.randint(np.asarray(moving_pcd.points).shape[0], size=max_number_pts),
        :,
    ]

    fixed_vtk, fixed_vtk_points = create_pointcloud_polydata(fixed_array)
    moving_vtk, moving_vtk_points = create_pointcloud_polydata(moving_array)

    print("Running ICP ----------------")
    icp_transform = vtk.vtkIterativeClosestPointTransform()
    icp_transform.SetTarget(fixed_vtk)
    icp_transform.SetSource(moving_vtk)

    if transform_type == "RigidBody":
        icp_transform.GetLandmarkTransform().SetModeToRigidBody()
    elif transform_type == "Similarity":
        icp_transform.GetLandmarkTransform().SetModeToSimilarity()
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
        "std" + str(mov_noise_std) + ".0"

        output_path = join(
            output_folder,
            "icp_transforms_" + icp_model + "_" + ldkfolder_suffix,
            ID + "Tfine" + str(noising_iteration) + ".txt",
        )

        np.savetxt(output_path, T)
        print("Tfine saved successfully !")

    return T


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    ldks_model = config["ldks_model"]
    movldk_location = config["USldks_location"]
    fixldk_location = config["CTldks_location"]
    movldk_files = natsorted(glob(join(movldk_location, "*_ldkUS.txt")))
    ldkreg_output_folder = config["initreg_location"]
    movldk_noise_std = 0
    number_of_iterations = 1
    ldkfolder_suffix = "std" + str(movldk_noise_std) + ".0"

    makedir(join(ldkreg_output_folder, "ldks_transforms_" + ldkfolder_suffix))

    icp_model = config["icp_model"]
    movmesh_location = config["USmesh_location"]
    fixmesh_location = config["CTmesh_location"]
    movmesh_files = natsorted(glob(join(movmesh_location, "*meshfaceUS.obj")))
    icpreg_output_folder = config["icpreg_location"]

    makedir(
        join(
            icpreg_output_folder, "icp_transforms_" + icp_model + "_" + ldkfolder_suffix
        )
    )

    for movmesh_file in movmesh_files:
        dt_movmesh = dt.Mesh(movmesh_file, "gt")
        ID = dt_movmesh.individual_name
        fixmesh_file = join(fixmesh_location, ID + "meshfaceCT.obj")
        dt_fixmesh = dt.Mesh(fixmesh_file, "gt")

        movldk_file = join(movldk_location, ID + "_ldkUS.txt")
        dt_movldk = dt.Landmarks(movldk_file, "gt")
        fixldk_file = join(fixldk_location, ID + "_ldkCT.txt")
        dt_fixldk = dt.Landmarks(fixldk_file, "gt")

        o3dpcd_mov = dt_movmesh.to_o3dpcd()
        o3dpcd_fix = dt_fixmesh.to_o3dpcd()

        for i in range(number_of_iterations):
            print("processing ID: ", ID)

            T_ldks = ldks_transform(
                dt_movldk,
                dt_fixldk,
                ldks_model,
                use234=True,
                mov_noise_std=movldk_noise_std,
                noising_iteration=i,
                output_folder=ldkreg_output_folder,
            )

            o3dpcd_mov_Tldks = copy.deepcopy(o3dpcd_mov).transform(T_ldks)

            T_icp = icp_without_distance_threshold(
                o3dpcd_mov_Tldks,
                o3dpcd_fix,
                max_number_pts=25000,
                model=config["icp_model"],
                max_iter=50,
                tol=1e-6,
                mov_noise_std=movldk_noise_std,
                noising_iteration=i,
                output_folder=icpreg_output_folder,
            )
