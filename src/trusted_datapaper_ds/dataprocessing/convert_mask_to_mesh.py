from os.path import join

import yaml
from natsort import natsorted

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.utils import build_many_mask_analist, makedir, parse_args


def main(
    config,
    ctlist,
    uslist,
    CTmask_to_mesh,
    USmask_to_mesh,
):
    ann = config["annotator_mask_to_mesh"]

    if CTmask_to_mesh:
        USlike_IDlist = None
        CTlike_IDlist = ctlist

        ctma1_files, ctma2_files, ctmagt_files = build_many_mask_analist(
            "CT", config, USlike_IDlist, CTlike_IDlist
        )
        if ann == "1":
            ctmask_files = ctma1_files
        elif ann == "2":
            ctmask_files = ctma2_files
        elif ann == "gt":
            ctmask_files = ctmagt_files

        mesh_dirname = join(config["myDATA"], config["CTme" + ann + "fol"])
        makedir(mesh_dirname)
        for ctmask_file in ctmask_files:
            ctmask = dt.Mask(ctmask_file, annotatorID=ann)
            (
                o3d_meshCT_L,
                o3d_meshCT_R,
                o3d_pcdCT_L,
                o3d_pcdCT_R,
                mask_cleaned_nib,
            ) = ctmask.to_mesh_and_pcd(
                mesh_dirname=mesh_dirname,
                pcd_dirname=None,
                mask_cleaning=False,
            )

    # Convert list of Masks to Meshes and PCDs (here US ground truth)###
    # Note: "mesh_dirname" is the directory to save the mesh,
    #       "pcd_dirname" is the directory to save the point cloud
    if USmask_to_mesh:
        USlike_IDlist = uslist
        CTlike_IDlist = None

        usma1_files, usma2_files, usmagt_files = build_many_mask_analist(
            "US", config, USlike_IDlist, CTlike_IDlist
        )
        if ann == "1":
            usmask_files = usma1_files
        elif ann == "2":
            usmask_files = usma2_files
        elif ann == "gt":
            usmask_files = usmagt_files

        mesh_dirname = join(config["myDATA"], config["USme" + ann + "fol"])
        makedir(mesh_dirname)
        for usmask_file in usmask_files:
            usmask = dt.Mask(usmask_file, annotatorID=ann)
            o3d_meshUS, o3d_pcdUS, mask_cleaned_nib = usmask.to_mesh_and_pcd(
                mesh_dirname=mesh_dirname,
                pcd_dirname=None,
                mask_cleaning=False,
            )

    return


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    allct = natsorted(
        config["CTfold"]["cv1"]
        + config["CTfold"]["cv2"]
        + config["CTfold"]["cv3"]
        + config["CTfold"]["cv4"]
        + config["CTfold"]["cv5"]
    )

    allus = natsorted(
        config["USfold"]["cv1"]
        + config["USfold"]["cv2"]
        + config["USfold"]["cv3"]
        + config["USfold"]["cv4"]
        + config["USfold"]["cv5"]
    )

    ctlist = allct
    uslist = allus

    main(
        config,
        ctlist,
        uslist,
        CTmask_to_mesh=bool(config["CTmask_to_mesh"]),
        USmask_to_mesh=bool(config["USmask_to_mesh"]),
    )
