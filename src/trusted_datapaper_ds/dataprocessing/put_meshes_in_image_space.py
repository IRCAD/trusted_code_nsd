"""
    Copyright (C) 2022-2024 IRCAD France - All rights reserved. *
    This file is part of Disrumpere. *
    Disrumpere can not be copied, modified and/or distributed without
    the express permission of IRCAD France.
"""

import copy
from os.path import join

import numpy as np
import open3d as o3d
import yaml
from natsort import natsorted

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.utils import build_many_me_ld_analist, makedir, parse_args


def main(
    config,
    ctlist,
    uslist,
    CTmesh_over_images,
    USmesh_over_images,
):
    ann = config["annotator_mov_mesh"]

    put_transform_into_ras = np.diag([-1, -1, 1, 1])
    put_transform_into_lps = np.diag([-1, -1, 1, 1])

    if CTmesh_over_images:
        USlike_IDlist = ctlist
        CTlike_IDlist = None

        (
            me1_files,
            me2_files,
            megt_files,
            ld1_files,
            ld2_files,
            ldgt_files,
        ) = build_many_me_ld_analist("CT", config, USlike_IDlist, CTlike_IDlist)

        if ann == "1":
            me_files = me1_files
        elif ann == "2":
            me_files = me2_files
        elif ann == "gt":
            me_files = megt_files

        new_mesh_dirname = join(config["myDATA"], config["CTme" + ann + "_inimg_fol"])
        makedir(new_mesh_dirname)

        for me_file in me_files:
            mesh = dt.Mesh(me_file, annotatorID=ann)
            o3dmesh = mesh.o3dmesh

            ID = mesh.individual_name
            print("Processing: ", ID)

            tbackmesh_path = join(
                config["data_location"],
                config["CT_tbackmesh_transforms"],
                ID[:-1] + "tbackmesh.txt",
            )

            t_meshlps_to_imgras = np.loadtxt(tbackmesh_path)

            t_mesh_to_img_inras = (
                put_transform_into_ras @ t_meshlps_to_imgras @ put_transform_into_lps
            )

            tbackmesh = t_mesh_to_img_inras.copy()

            new_o3d_meshCT = copy.deepcopy(o3dmesh).transform(tbackmesh)

            new_mesh_path = join(new_mesh_dirname, mesh.basename)

            o3d.io.write_triangle_mesh(new_mesh_path, new_o3d_meshCT)

    if USmesh_over_images:
        USlike_IDlist = ctlist
        CTlike_IDlist = None

        (
            me1_files,
            me2_files,
            megt_files,
            ld1_files,
            ld2_files,
            ldgt_files,
        ) = build_many_me_ld_analist("US", config, USlike_IDlist, USlike_IDlist)

        if ann == "1":
            me_files = me1_files
        elif ann == "2":
            me_files = me2_files
        elif ann == "gt":
            me_files = megt_files

        new_mesh_dirname = join(config["myDATA"], config["USme" + ann + "_inimg_fol"])
        makedir(new_mesh_dirname)

        for me_file in me_files:
            mesh = dt.Mesh(me_file, annotatorID=ann)
            o3dmesh = mesh.o3dmesh

            ID = mesh.individual_name
            print("Processing: ", ID)

            Identity4 = np.eye(4)
            tbackmesh = put_transform_into_ras @ Identity4

            new_o3d_meshUS = copy.deepcopy(o3dmesh).transform(tbackmesh)

            new_mesh_path = join(new_mesh_dirname, mesh.basename)

            o3d.io.write_triangle_mesh(new_mesh_path, new_o3d_meshUS)

    return


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    allct = natsorted(
        config["CTfoldmesh"]["cv1"]
        + config["CTfoldmesh"]["cv2"]
        + config["CTfoldmesh"]["cv3"]
        + config["CTfoldmesh"]["cv4"]
        + config["CTfoldmesh"]["cv5"]
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
        CTmesh_over_images=bool(config["CTmesh_over_images"]),
        USmesh_over_images=bool(config["USmesh_over_images"]),
    )
