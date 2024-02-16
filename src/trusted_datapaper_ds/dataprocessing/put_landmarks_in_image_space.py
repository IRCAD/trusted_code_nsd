"""
    Copyright (C) 2022-2024 IRCAD France - All rights reserved. *
    This file is part of Disrumpere. *
    Disrumpere can not be copied, modified and/or distributed without
    the express permission of IRCAD France.
"""

from os.path import join

import numpy as np
import yaml
from natsort import natsorted

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.utils import build_many_me_ld_analist, makedir, parse_args


def main(
    config,
    ctlist,
    CTldks_over_images,
):
    ann = config["annotator_mov_ldks"]

    if CTldks_over_images:
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
            ld_files = ld1_files
        elif ann == "2":
            ld_files = ld2_files
        elif ann == "gt":
            ld_files = ldgt_files

        new_ld_dirname = join(config["myDATA"], config["CTld" + ann + "_inimg_fol"])
        makedir(new_ld_dirname)

        for ld_file in ld_files:
            ld = dt.Landmarks(ld_file, annotatorID=ann)
            ldarray = ld.nparray
            ld_extend = np.ones((ldarray.shape[0], 4))
            ld_extend[:, :3] = ldarray.copy()

            ID = ld.individual_name
            print("Processing: ", ID)

            tbackldk_path = join(
                config["data_location"],
                config["CT_tbackldk_transforms"],
                ID[:-1] + "tbackldk.txt",
            )

            tbackldk_inras = np.loadtxt(tbackldk_path)

            new_ld_extend = (tbackldk_inras @ ld_extend.T).T

            new_ld = new_ld_extend[:, :3]

            new_ld_path = join(new_ld_dirname, ld.basename)

            np.savetxt(new_ld_path, new_ld)

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
        CTldks_over_images=bool(config["CTldks_over_images"]),
    )
