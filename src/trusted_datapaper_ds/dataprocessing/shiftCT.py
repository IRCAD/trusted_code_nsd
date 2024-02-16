"""
    Copyright (C) 2022-2024 IRCAD France - All rights reserved. *
    This file is part of Disrumpere. *
    Disrumpere can not be copied, modified and/or distributed without
    the express permission of IRCAD France.
"""

from os.path import join

import yaml
from natsort import natsorted

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.utils import makedir, parse_args


def main(
    config,
    ctlist,
    shiftCTimg_origin,
):
    """
    This function executes CT image origin shifting operations based on the provided configuration and settings.

    Args:
        config (dict): A dictionary containing configuration parameters, including:
            - `myDATA`: The base directory for data storage.
            - `CTimgfol`: The subdirectory containing CT images.
            - `CT0imgfol`: The subdirectory to save shifted CT images (if enabled).
            - `CT_tbackmesh_transforms`: The subdirectory to save transformation matrices for shifting meshes to
            the CT coordinate system (if enabled).
            - `CT_tbackldk_transforms`: The subdirectory to save transformation matrices for shifting landmarks to
            the CT coordinate system (if enabled).
            - `data_location`: The main data location.
            - `CTimg_end`: The file extension for CT images.
        ctlist (list[str]): A list of individual identifiers for the CT images to be processed.
        shift_ct_origin (bool): A flag indicating whether to perform CT image origin shifting.

    Raises:
        ValueError: If `shift_ct_origin` is True and any essential configuration values are missing.

    Returns:
        None
    """

    if shiftCTimg_origin:
        shifted_dirname = join(config["myDATA"], config["CT0imgfol"])
        makedir(shifted_dirname)
        shiftback_mesh_transforms_dirname = join(
            config["myDATA"], config["CT_tbackmesh_transforms"]
        )
        makedir(shiftback_mesh_transforms_dirname)
        shiftback_ldks_transforms_dirname = join(
            config["myDATA"], config["CT_tbackldk_transforms:"]
        )
        makedir(shiftback_ldks_transforms_dirname)
        for ind in ctlist:
            individual = ind
            imgpath = join(
                config["data_location"],
                config["CTimgfol"],
                individual + config["CTimg_end"],
            )
            ctimg = dt.Image(imgpath)
            ctimg.shift_origin(
                shifted_dirname=shifted_dirname,
                shiftback_mesh_transforms_dirname=shiftback_mesh_transforms_dirname,
                shiftback_ldks_transforms_dirname=shiftback_ldks_transforms_dirname,
            )
    return


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    allct = natsorted(
        config["CTfoldmask"]["cv1"]
        + config["CTfoldmask"]["cv2"]
        + config["CTfoldmask"]["cv3"]
        + config["CTfoldmask"]["cv4"]
        + config["CTfoldmask"]["cv5"]
    )

    ctlist = allct

    main(
        config,
        ctlist,
        shiftCTimg_origin=config["shiftCTimg_origin"],
    )
