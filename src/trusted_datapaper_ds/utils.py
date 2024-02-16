"""
    Copyright (C) 2022-2024 IRCAD France - All rights reserved. *
    This file is part of Disrumpere. *
    Disrumpere can not be copied, modified and/or distributed without
    the express permission of IRCAD France.
"""

import argparse
import os
from os.path import isfile, join

import numpy as np


def parse_args():
    PARSER = argparse.ArgumentParser(description="")
    PARSER.add_argument(
        "--config_path", type=str, required=True, help="path to the parameters yml file"
    )
    PARSER.add_argument(
        "--num_processors",
        type=int,
        default=2,
        help="number of processors used \
                       for preprocessing",
    )
    ARGS = PARSER.parse_args()
    return ARGS


def makedir(folder):
    try:
        os.makedirs(folder)
        print("Directory ", folder, " Created ")
    except FileExistsError:
        print("Directory ", folder, " already exists")
    return


def detect_outliers(data, k=1.5):
    # change K to 2 or 1.5. This can be what you need it to be.
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = np.median(data) - k * iqr
    upper_bound = np.median(data) + k * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    number_of_outliers = len(outliers)
    return number_of_outliers


def build_analist(
    data_config, modality, datatype, annotatorID, USlike_IDlist=None, CTlike_IDlist=None
):
    """
    This function build a list of data paths. It is useful to run some operations in series.
    INPUTS:
        data_config: the yaml object containing the elements in config_file.yml
        modality(respect the word case):
                'US' for Ultrasound
                'CT' for CT
        datatype:
                'img' for image
                'ma' for mask
                'me' for mesh
                'ld' for landmarks
        annotatorID:
                'gt' for ground truth
                '1' for annotator1
                '2' for annotator2
        USlike_IDlist: list of kidneys ID in the format: individual+kidney_side (ex: ['01R', '02L'])
        CTlike_IDlist: list of kidneys ID in the format: individual+kidney_side (ex: ['01', '02'])
    OUTPUT:
        data_path_list: the list of data paths
    """
    assert modality in ["US", "CT"], " modality must be in ['US', 'CT'] "
    assert datatype in [
        "img",
        "ma",
        "me",
        "ld",
    ], " datatype must be 'US' or 'CT' in ['img', 'ma', 'me', 'ld'] "
    assert annotatorID in [
        "gt",
        "1",
        "2",
    ], " annotatorID must be in ['gt', '1', '2'] "
    assert (
        int(USlike_IDlist is None) + int(CTlike_IDlist is None)
    ) == 1, "one and only one IDlist must be empty"

    config = data_config

    if annotatorID == "gt":
        ma_middle = ""
        me_middle = ""
    else:
        if modality == "CT":
            ma_middle = "_" + annotatorID
        if modality == "US":
            ma_middle = annotatorID

        me_middle = annotatorID

    if datatype == "img":
        str_var_IDlist = modality.upper() + "like_IDlist"
        var = vars()
        var_IDlist = var.__getitem__(str_var_IDlist)
        IDlist = var_IDlist
        data_path_list = [
            join(
                config["data_location"],
                config[modality + "imgfol"],
                individual + config[modality + "img_end"],
            )
            for individual in IDlist
            if isfile(
                join(
                    config["data_location"],
                    config[modality + "imgfol"],
                    individual + config[modality + "img_end"],
                )
            )
        ]

    if datatype == "ma":
        if USlike_IDlist is not None:
            if modality == "CT":
                print("You are building a list of split kidney masks in CT")
                start = "CTspma"
            if modality == "US":
                print("You are building a list of kidney masks in US")
                start = "USma"
            IDlist = USlike_IDlist

            data_path_list = [
                join(
                    config["data_location"],
                    config[start + annotatorID + "fol"],
                    individual + ma_middle + config[modality + "ma_end"],
                )
                for individual in IDlist
                if isfile(
                    join(
                        config["data_location"],
                        config[start + annotatorID + "fol"],
                        individual + ma_middle + config[modality + "ma_end"],
                    )
                )
            ]

        if CTlike_IDlist is not None:
            assert modality == "CT", "modality must be 'CT' "
            print("You are building a list of double kidney masks in CT")
            IDlist = CTlike_IDlist
            data_path_list = [
                join(
                    config["data_location"],
                    config["CTma" + annotatorID + "fol"],
                    individual + ma_middle + config["CTma_end"],
                )
                for individual in IDlist
                if isfile(
                    join(
                        config["data_location"],
                        config["CTma" + annotatorID + "fol"],
                        individual + ma_middle + config["CTma_end"],
                    )
                )
            ]

    if datatype == "me":
        assert (USlike_IDlist is not None) and (
            CTlike_IDlist is None
        ), " The IDlist must necessary be a  USlike_IDlist."
        IDlist = USlike_IDlist
        data_path_list = [
            join(
                config["data_location"],
                config[modality + "me" + annotatorID + "fol"],
                individual + me_middle + config[modality + "me_end"],
            )
            for individual in IDlist
            if isfile(
                join(
                    config["data_location"],
                    config[modality + "me" + annotatorID + "fol"],
                    individual + me_middle + config[modality + "me_end"],
                )
            )
        ]

    if datatype == "ld":
        assert (USlike_IDlist is not None) and (
            CTlike_IDlist is None
        ), " The IDlist must necessary be a  USlike_IDlist."
        IDlist = USlike_IDlist
        data_path_list = [
            join(
                config["data_location"],
                config[modality + "ld" + annotatorID + "fol"],
                individual + me_middle + config[modality + "ld_end"],
            )
            for individual in IDlist
            if isfile(
                join(
                    config["data_location"],
                    config[modality + "ld" + annotatorID + "fol"],
                    individual + me_middle + config[modality + "ld_end"],
                )
            )
        ]

    return data_path_list


def build_many_mask_analist(mod, config, USlike_IDlist, CTlike_IDlist):
    ma1_files = build_analist(
        data_config=config,
        modality=mod,
        datatype="ma",
        annotatorID="1",
        USlike_IDlist=USlike_IDlist,
        CTlike_IDlist=CTlike_IDlist,
    )
    ma2_files = build_analist(
        data_config=config,
        modality=mod,
        datatype="ma",
        annotatorID="2",
        USlike_IDlist=USlike_IDlist,
        CTlike_IDlist=CTlike_IDlist,
    )
    magt_files = build_analist(
        data_config=config,
        modality=mod,
        datatype="ma",
        annotatorID="gt",
        USlike_IDlist=USlike_IDlist,
        CTlike_IDlist=CTlike_IDlist,
    )

    return ma1_files, ma2_files, magt_files


def build_many_me_ld_analist(mod, config, USlike_IDlist, CTlike_IDlist=None):
    me1_files = build_analist(
        data_config=config,
        modality=mod,
        datatype="me",
        annotatorID="1",
        USlike_IDlist=USlike_IDlist,
        CTlike_IDlist=None,
    )
    me2_files = build_analist(
        data_config=config,
        modality=mod,
        datatype="me",
        annotatorID="2",
        USlike_IDlist=USlike_IDlist,
        CTlike_IDlist=None,
    )
    megt_files = build_analist(
        data_config=config,
        modality=mod,
        datatype="me",
        annotatorID="gt",
        USlike_IDlist=USlike_IDlist,
        CTlike_IDlist=None,
    )
    ld1_files = build_analist(
        data_config=config,
        modality=mod,
        datatype="ld",
        annotatorID="1",
        USlike_IDlist=USlike_IDlist,
        CTlike_IDlist=None,
    )
    ld2_files = build_analist(
        data_config=config,
        modality=mod,
        datatype="ld",
        annotatorID="2",
        USlike_IDlist=USlike_IDlist,
        CTlike_IDlist=None,
    )
    ldgt_files = build_analist(
        data_config=config,
        modality=mod,
        datatype="ld",
        annotatorID="gt",
        USlike_IDlist=USlike_IDlist,
        CTlike_IDlist=None,
    )

    return me1_files, me2_files, megt_files, ld1_files, ld2_files, ldgt_files
