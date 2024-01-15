import argparse
import os
from os.path import isfile, join


def parse_args():
    PARSER = argparse.ArgumentParser(description="")
    PARSER.add_argument(
        "--config_path", type=str, required=True, help="path to the parameters yml file"
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


def build_list(
    data_config, modality, datatype, annotatorID, USlike_IDlist=None, CTlike_IDlist=None
):
    """
    This function build a list of data paths. It is useful to run some operations in series.
    INPUTS:
        data_config: the yaml object containing the elements in config_file.yml
        modality(respect the word case):
                'us' for Ultrasound
                'ct' for CT
        datatype:
                'img' for image
                'ma' for mask
                'me' for mesh
                'ld' for landmarks
        annotatorID:
                'gt' for ground truth
                'annotator1' for annotator1
                'annotator2' for annotator2
        USlike_IDlist: list of kidneys ID in the format: individual+kidney_side (ex: ['01R', '02L'])
        CTlike_IDlist: list of kidneys ID in the format: individual+kidney_side (ex: ['01', '02'])
    OUTPUT:
        data_path_list: the list of data paths
    """
    assert modality in ["us", "ct"], " modality must be in ['us', 'ct'] "
    assert datatype in [
        "img",
        "ma",
        "me",
        "ld",
    ], " datatype must be 'us' or 'ct' in ['img', 'ma', 'me', 'ld'] "
    assert annotatorID in [
        "gt",
        "annotator1",
        "annotator2",
    ], " annotatorID must be in ['gt', 'annotator1', 'annotator2'] "
    assert (
        int(USlike_IDlist is None) + int(CTlike_IDlist is None)
    ) == 1, "one and only one IDlist must be empty"

    config = data_config

    if annotatorID == "gt":
        ma_middle = ""
        me_middle = ""
    else:
        if modality == "ct":
            ma_middle = "_" + config[annotatorID]
        if modality == "us":
            ma_middle = config[annotatorID]

        me_middle = config[annotatorID]

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
            if modality == "ct":
                print("You are building a list of split kidney masks in CT")
                start = "ctspma"
            if modality == "us":
                print("You are building a list of kidney masks in US")
                start = "usma"
            IDlist = USlike_IDlist

            data_path_list = [
                join(
                    config["data_location"],
                    config[start + config[annotatorID] + "fol"],
                    individual + ma_middle + config[modality + "ma_end"],
                )
                for individual in IDlist
                if isfile(
                    join(
                        config["data_location"],
                        config[start + config[annotatorID] + "fol"],
                        individual + ma_middle + config[modality + "ma_end"],
                    )
                )
            ]

        if CTlike_IDlist is not None:
            assert modality == "ct", "modality must be 'ct' "
            print("You are building a list of double kidney masks in CT")
            IDlist = CTlike_IDlist
            data_path_list = [
                join(
                    config["data_location"],
                    config["ctma" + config[annotatorID] + "fol"],
                    individual + ma_middle + config["ctma_end"],
                )
                for individual in IDlist
                if isfile(
                    join(
                        config["data_location"],
                        config["ctma" + config[annotatorID] + "fol"],
                        individual + ma_middle + config["ctma_end"],
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
                config[modality + "me" + config[annotatorID] + "fol"],
                individual + me_middle + config[modality + "me_end"],
            )
            for individual in IDlist
            if isfile(
                join(
                    config["data_location"],
                    config[modality + "me" + config[annotatorID] + "fol"],
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
                config[modality + "ld" + config[annotatorID] + "fol"],
                individual + me_middle + config[modality + "ld_end"],
            )
            for individual in IDlist
            if isfile(
                join(
                    config["data_location"],
                    config[modality + "ld" + config[annotatorID] + "fol"],
                    individual + me_middle + config[modality + "ld_end"],
                )
            )
        ]

    return data_path_list


def build_many_mask_list(mod, config, USlike_IDlist, CTlike_IDlist):
    ma1_files = build_list(
        data_config=config,
        modality=mod,
        datatype="ma",
        annotatorID="annotator1",
        USlike_IDlist=USlike_IDlist,
        CTlike_IDlist=CTlike_IDlist,
    )
    ma2_files = build_list(
        data_config=config,
        modality=mod,
        datatype="ma",
        annotatorID="annotator2",
        USlike_IDlist=USlike_IDlist,
        CTlike_IDlist=CTlike_IDlist,
    )
    magt_files = build_list(
        data_config=config,
        modality=mod,
        datatype="ma",
        annotatorID="gt",
        USlike_IDlist=USlike_IDlist,
        CTlike_IDlist=CTlike_IDlist,
    )

    return ma1_files, ma2_files, magt_files


def build_many_me_ld_list(mod, config, USlike_IDlist, CTlike_IDlist=None):
    me1_files = build_list(
        data_config=config,
        modality=mod,
        datatype="me",
        annotatorID="annotator1",
        USlike_IDlist=USlike_IDlist,
        CTlike_IDlist=None,
    )
    me2_files = build_list(
        data_config=config,
        modality=mod,
        datatype="me",
        annotatorID="annotator2",
        USlike_IDlist=USlike_IDlist,
        CTlike_IDlist=None,
    )
    megt_files = build_list(
        data_config=config,
        modality=mod,
        datatype="me",
        annotatorID="gt",
        USlike_IDlist=USlike_IDlist,
        CTlike_IDlist=None,
    )
    ld1_files = build_list(
        data_config=config,
        modality=mod,
        datatype="ld",
        annotatorID="annotator1",
        USlike_IDlist=USlike_IDlist,
        CTlike_IDlist=None,
    )
    ld2_files = build_list(
        data_config=config,
        modality=mod,
        datatype="ld",
        annotatorID="annotator2",
        USlike_IDlist=USlike_IDlist,
        CTlike_IDlist=None,
    )
    ldgt_files = build_list(
        data_config=config,
        modality=mod,
        datatype="ld",
        annotatorID="gt",
        USlike_IDlist=USlike_IDlist,
        CTlike_IDlist=None,
    )

    return me1_files, me2_files, megt_files, ld1_files, ld2_files, ldgt_files
