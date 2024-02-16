"""
    Copyright (C) 2022-2024 IRCAD France - All rights reserved. *
    This file is part of Disrumpere. *
    Disrumpere can not be copied, modified and/or distributed without
    the express permission of IRCAD France.
"""

from os.path import dirname, join

import numpy as np
import pandas as pd
import yaml

from trusted_datapaper_ds.utils import parse_args


def stat_summary(config, modality, analysis_folder):
    """
    Calculates and summarizes statistics from multiple analysis results.

    This function reads Dice and othermetric CSV files for a specific modality
    and calculates various summary statistics (mean and standard
    deviation) across different cross-validation folds, and returns two DataFrames
    containing these statistics.

    Args:
        config (dict): Configuration dictionary.
        modality (str): The modality of the data analyze (e.g., "CT", "US").
        analysis_folder (str): The path to the CSV files.

    Returns:
        tuple: A tuple containing two pandas.DataFrame objects:
            - The first DataFrame contains Dice and nearest neighbor distance statistics.
            - The second DataFrame contains landmark distance statistics.
    """

    csv_dice_file = join(analysis_folder, modality + "_dice.csv")
    dicedf = pd.read_csv(csv_dice_file, index_col=0)
    csv_othermetric_file = join(analysis_folder, modality + "_othermetric.csv")
    othermetricdf = pd.read_csv(csv_othermetric_file, index_col=0)

    dfa = pd.DataFrame()
    dfb = pd.DataFrame()

    cv_list = ["cv1", "cv2", "cv3", "cv4", "cv5"]

    dice1_mean_list = list()
    dice2_mean_list = list()
    dst_nn1_mean_list = list()
    dst_nn2_mean_list = list()
    h951_mean_list = list()
    h952_mean_list = list()
    lm1dist_mean_list = list()
    lm2dist_mean_list = list()
    lm3dist_mean_list = list()
    lm4dist_mean_list = list()
    lm5dist_mean_list = list()
    lm6dist_mean_list = list()
    lm7dist_mean_list = list()

    for cv_str in cv_list:
        if modality == "CT":
            cvmask0 = config["CTfoldmask"][cv_str]
            cvmask = [int(x) for x in cvmask0]
            cvmesh = config["CTfoldmesh"][cv_str]

        if modality == "US":
            cvmask = config["USfold"][cv_str]
            cvmesh = config["USfold"][cv_str]

        cv_dfa = pd.DataFrame()
        cv_dfb = pd.DataFrame()

        for ID in cvmask:
            dice1 = np.round((dicedf.dice1.loc[ID] * 100), 3)
            dice2 = np.round((dicedf.dice2.loc[ID] * 100), 3)

            cva_values = {
                "kidney_id": ID,
                "dice1": dice1,
                "dice2": dice2,
            }
            cv_dfa = cv_dfa.append(cva_values, ignore_index=True)

        for ID in cvmesh:
            h95mesh1 = np.round(othermetricdf.h95mesh1.loc[ID], 3)
            h95mesh2 = np.round(othermetricdf.h95mesh2.loc[ID], 3)

            nndst1 = np.round(othermetricdf.nndst1.loc[ID], 3)
            nndst2 = np.round(othermetricdf.nndst2.loc[ID], 3)

            lm1_dist = np.round(othermetricdf.lm1_dist.loc[ID], 3)
            lm2_dist = np.round(othermetricdf.lm2_dist.loc[ID], 3)
            lm3_dist = np.round(othermetricdf.lm3_dist.loc[ID], 3)
            lm4_dist = np.round(othermetricdf.lm4_dist.loc[ID], 3)
            lm5_dist = np.round(othermetricdf.lm5_dist.loc[ID], 3)
            lm6_dist = np.round(othermetricdf.lm6_dist.loc[ID], 3)
            lm7_dist = np.round(othermetricdf.lm7_dist.loc[ID], 3)

            cvb_values = {
                "kidney_id": ID,
                "h95mesh1": h95mesh1,
                "h95mesh2": h95mesh2,
                "nndst1": nndst1,
                "nndst2": nndst2,
                "lm1dist": lm1_dist,
                "lm2dist": lm2_dist,
                "lm3dist": lm3_dist,
                "lm4dist": lm4_dist,
                "lm5dist": lm5_dist,
                "lm6dist": lm6_dist,
                "lm7dist": lm7_dist,
            }
            cv_dfb = cv_dfb.append(cvb_values, ignore_index=True)

        dice1_mean = cv_dfa.dice1.mean()
        dice1_mean_list.append(dice1_mean)
        dice2_mean = cv_dfa.dice2.mean()
        dice2_mean_list.append(dice2_mean)

        haus951_mean = cv_dfb.h95mesh1.mean()
        h951_mean_list.append(haus951_mean)
        haus952_mean = cv_dfb.h95mesh2.mean()
        h952_mean_list.append(haus952_mean)
        dst_nn1_mean = cv_dfb.nndst1.mean()
        dst_nn1_mean_list.append(dst_nn1_mean)
        dst_nn2_mean = cv_dfb.nndst2.mean()
        dst_nn2_mean_list.append(dst_nn2_mean)
        lm1dist_mean = cv_dfb.lm1dist.mean()
        lm1dist_mean_list.append(lm1dist_mean)
        lm2dist_mean = cv_dfb.lm2dist.mean()
        lm2dist_mean_list.append(lm2dist_mean)
        lm3dist_mean = cv_dfb.lm3dist.mean()
        lm3dist_mean_list.append(lm3dist_mean)
        lm4dist_mean = cv_dfb.lm4dist.mean()
        lm4dist_mean_list.append(lm4dist_mean)
        lm5dist_mean = cv_dfb.lm5dist.mean()
        lm5dist_mean_list.append(lm5dist_mean)
        lm6dist_mean = cv_dfb.lm6dist.mean()
        lm6dist_mean_list.append(lm6dist_mean)
        lm7dist_mean = cv_dfb.lm7dist.mean()
        lm7dist_mean_list.append(lm7dist_mean)

    dice1_mean = np.mean(dice1_mean_list)
    dice2_mean = np.mean(dice2_mean_list)
    h951_mean = np.mean(h951_mean_list)
    h952_mean = np.mean(h952_mean_list)
    nn1_mean = np.mean(dst_nn1_mean_list)
    nn2_mean = np.mean(dst_nn2_mean_list)
    lm1dist_mean = np.mean(lm1dist_mean_list)
    lm2dist_mean = np.mean(lm2dist_mean_list)
    lm3dist_mean = np.mean(lm3dist_mean_list)
    lm4dist_mean = np.mean(lm4dist_mean_list)
    lm5dist_mean = np.mean(lm5dist_mean_list)
    lm6dist_mean = np.mean(lm6dist_mean_list)
    lm7dist_mean = np.mean(lm7dist_mean_list)

    dice1_std = np.std(dice1_mean_list)
    dice2_std = np.std(dice2_mean_list)
    h951_std = np.std(h951_mean_list)
    h952_std = np.std(h952_mean_list)
    nn1_std = np.std(dst_nn1_mean_list)
    nn2_std = np.std(dst_nn2_mean_list)
    lm1dist_std = np.std(lm1dist_mean_list)
    lm2dist_std = np.std(lm2dist_mean_list)
    lm3dist_std = np.std(lm3dist_mean_list)
    lm4dist_std = np.std(lm4dist_mean_list)
    lm5dist_std = np.std(lm5dist_mean_list)
    lm6dist_std = np.std(lm6dist_mean_list)
    lm7dist_std = np.std(lm7dist_mean_list)

    valuesa = {
        "dice1_mean": dice1_mean,
        "dice2_mean": dice2_mean,
        "nn1_mean": nn1_mean,
        "nn2_mean": nn2_mean,
        "h951_mean": h951_mean,
        "h952_mean": h952_mean,
        "dice1_std": dice1_std,
        "dice2_std": dice2_std,
        "nn1_std": nn1_std,
        "nn2_std": nn2_std,
        "h951_std": h951_std,
        "h952_std": h952_std,
    }

    valuesb = {
        "lm1dist_mean": lm1dist_mean,
        "lm2dist_mean": lm2dist_mean,
        "lm3dist_mean": lm3dist_mean,
        "lm4dist_mean": lm4dist_mean,
        "lm5dist_mean": lm5dist_mean,
        "lm6dist_mean": lm6dist_mean,
        "lm7dist_mean": lm7dist_mean,
        "lm1dist_std": lm1dist_std,
        "lm2dist_std": lm2dist_std,
        "lm3dist_std": lm3dist_std,
        "lm4dist_std": lm4dist_std,
        "lm5dist_std": lm5dist_std,
        "lm6dist_std": lm6dist_std,
        "lm7dist_std": lm7dist_std,
    }

    dfa = dfa.append(valuesa, ignore_index=True)
    dfb = dfb.append(valuesb, ignore_index=True)

    return dfa, dfb


def main(config):
    usdata_analysis = bool(config["usdata_analysis"])
    ctdata_analysis = bool(config["ctdata_analysis"])

    if usdata_analysis:
        modality = "US"
        print(modality + " DATA ANALYSIS")
        usdatanalysis_folder = join(config["myDATA"], config["US_analysis_folder"])
        df_overlap, df_landmarks = stat_summary(config, modality, usdatanalysis_folder)
        overlapsummary_file = join(
            usdatanalysis_folder, modality + "_overlapsummary.csv"
        )
        landmarksummary_file = join(
            usdatanalysis_folder, modality + "_landmarksummary.csv"
        )
        df_overlap.to_csv(overlapsummary_file)
        df_landmarks.to_csv(landmarksummary_file)
        print("stats summary save as :", dirname(overlapsummary_file))

    if ctdata_analysis:
        modality = "CT"
        print(modality + " DATA ANALYSIS")
        ctdatanalysis_folder = join(config["myDATA"], config["CT_analysis_folder"])
        df_overlap, df_landmarks = stat_summary(config, modality, ctdatanalysis_folder)
        overlapsummary_file = join(
            ctdatanalysis_folder, modality + "_overlapsummary.csv"
        )
        landmarksummary_file = join(
            ctdatanalysis_folder, modality + "_landmarksummary.csv"
        )
        df_overlap.to_csv(overlapsummary_file)
        df_landmarks.to_csv(landmarksummary_file)
        print("stats summary save as :", dirname(overlapsummary_file))

    return


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    main(config)
