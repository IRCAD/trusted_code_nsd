from os.path import join

import numpy as np

# from kidney_processing.metrics import *
import pandas as pd
import yaml
from scipy import stats
from statsmodels.stats.descriptivestats import sign_test

from trusted_datapaper_ds.utils import parse_args


def detect_outliers(data, k=1.5):
    # change K to 2 or 1.5. This can be what you need it to be.
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = np.median(data) - k * iqr
    upper_bound = np.median(data) + k * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    number_of_outliers = len(outliers)
    return number_of_outliers


def seg_analysis(config, seg_evaluation_outputs_folder: str):
    modality = config["modality"]

    resultfile_csv = join(seg_evaluation_outputs_folder, modality + "_segresults.csv")
    resultdf = pd.read_csv(resultfile_csv, index_col=0)

    summarydf = pd.DataFrame()
    cv_list = ["cv1", "cv2", "cv3", "cv4", "cv5"]

    dice_mean_list = list()
    dst_nn_mean_list = list()
    h95_mean_list = list()

    for cv_str in cv_list:
        cv = config[cv_str]
        print("CV: ", cv)
        cv_df = pd.DataFrame()

        for ID in cv:
            dice = np.round((resultdf.dice.loc[ID] * 100), 3)
            h95mask = np.round(resultdf.h95mask.loc[ID], 3)
            nndst = np.round(resultdf.nndst.loc[ID], 3)

            cv_values = {
                "kidney_id": ID,
                "dice": dice,
                "h95mask": h95mask,
                "nndst": nndst,
            }
            cv_df = cv_df.append(cv_values, ignore_index=True)

        # print(cv_df)

        dice_mean = cv_df.dice.mean()
        dice_mean_list.append(dice_mean)
        haus95_mask_mean = cv_df.h95mask.mean()
        h95_mean_list.append(haus95_mask_mean)
        dst_nn_mean = cv_df.nndst.mean()
        dst_nn_mean_list.append(dst_nn_mean)

        values = {
            "cv": cv_str,
            "mean_dice": dice_mean,
            "mean_h95mask": haus95_mask_mean,
            "dst_mean_nn": dst_nn_mean,
        }
        summarydf = summarydf.append(values, ignore_index=True)

    dice_mean = summarydf.mean_dice.mean()
    h95_mean = summarydf.mean_h95mask.mean()
    nn_mean = summarydf.dst_mean_nn.mean()
    dice_std = summarydf.mean_dice.std()
    h95_std = summarydf.mean_h95mask.std()
    nn_std = summarydf.dst_mean_nn.std()
    dice_median = summarydf.mean_dice.median()
    h95_median = summarydf.mean_h95mask.median()
    nn_median = summarydf.dst_mean_nn.median()

    return (
        dice_mean_list,
        dst_nn_mean_list,
        h95_mean_list,
        dice_mean,
        nn_mean,
        h95_mean,
        dice_std,
        nn_std,
        h95_std,
        dice_median,
        nn_median,
        h95_median,
    )


def main(config):
    """
    :param:
    :return:
    """

    segresults_folder = config["segresults_folder"]
    modality = config["modality"]

    dice = pd.DataFrame()
    h95 = pd.DataFrame()
    nn = pd.DataFrame()

    """ Build the means for reference methods"""
    ref1model = config["ref1model"]
    ref1target = config["ref1target"]

    ref2model = config["ref2model"]
    ref2target = config["ref2target"]

    ref1folder = join(segresults_folder, ref1model, ref1target)
    ref2folder = join(segresults_folder, ref2model, ref2target)

    (
        dice1_mean_list,
        nn1_mean_list,
        h1_mean_list,
        dice1_mean,
        nn1_mean,
        h1_mean,
        dice1_std,
        nn1_std,
        h1_std,
        dice1_median,
        nn1_median,
        h1_median,
    ) = seg_analysis(config, seg_evaluation_outputs_folder=ref1folder)

    (
        dice2_mean_list,
        nn2_mean_list,
        h2_mean_list,
        dice2_mean,
        nn2_mean,
        h2_mean,
        dice2_std,
        nn2_std,
        h2_std,
        dice2_median,
        nn2_median,
        h2_median,
    ) = seg_analysis(config, seg_evaluation_outputs_folder=ref2folder)

    if modality == "us":
        ref_dice_mean_list = dice1_mean_list.copy()
        ref_dst_nn_mean_list = nn1_mean_list.copy()
        ref_h95_mean_list = h2_mean_list.copy()

    if modality == "ct":
        ref_dice_mean_list = dice2_mean_list.copy()
        ref_dst_nn_mean_list = nn1_mean_list.copy()
        ref_h95_mean_list = h1_mean_list.copy()

    """ Build the means for all the methods"""
    # for segmodel in ["unet", "vnet", "nnunet", "cotr"]:
    #     for target_kind in ["double_targets_training", "single_target_training"]:
    for segmodel in ["nnunet"]:
        for target_kind in ["single_target_training"]:
            seg_evaluation_outputs_folder = join(
                segresults_folder, segmodel, target_kind
            )

            (
                dice_mean_list,
                dst_nn_mean_list,
                h95_mean_list,
                dice_mean,
                nn_mean,
                h95_mean,
                dice_std,
                nn_std,
                h95_std,
                dice_median,
                nn_median,
                h95_median,
            ) = seg_analysis(
                config, seg_evaluation_outputs_folder=seg_evaluation_outputs_folder
            )

            dice_sign_test_M = np.nan
            dice_sign_test_pvalue = np.nan
            dice_wilcoxon_pvalue = np.nan
            dice_wilcoxon_statistic = np.nan

            print(
                segmodel,
                target_kind,
                "ref_dice_mean_list",
                ref_dice_mean_list,
                "dice_mean_list",
                dice_mean_list,
            )

            dice_stat, dice_t_pvalue = stats.ttest_ind(
                ref_dice_mean_list, dice_mean_list, equal_var=False
            )
            dice_shapiro_stat, dice_shapiro_pvalue = stats.shapiro(
                np.array(ref_dice_mean_list) - np.array(dice_mean_list)
            )
            dice_number_of_outliers = detect_outliers(
                np.array(ref_dice_mean_list) - np.array(dice_mean_list)
            )
            try:
                dice_sign_test_M, dice_sign_test_pvalue = sign_test(
                    np.array(ref_dice_mean_list) - np.array(dice_mean_list)
                )
            except ValueError:
                pass

            try:
                if dice_shapiro_pvalue < 0.05:
                    dice_wilcoxon = stats.wilcoxon(
                        x=ref_dice_mean_list,
                        y=dice_mean_list,
                        zero_method="wilcox",
                        mode="approx",
                    )
                else:
                    dice_wilcoxon = stats.wilcoxon(
                        x=ref_dice_mean_list,
                        y=dice_mean_list,
                        zero_method="wilcox",
                        correction=False,
                        mode="approx",
                    )
                dice_wilcoxon_pvalue = dice_wilcoxon.pvalue
                dice_wilcoxon_statistic = dice_wilcoxon.statistic
            except ValueError:
                pass

            h95_sign_test_M = np.nan
            h95_sign_test_pvalue = np.nan
            h95_wilcoxon_pvalue = np.nan
            h95_wilcoxon_statistic = np.nan

            h95_stat, h95_t_pvalue = stats.ttest_ind(
                ref_h95_mean_list, h95_mean_list, equal_var=False
            )
            h95_shapiro_stat, h95_shapiro_pvalue = stats.shapiro(
                np.array(ref_h95_mean_list) - np.array(h95_mean_list)
            )
            h95_number_of_outliers = detect_outliers(
                np.array(ref_h95_mean_list) - np.array(h95_mean_list)
            )
            try:
                h95_sign_test_M, h95_sign_test_pvalue = sign_test(
                    np.array(ref_h95_mean_list) - np.array(h95_mean_list)
                )
            except ValueError:
                pass

            try:
                if h95_shapiro_pvalue < 0.05:
                    h95_wilcoxon = stats.wilcoxon(
                        x=ref_h95_mean_list,
                        y=h95_mean_list,
                        zero_method="wilcox",
                        mode="approx",
                    )
                else:
                    h95_wilcoxon = stats.wilcoxon(
                        x=ref_h95_mean_list,
                        y=h95_mean_list,
                        zero_method="wilcox",
                        correction=False,
                        mode="approx",
                    )
                h95_wilcoxon_pvalue = h95_wilcoxon.pvalue
                h95_wilcoxon_statistic = h95_wilcoxon.statistic
            except ValueError:
                pass

            nn_sign_test_M = np.nan
            nn_sign_test_pvalue = np.nan
            nn_wilcoxon_pvalue = np.nan
            nn_wilcoxon_statistic = np.nan
            nn_stat, nn_t_pvalue = stats.ttest_ind(
                ref_dst_nn_mean_list, dst_nn_mean_list, equal_var=False
            )
            nn_shapiro_stat, nn_shapiro_pvalue = stats.shapiro(
                np.array(ref_dst_nn_mean_list) - np.array(dst_nn_mean_list)
            )
            nn_number_of_outliers = detect_outliers(
                np.array(ref_dst_nn_mean_list) - np.array(dst_nn_mean_list)
            )
            try:
                nn_sign_test_M, nn_sign_test_pvalue = sign_test(
                    np.array(ref_dst_nn_mean_list) - np.array(dst_nn_mean_list)
                )
            except ValueError:
                pass

            try:
                if nn_shapiro_pvalue < 0.05:
                    nn_wilcoxon = stats.wilcoxon(
                        x=ref_dst_nn_mean_list,
                        y=dst_nn_mean_list,
                        zero_method="wilcox",
                        mode="approx",
                    )
                else:
                    nn_wilcoxon = stats.wilcoxon(
                        x=ref_dst_nn_mean_list,
                        y=dst_nn_mean_list,
                        zero_method="wilcox",
                        correction=False,
                        mode="approx",
                    )
                nn_wilcoxon_pvalue = nn_wilcoxon.pvalue
                nn_wilcoxon_statistic = nn_wilcoxon.statistic
            except ValueError:
                pass

            dice_values = {
                "modality": modality,
                "segmodel": segmodel,
                "target_kind": target_kind,
                "median": np.round(dice_median, 4),
                "mean": np.round(dice_mean, 4),
                "std": np.round(dice_std, 4),
                "shapiro_pvalue": np.round(dice_shapiro_pvalue, 4),
                "numbers_of_outliers": dice_number_of_outliers,
                "t_pvalue": np.round(dice_t_pvalue, 4),
                "wilcoxon_pvalue": np.round(dice_wilcoxon_pvalue, 4),
                "wilcoxon_statistic": np.round(dice_wilcoxon_statistic, 4),
                "sign_test_pvalue": np.round(dice_sign_test_pvalue, 4),
                "sign_test_statistic": np.round(dice_sign_test_M, 4),
            }
            dice = dice.append(dice_values, ignore_index=True)

            h95_values = {
                "modality": modality,
                "segmodel": segmodel,
                "target_kind": target_kind,
                "median": np.round(h95_median, 4),
                "mean": np.round(h95_mean, 4),
                "std": np.round(h95_std, 4),
                "shapiro_pvalue": np.round(h95_shapiro_pvalue, 4),
                "numbers_of_outliers": h95_number_of_outliers,
                "t_pvalue": np.round(h95_t_pvalue, 4),
                "wilcoxon_pvalue": np.round(h95_wilcoxon_pvalue, 4),
                "wilcoxon_statistic": np.round(h95_wilcoxon_statistic, 4),
                "sign_test_pvalue": np.round(h95_sign_test_pvalue, 4),
                "sign_test_statistic": np.round(h95_sign_test_M, 4),
            }
            h95 = h95.append(h95_values, ignore_index=True)

            nn_values = {
                "modality": modality,
                "segmodel": segmodel,
                "target_kind": target_kind,
                "median": np.round(nn_median, 4),
                "mean": np.round(nn_mean, 4),
                "std": np.round(nn_std, 4),
                "shapiro_pvalue": np.round(nn_shapiro_pvalue, 4),
                "numbers_of_outliers": nn_number_of_outliers,
                "t_pvalue": np.round(nn_t_pvalue, 4),
                "wilcoxon_pvalue": np.round(nn_wilcoxon_pvalue, 4),
                "wilcoxon_statistic": np.round(nn_wilcoxon_statistic, 4),
                "sign_test_pvalue": np.round(nn_sign_test_pvalue, 4),
                "sign_test_statistic": np.round(nn_sign_test_M, 4),
            }
            nn = nn.append(nn_values, ignore_index=True)

    dice_pth = join(segresults_folder, "dice_summaryresults.csv")
    dice.to_csv(dice_pth, index=False)

    h95_pth = join(segresults_folder, "h95_summaryresults.csv")
    h95.to_csv(h95_pth, index=False)

    nn_pth = join(segresults_folder, "nn_summaryresults.csv")
    nn.to_csv(nn_pth, index=False)

    return


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    main(config)
