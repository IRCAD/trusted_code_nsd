from os.path import join

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from statsmodels.stats.descriptivestats import sign_test

from trusted_datapaper_ds.utils import detect_outliers, parse_args


def reg_analysis(config, reg_evaluation_outputs_folder, std):
    resultfile_csv = join(
        reg_evaluation_outputs_folder, "std" + str(std) + "results.csv"
    )
    resultdf = pd.read_csv(resultfile_csv, index_col=0)

    cv_list = ["cv1", "cv2", "cv3", "cv4", "cv5"]

    tre_mean_list = list()
    dice_mean_list = list()
    dst_nn_mean_list = list()
    h95_mean_list = list()

    for cv_str in cv_list:
        cv = config[cv_str]

        cv_df = pd.DataFrame()

        for ID in cv:
            tre = np.round(resultdf.dst_TRE0.loc[ID], 3)
            dice = np.round((resultdf.monai_dice.loc[ID] * 100), 3)
            h95mesh = np.round(resultdf.monai_h95.loc[ID], 3)
            nndst = np.round(resultdf.dst_nn.loc[ID], 3)

            cv_values = {
                "kidney_id": ID,
                "tre": tre,
                "dice": dice,
                "h95mesh": h95mesh,
                "nndst": nndst,
            }
            cv_df = cv_df.append(cv_values, ignore_index=True)

        # print(cv_df)
        tre_mean = cv_df.tre.mean()
        tre_mean_list.append(tre_mean)
        dice_mean = cv_df.dice.mean()
        dice_mean_list.append(dice_mean)
        haus95_mask_mean = cv_df.h95mesh.mean()
        h95_mean_list.append(haus95_mask_mean)
        dst_nn_mean = cv_df.nndst.mean()
        dst_nn_mean_list.append(dst_nn_mean)

    tre_mean = np.mean(tre_mean_list)
    dice_mean = np.mean(dice_mean_list)
    h95_mean = np.mean(h95_mean_list)
    nn_mean = np.mean(dst_nn_mean_list)

    tre_std = np.std(tre_mean_list)
    dice_std = np.std(dice_mean_list)
    h95_std = np.std(h95_mean_list)
    nn_std = np.std(dst_nn_mean_list)

    return (
        tre_mean_list,
        dice_mean_list,
        dst_nn_mean_list,
        h95_mean_list,
        tre_mean,
        dice_mean,
        nn_mean,
        h95_mean,
        tre_std,
        dice_std,
        nn_std,
        h95_std,
    )


def main(config, std=0.0):
    """
    :param:
    :return:
    """

    regresults_folder = config["regresults_folder"]

    tre = pd.DataFrame()
    dice = pd.DataFrame()
    h95 = pd.DataFrame()
    nn = pd.DataFrame()

    """ Build the means for reference methods"""

    refmethod = config["refmethod"]
    reftransform = config["reftransform"]

    print("PROCESSING THE REFERENCES: ")
    print("ref: ", refmethod, " ", reftransform)

    reffolder = join(regresults_folder, refmethod, reftransform)

    (
        tre1_mean_list,
        dice1_mean_list,
        nn1_mean_list,
        h1_mean_list,
        tre1_mean,
        dice1_mean,
        nn1_mean,
        h1_mean,
        tre1_std,
        dice1_std,
        nn1_std,
        h1_std,
    ) = reg_analysis(config, reg_evaluation_outputs_folder=reffolder, std=std)

    ref_tre_mean_list = tre1_mean_list.copy()
    ref_dice_mean_list = dice1_mean_list.copy()
    ref_dst_nn_mean_list = nn1_mean_list.copy()
    ref_h95_mean_list = h1_mean_list.copy()

    """ Build the means for all the methods"""

    list_regmethods = config["list_regmethods"]
    list_regtransforms = config["list_regtransforms"]

    for regmethod in list_regmethods:
        for regtransform in list_regtransforms:
            print("PROCESSING: ", regmethod, " ", regtransform)

            reg_evaluation_outputs_folder = join(
                regresults_folder, regmethod, regtransform
            )

            (
                tre_mean_list,
                dice_mean_list,
                dst_nn_mean_list,
                h95_mean_list,
                tre_mean,
                dice_mean,
                nn_mean,
                h95_mean,
                tre_std,
                dice_std,
                nn_std,
                h95_std,
            ) = reg_analysis(
                config,
                reg_evaluation_outputs_folder=reg_evaluation_outputs_folder,
                std=std,
            )

            tre_sign_test_M = np.nan
            tre_sign_test_pvalue = np.nan
            tre_wilcoxon_pvalue = np.nan
            tre_wilcoxon_statistic = np.nan

            tre_stat, tre_t_pvalue = stats.ttest_ind(
                ref_tre_mean_list, tre_mean_list, equal_var=False
            )
            tre_shapiro_stat, tre_shapiro_pvalue = stats.shapiro(
                np.array(ref_tre_mean_list) - np.array(tre_mean_list)
            )
            tre_number_of_outliers = detect_outliers(
                np.array(ref_tre_mean_list) - np.array(tre_mean_list)
            )
            try:
                tre_sign_test_M, tre_sign_test_pvalue = sign_test(
                    np.array(ref_tre_mean_list) - np.array(tre_mean_list)
                )
            except ValueError:
                pass

            try:
                if tre_shapiro_pvalue < 0.05:
                    tre_wilcoxon = stats.wilcoxon(
                        x=ref_tre_mean_list,
                        y=tre_mean_list,
                        zero_method="wilcox",
                        mode="approx",
                    )
                else:
                    tre_wilcoxon = stats.wilcoxon(
                        x=ref_tre_mean_list,
                        y=tre_mean_list,
                        zero_method="wilcox",
                        correction=False,
                        mode="approx",
                    )
                tre_wilcoxon_pvalue = tre_wilcoxon.pvalue
                tre_wilcoxon_statistic = tre_wilcoxon.statistic
            except ValueError:
                pass

            dice_sign_test_M = np.nan
            dice_sign_test_pvalue = np.nan
            dice_wilcoxon_pvalue = np.nan
            dice_wilcoxon_statistic = np.nan

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

            tre_values = {
                "regmethod": regmethod,
                "regtransform": regtransform,
                "mean": np.round(tre_mean, 4),
                "std": np.round(tre_std, 4),
                "shapiro_pvalue": np.round(tre_shapiro_pvalue, 4),
                "numbers_of_outliers": tre_number_of_outliers,
                "t_pvalue": np.round(tre_t_pvalue, 4),
                "wilcoxon_pvalue": np.round(tre_wilcoxon_pvalue, 4),
                "wilcoxon_statistic": np.round(tre_wilcoxon_statistic, 4),
                "sign_test_pvalue": np.round(tre_sign_test_pvalue, 4),
                "sign_test_statistic": np.round(tre_sign_test_M, 4),
            }
            tre = tre.append(tre_values, ignore_index=True)

            dice_values = {
                "regmethod": regmethod,
                "regtransform": regtransform,
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
                "regmethod": regmethod,
                "regtransform": regtransform,
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
                "regmethod": regmethod,
                "regtransform": regtransform,
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

    tre_pth = join(regresults_folder, "tre_summaryresults.csv")
    tre.to_csv(tre_pth, index=False)

    dice_pth = join(regresults_folder, "dice_summaryresults.csv")
    dice.to_csv(dice_pth, index=False)

    h95_pth = join(regresults_folder, "h95_summaryresults.csv")
    h95.to_csv(h95_pth, index=False)

    nn_pth = join(regresults_folder, "nn_summaryresults.csv")
    nn.to_csv(nn_pth, index=False)

    return


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    main(config, std=0.0)
