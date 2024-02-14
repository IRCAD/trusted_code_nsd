from os.path import join

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from statsmodels.stats.descriptivestats import sign_test

from trusted_datapaper_ds.utils import detect_outliers, parse_args


def seg_analysis(config, seg_evaluation_outputs_folder: str):
    modality = config["modality"]

    resultfile_csv = join(seg_evaluation_outputs_folder, "segresults.csv")
    resultdf = pd.read_csv(resultfile_csv, index_col=0)

    if modality == "US":
        cv_list = ["uscv1", "uscv2", "uscv3", "uscv4", "uscv5"]
    if modality == "CT":
        cv_list = ["ctcv1", "ctcv2", "ctcv3", "ctcv4", "ctcv5"]

    dice_mean_list = list()
    dst_nn_mean_list = list()
    h95_mean_list = list()

    for cv_str in cv_list:
        cv = config[cv_str]

        cv_df = pd.DataFrame()

        for ID in cv:
            dice = np.round((resultdf.dice.loc[ID] * 100), 3)
            h95mesh = np.round(resultdf.h95mesh.loc[ID], 3)
            nndst = np.round(resultdf.nndst.loc[ID], 3)

            cv_values = {
                "kidney_id": ID,
                "dice": dice,
                "h95mesh": h95mesh,
                "nndst": nndst,
            }
            cv_df = cv_df.append(cv_values, ignore_index=True)

        # print(cv_df)
        dice_mean = cv_df.dice.mean()
        dice_mean_list.append(dice_mean)
        haus95_mask_mean = cv_df.h95mesh.mean()
        h95_mean_list.append(haus95_mask_mean)
        dst_nn_mean = cv_df.nndst.mean()
        dst_nn_mean_list.append(dst_nn_mean)

    dice_mean = np.mean(dice_mean_list)
    h95_mean = np.mean(h95_mean_list)
    nn_mean = np.mean(dst_nn_mean_list)

    dice_std = np.std(dice_mean_list)
    h95_std = np.std(h95_mean_list)
    nn_std = np.std(dst_nn_mean_list)

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
    )


def main(config):
    """
    Evaluate various segmentation methods and generate summary results.

    This function reads a configuration dictionary containing settings and iterates
    through a list of segmentation models and training targets. For each combination,
    it performs the following steps:

    1. **Process reference methods:** Calculates mean Dice, Hausdorff 95, and nearest neighbor
       distance metrics for the specified reference model and target.
    2. **Calculate metrics for each method:** Evaluates the segmentation results for the current
       model and target, computing the same metrics as for the reference.
    3. **Perform statistical significance tests:** Conducts Shapiro-Wilk normality tests, paired t-tests,
       and Wilcoxon tests to compare the metrics of the current method with the reference's.
    4. **Save summary results to CSV files:** Generates separate CSV files for Dice, Hausdorff 95,
       and nearest neighbor distance summary results, including mean, standard deviation,
       p-values from various tests, and outlier information.

    Args:
        config (dict): Configuration dictionary containing experiment settings. Must include
                       the following keys:
                           - `segresults_folder`: Path to the folder containing segmentation results.
                           - `modality`: Imaging modality (e.g., "CT", "US").
                           - `refmodel`: Reference segmentation model name.
                           - `reftarget`: Reference segmentation target name.
                           - `list_segmodels`: List of segmentation models to evaluate.
                           - `list_training_target`: List of training targets for each model.
                           - `seg_evaluation_outputs_folder`: Path to the folder containing
                                                          segmentation evaluation outputs.

    Returns:
        None
    """

    segresults_folder = join(config["segresults_folder"], config["modality"])
    modality = config["modality"]

    dice = pd.DataFrame()
    h95 = pd.DataFrame()
    nn = pd.DataFrame()

    """ Build the means for reference methods"""

    refmodel = config["refmodel"]
    reftarget = config["reftarget"]

    print("PROCESSING THE REFERENCES: ")
    print("ref: ", refmodel, " ", reftarget)

    reffolder = join(segresults_folder, refmodel, reftarget)

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
    ) = seg_analysis(config, seg_evaluation_outputs_folder=reffolder)

    ref_dice_mean_list = (
        dice1_mean_list.copy()
    )  # nnunet_double is the ref for dice in US
    ref_dst_nn_mean_list = nn1_mean_list.copy()  # nnunet_double is the ref for nn in US
    ref_h95_mean_list = h1_mean_list.copy()  # cotr_single is the ref for h95 in US

    """ Build the means for all the methods"""

    list_segmodel = config["list_segmodels"]
    list_training_target = config["list_training_target"]

    for segmodel in list_segmodel:
        for target_kind in list_training_target:
            print("PROCESSING: ", segmodel, " ", target_kind)

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
            ) = seg_analysis(
                config, seg_evaluation_outputs_folder=seg_evaluation_outputs_folder
            )

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

            dice_values = {
                "modality": modality,
                "segmodel": segmodel,
                "target_kind": target_kind,
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
    print("dice summary results saved as: ", dice_pth)

    h95_pth = join(segresults_folder, "h95_summaryresults.csv")
    h95.to_csv(h95_pth, index=False)
    print("h95 summary results saved as: ", h95_pth)

    nn_pth = join(segresults_folder, "nn_summaryresults.csv")
    nn.to_csv(nn_pth, index=False)
    print("nn summary results saved as: ", nn_pth)

    return


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    main(config)
