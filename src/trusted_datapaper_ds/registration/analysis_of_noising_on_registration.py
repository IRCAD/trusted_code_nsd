from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from trusted_datapaper_ds.utils import makedir, parse_args

matplotlib.use("Agg")
plt.rcParams.update({"font.size": 25})


def result_csv_to_df(results_location: str, regmethod: str, transform: str, std: str):
    path = join(
        results_location, regmethod, transform, "std" + str(std) + ".0" + "results.csv"
    )
    df = pd.read_csv(path, sep=",", header=0, index_col=0)
    df["dice"] *= 100

    return df


def create_all_noisy_dataframes(config):
    results_location = config["regresults_folder"]

    global cases_list, list_regmethods, list_transforms, list_std

    cases_list = []
    list_regmethods = config["list_regmethods"]
    list_transforms = config["list_regtransforms"]
    list_std = config["list_std"]

    for regmethod in list_regmethods:
        for transform in list_transforms:
            for std in list_std:
                index = str(2 * list_std.index(std))
                globals()[regmethod + "_" + transform + index] = result_csv_to_df(
                    results_location=results_location,
                    regmethod=regmethod,
                    transform=transform,
                    std=std,
                )
                cases_list.append(regmethod + "_" + transform + index)

    return


def single_affine_rigid_box_plotting(
    figures_location: str, metric: str, metric_threshup: str
):
    global recap1, noise1_count_excluded, noise1_count_excluded_list
    recap1 = pd.DataFrame()
    noise1_count_excluded = []
    noise1_count_excluded_list = []

    for noise in list_std:
        optim_count_excluded = []
        for optim in list_regmethods:
            for transfo in ["rigid", "affine"]:
                df = globals()[optim + "_" + transfo + str(noise)]
                if metric == "tre":
                    build = pd.DataFrame(df.tre)
                    excluded = df.tre[df.tre > metric_threshup]

                if metric == "dice":
                    build = pd.DataFrame(df.dice)
                    excluded = df.dice[df.dice < metric_threshup]

                if metric == "nndst":
                    build = pd.DataFrame(df.nndst)
                    excluded = df.nndst[df.nndst > metric_threshup]

                if metric == "h95mesh":
                    build = pd.DataFrame(df.h95mesh)
                    excluded = df.h95mesh[df.h95mesh > metric_threshup]

                build["level"] = "std=" + str(noise) + "mm"
                build["optim"] = optim
                build["transfo"] = transfo
                recap1 = recap1.append(build)

                optim_count_excluded.append(excluded.shape[0])

        noise1_count_excluded.append(optim_count_excluded)
        noise1_count_excluded_list.append("".join(str(e) for e in optim_count_excluded))

    global data
    data = pd.melt(
        recap1,
        id_vars=["level", "optim", "transfo"],
        var_name="metric",
        value_name=metric,
    )
    data.rename(columns={"transfo": "Transform"}, inplace=True)

    data.optim = data.optim.replace("imfLNCC", "IF-LNCC")
    data.optim = data.optim.replace("imfLC2", "IF-LC2")

    ax1 = sns.catplot(
        data=data,
        x="optim",
        y=metric,
        hue="Transform",
        col="level",
        kind="box",
        aspect=1,
        palette=None,
        legend=True,
        legend_out=True,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": "8",
        },
        margin_titles=False,
    )

    ax1.set_titles("{col_name}")

    if metric == "dice":
        ax1.set(ylim=(metric_threshup, 100))
    else:
        ax1.set(ylim=(0, metric_threshup))

    # plt.legend(loc='best', fontsize=20)

    ax1.set_xticklabels(fontsize=20)

    if metric == "tre":
        ax1.set_axis_labels("", "TRE (mm)")
    if metric == "dice":
        ax1.set_axis_labels("", "Dice score (%)")
    if metric == "nndst":
        ax1.set_axis_labels("", "NN dist. (mm)")
    if metric == "h95mesh":
        ax1.set_axis_labels("", "HD95 (mm)")

    i = 0
    for ax in ax1.axes.flat:
        if metric == "dice":
            ax.set_yticks(np.linspace(metric_threshup, 100, 7))
        else:
            ax.set_yticks(np.linspace(0, metric_threshup, 7))

        ax.grid(True, axis="y")

        noise1_count_excluded_i = noise1_count_excluded[i]

        ax_bis = ax.twiny()
        ax_bis.set(xlim=(-0.6, len(noise1_count_excluded_i) - 0.4))
        sns.lineplot(range(len(noise1_count_excluded_i)), visible=False)
        ax_bis.set_xticks(range(len(noise1_count_excluded_i)), size=1)
        ax_bis.set_xticklabels([j for j in noise1_count_excluded_i])
        ax_bis.tick_params(axis="both", which="both", length=0)

        i += 1

    plt.savefig(
        join(figures_location, "affine_rigid_" + metric + "_boxplot_all.png"),
        bbox_inches="tight",
    )

    plt.close()

    return


def affine_rigid_box_plotting(
    figures_location: str,
    threshup_TRE: float,
    threshup_Dice: float,
    threshup_nn: float,
    threshup_h95: float,
):
    single_affine_rigid_box_plotting(
        figures_location=figures_location, metric="tre", metric_threshup=threshup_TRE
    )

    single_affine_rigid_box_plotting(
        figures_location=figures_location, metric="dice", metric_threshup=threshup_Dice
    )

    single_affine_rigid_box_plotting(
        figures_location=figures_location, metric="nndst", metric_threshup=threshup_nn
    )

    single_affine_rigid_box_plotting(
        figures_location=figures_location,
        metric="h95mesh",
        metric_threshup=threshup_h95,
    )
    return


def main(config):
    results_location = config["regresults_folder"]

    figures_location = join(results_location, "figures")
    makedir(figures_location)

    create_all_noisy_dataframes(config)

    """with troncature"""
    affine_rigid_box_plotting(
        figures_location=figures_location,
        threshup_TRE=30.0,
        threshup_Dice=10.0,
        threshup_nn=12.0,
        threshup_h95=45.0,
    )


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    main(config)
