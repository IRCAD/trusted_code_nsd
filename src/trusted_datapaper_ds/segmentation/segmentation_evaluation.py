"""
In this file, we analyse the fused manual segmentations versus each annotator.
"""

from glob import glob
from itertools import zip_longest
from os.path import join

import numpy as np
import pandas as pd
import yaml
from natsort import natsorted

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.metrics import Dice, HausMesh, MeanNNDistance
from trusted_datapaper_ds.utils import makedir, parse_args


def loader(maauto_files, magt_files, meauto_files, megt_files):
    for maauto_file, magt_file, meauto_file, megt_file in zip_longest(
        maauto_files, magt_files, meauto_files, megt_files
    ):
        maauto = dt.Mask(maauto_file, "auto", split=modality == "CT")
        magt = dt.Mask(magt_file, annotatorID="gt", split=modality == "CT")
        megt = dt.Mesh(megt_file, annotatorID="gt")
        meauto = dt.Mesh(meauto_file, annotatorID="auto")

        yield maauto, magt, megt, meauto


def segeval(maauto, magt, megt, meauto):
    dice1 = np.nan
    hmesh1 = np.nan
    nndst1 = np.nan

    assert maauto.individual_name == magt.individual_name
    assert meauto.individual_name == megt.individual_name

    ID = maauto.individual_name

    print("Processing: ", ID)

    # Evaluate Dice score and Hausdorff95 metrics between masks:
    dice = Dice(maauto.path, magt.path)
    try:
        dice1 = dice.evaluate_dice()
    except ValueError:
        error_message = (
            "There is an error when computing Dice for individual " + ID + ". \n"
        )
        print(error_message)

    # Evaluate Hausdorf metric between meshes:
    try:
        hausmesh = HausMesh(95)
        hmesh1 = hausmesh.evaluate_mesh(meauto.o3dmesh, megt.o3dmesh)
    except ValueError:
        error_message = (
            "There is an error when computing HausdorffMask for individual "
            + ID
            + ". \n"
        )
        print(error_message)

    # Evaluate mean surface-to-surface nearest neighbour distance:
    try:
        d_nn = MeanNNDistance()
        nndst1 = d_nn.evaluate_mesh(meauto.o3dmesh, megt.o3dmesh)
    except ValueError:
        error_message = (
            "There is an error when computing nn_dist for individual " + ID + ". \n"
        )
        print(error_message)

    # Results saving
    values = {
        "kidney_id": ID,
        "dice": dice1,
        "h95mesh": hmesh1,
        "nndst": nndst1,
    }

    return values


def main(csv_file, maauto_files, magt_files, meauto_files, megt_files):
    df = pd.DataFrame()

    for maauto, magt, megt, meauto in loader(
        maauto_files, magt_files, meauto_files, megt_files
    ):
        values = segeval(maauto, magt, megt, meauto)
        df = df.append(values, ignore_index=True)
        # print(df)

    df.to_csv(csv_file, index=False)

    return


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    modality = config["modality"]
    results_folder = config["segresults_folder"]

    list_segmodel = config["list_segmodels"]
    list_training_target = config["list_training_target"]

    for segmodel in list_segmodel:
        for training_target in list_training_target:
            print(segmodel)
            print(training_target)

            makedir(join(results_folder, segmodel, training_target))
            csv_file = join(results_folder, segmodel, training_target, "segresults.csv")

            if modality == "CT":
                magt_folder = config["gtsplitmask_location"]
                maauto_folder = join(
                    config["splitmask_seglocation"],
                    segmodel,
                    training_target,
                )
            if modality == "US":
                magt_folder = config["gtmask_location"]
                maauto_folder = join(
                    config["mask_seglocation"], segmodel, training_target
                )

            megt_folder = config["gtmesh_location"]
            meauto_folder = join(config["mesh_seglocation"], segmodel, training_target)

            magt_files = natsorted(glob(join(magt_folder, "*.nii.gz")))
            maauto_files = natsorted(glob(join(maauto_folder, "*.nii.gz")))
            meauto_files = natsorted(glob(join(meauto_folder, "*.obj")))
            megt_files = natsorted(glob(join(megt_folder, "*.obj")))

            main(csv_file, maauto_files, magt_files, meauto_files, megt_files)
