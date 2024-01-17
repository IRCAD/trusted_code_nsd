"""
In this file, we analyse the fused manual segmentations versus each annotator.
"""

from glob import glob
from os.path import join

import numpy as np
import pandas as pd
import yaml
from natsort import natsorted

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.metrics import Dice, Haus95Mask, HausMesh, MeanNNDistance
from trusted_datapaper_ds.utils import makedir, parse_args


def segeval(
    modality,
    results_folder,
    magt_files,
    maauto_files,
    megt_files,
    meauto_files,
):
    assert len(magt_files) == len(maauto_files) and len(megt_files) == len(
        meauto_files
    ), "There is an incompatibility about the number of files in the lists you give me."

    csv_dice_file = join(results_folder, modality + "_dice.csv")
    csv_othermetric_file = join(results_folder, modality + "_othermetric.csv")

    df_dice = pd.DataFrame()
    df_othermetric = pd.DataFrame()

    magt_files = natsorted(magt_files)
    maauto_files = natsorted(maauto_files)
    megt_files = natsorted(megt_files)
    meauto_files = natsorted(meauto_files)

    for i, magt_file in enumerate(magt_files):
        dice1 = np.nan
        haus1 = np.nan

        magt = dt.Mask(magt_file, annotatorID="gt")
        maauto = dt.Mask(maauto_files[i], annotatorID="auto")

        if modality == "us":
            assert magt.modality == "US", "The mask seems not to be for a US image"
        if modality == "ct":
            assert magt.modality == "CT", "The mask seems not to be for a CT image"

        ID = magt.individual_name

        print("Processing ", ID)

        assert (
            magt.individual_name == maauto.individual_name
        ), "The masks are from different individuals"

        # Evaluate Dice score and Hausdorff95 metrics between masks:
        try:
            dice = Dice(maauto_files[i], magt_file)
            dice1 = dice.evaluate_overlap()
        except ValueError:
            error_message = (
                "There is an error when computing Dice for individual " + ID + ". \n"
            )
            print(error_message)

        try:
            haus = Haus95Mask(maauto_files[i], magt_file)
            haus1 = haus.evaluate_overlap()
        except ValueError:
            error_message = (
                "There is an error when computing Haus95Mask for individual "
                + ID
                + ". \n"
            )
            print(error_message)

        # Results saving
        values = {
            "kidney_id": ID,
            "dice": dice1,
            "h95mask": haus1,
        }
        df_dice = df_dice.append(values, ignore_index=True)

    df_dice.to_csv(csv_dice_file, index=False)

    # Evaluate Dice score, Hausdorff95, nn_distance metrics between meshes:
    for i, megt_file in enumerate(megt_files):
        haus1 = np.nan
        nndst1 = np.nan

        megt = dt.Mesh(megt_file, annotatorID="gt")
        meauto = dt.Mesh(meauto_files[i], annotatorID="auto")

        if modality == "us":
            assert magt.modality == "US", "The mask seems not to be for a US image"
        if modality == "ct":
            assert magt.modality == "CT", "The mask seems not to be for a CT image"

        ID = megt.individual_name

        print("Processing ", ID)

        assert (
            megt.individual_name == meauto.individual_name
        ), "The meshes are from different individuals"

        # Evaluate Hausdorf metric between meshes:
        try:
            haus = HausMesh(95)
            haus1 = haus.evaluate_mesh(meauto.o3dmesh, megt.o3dmesh)
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
            "h95mesh": haus1,
            "nndst": nndst1,
        }
        df_othermetric = df_othermetric.append(values, ignore_index=True)

    df_othermetric.to_csv(csv_othermetric_file, index=False)

    return


def main(modality, results_folder, magt_files, maauto_files, megt_files, meauto_files):
    segeval(
        modality,
        results_folder,
        magt_files,
        maauto_files,
        megt_files,
        meauto_files,
    )
    return


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    modality = config["modality"]
    results_folder = config["segresults_folder"]

    magt_folder = config["gtmask_location"]
    megt_folder = config["gtmesh_location"]

    maauto_folder = join(
        config["mask_seglocation"], config["segmodel"], config["training_target"]
    )
    meauto_folder = join(
        config["mesh_seglocation"], config["segmodel"], config["training_target"]
    )

    magt_files = natsorted(glob(join(magt_folder, "*.nii.gz")))
    maauto_files = natsorted(glob(join(maauto_folder, "*.nii.gz")))

    megt_files = natsorted(glob(join(megt_folder, "*.obj")))
    meauto_files = natsorted(glob(join(meauto_folder, "*.obj")))

    results_folder = join(
        config["segresults_folder"], config["segmodel"], config["training_target"]
    )
    makedir(results_folder)

    main(modality, results_folder, magt_files, maauto_files, megt_files, meauto_files)
