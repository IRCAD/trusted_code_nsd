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
    config,
    maauto_files,
    meauto_files,
):
    modality = config["modality"]
    results_folder = config["segresults_folder"]
    segmodel = config["segmodel"]
    training_target = config["training_target"]

    assert len(maauto_files) == len(
        meauto_files
    ), "There is an incompatibility about the number of files in the lists you give me."

    csv_file = join(results_folder, segmodel, training_target, "segresults.csv")

    df = pd.DataFrame()

    maauto_files = natsorted(maauto_files)
    meauto_files = natsorted(meauto_files)

    if modality == "CT":
        magt_folder = config["gtsplitmask_location"]
    if modality == "US":
        magt_folder = config["gtmask_location"]
    meauto_folder = join(
        config["mesh_seglocation"], config["segmodel"], config["training_target"]
    )

    for maauto_file in maauto_files:
        dice1 = np.nan
        hmask1 = np.nan
        hmesh1 = np.nan
        nndst1 = np.nan

        maauto = dt.Mask(maauto_file, annotatorID="auto", split=modality == "CT")

        if modality == "US":
            assert maauto.modality == "US", "The mask seems not to be for a US image"
        if modality == "CT":
            assert maauto.modality == "CT", "The mask seems not to be for a CT image"

        ID = maauto.individual_name

        print("Processing: ", ID)

        magt_file = join(magt_folder, ID + config[modality + "ma_end"])

        megt_file = join(config["gtmesh_location"], ID + config[modality + "me_end"])
        megt = dt.Mesh(megt_file, annotatorID="gt")

        meauto_file = join(meauto_folder, ID + config[modality + "me_end"])
        meauto = dt.Mesh(meauto_file, annotatorID="auto")

        if modality == "US":
            assert megt.modality == "US", "The mesh seems not to be for a US image"
        if modality == "CT":
            assert megt.modality == "CT", "The mesh seems not to be for a CT image"

        assert (
            megt.individual_name == meauto.individual_name
        ), "The meshes are from different individuals"

        # Evaluate Dice score and Hausdorff95 metrics between masks:
        try:
            dice = Dice(maauto_file, magt_file)
            dice1 = dice.evaluate_overlap()
        except ValueError:
            error_message = (
                "There is an error when computing Dice for individual " + ID + ". \n"
            )
            print(error_message)

        try:
            hausmask = Haus95Mask(maauto_file, magt_file)
            hmask1 = hausmask.evaluate_overlap()
        except ValueError:
            error_message = (
                "There is an error when computing Haus95Mask for individual "
                + ID
                + ". \n"
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
            "h95mask": hmask1,
            "h95mesh": hmesh1,
            "nndst": nndst1,
        }
        df = df.append(values, ignore_index=True)

    df.to_csv(csv_file, index=False)

    return


def main(config, maauto_files, meauto_files):
    segeval(
        config,
        maauto_files,
        meauto_files,
    )
    return


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    modality = config["modality"]
    results_folder = config["segresults_folder"]

    if modality == "CT":
        magt_folder = config["gtsplitmask_location"]
        maauto_folder = join(
            config["splitmask_seglocation"],
            config["segmodel"],
            config["training_target"],
        )
    if modality == "US":
        magt_folder = config["gtmask_location"]
        maauto_folder = join(
            config["mask_seglocation"], config["segmodel"], config["training_target"]
        )

    meauto_folder = join(
        config["mesh_seglocation"], config["segmodel"], config["training_target"]
    )

    maauto_files = natsorted(glob(join(maauto_folder, "*.nii.gz")))
    meauto_files = natsorted(glob(join(meauto_folder, "*.obj")))

    results_folder = join(
        config["segresults_folder"], config["segmodel"], config["training_target"]
    )
    makedir(results_folder)

    main(config, maauto_files, meauto_files)
