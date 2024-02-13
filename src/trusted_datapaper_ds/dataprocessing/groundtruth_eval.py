"""
In this file, we analyse the fused manual segmentations versus each annotator.
"""
from os.path import join

import numpy as np
import pandas as pd
import yaml
from natsort import natsorted

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.metrics import Dice, HausMesh, MeanNNDistance
from trusted_datapaper_ds.utils import (
    build_many_mask_analist,
    build_many_me_ld_analist,
    makedir,
    parse_args,
)


def gteval(
    modality,
    analysis_folder,
    ma1_files,
    ma2_files,
    magt_files,
    me1_files,
    me2_files,
    megt_files,
    ld1_files,
    ld2_files,
    ldgt_files,
):
    assert (
        len(magt_files) == len(ma1_files)
        and len(magt_files) == len(ma2_files)
        and len(megt_files) == len(me1_files)
        and len(megt_files) == len(me2_files)
        and len(ldgt_files) == len(ld1_files)
        and len(ldgt_files) == len(ld2_files)
    ), "There is an incompatibility about the number of files in the lists you give me."

    csv_dice_file = join(analysis_folder, modality + "_dice.csv")
    csv_othermetric_file = join(analysis_folder, modality + "_othermetric.csv")

    df_dice = pd.DataFrame()
    df_othermetric = pd.DataFrame()

    ma1_files = natsorted(ma1_files)
    ma2_files = natsorted(ma2_files)
    magt_files = natsorted(magt_files)
    me1_files = natsorted(me1_files)
    me2_files = natsorted(me2_files)
    megt_files = natsorted(megt_files)
    ld1_files = natsorted(ld1_files)
    ld2_files = natsorted(ld2_files)
    ldgt_files = natsorted(ldgt_files)

    for i, magt_file in enumerate(magt_files):
        dice1 = np.nan
        dice2 = np.nan

        magt = dt.Mask(magt_file, annotatorID="gt")
        ma1 = dt.Mask(ma1_files[i], annotatorID="1")
        ma2 = dt.Mask(ma2_files[i], annotatorID="2")

        if modality == "US":
            assert magt.modality == "US", "The mask seems not to be for a US image"
        if modality == "CT":
            assert magt.modality == "CT", "The mask seems not to be for a CT image"

        assert magt.individual_name == ma1.individual_name
        assert magt.individual_name == ma2.individual_name

        ID = magt.individual_name

        print("Processing ", ID)

        # Evaluate Dice score and Hausdorff95 metrics between masks:
        try:
            dice = Dice(ma1.nparray, magt.nparray)
            dice1 = dice.evaluate_dice()
            dice = Dice(ma2.nparray, magt.nparray)
            dice2 = dice.evaluate_dice()
        except ValueError:
            error_message = (
                "There is an error when computing Dice for individual " + ID + ". \n"
            )
            print(error_message)

        # Results saving
        values = {
            "kidney_id": ID,
            "dice1": dice1,
            "dice2": dice2,
        }
        df_dice = df_dice.append(values, ignore_index=True)

    df_dice.to_csv(csv_dice_file, index=False)

    # Evaluate Dice score, Hausdorff95, nn_distance metrics between meshes, and , landmarks_distances:
    for i, megt_file in enumerate(megt_files):
        haus1 = np.nan
        haus2 = np.nan
        nndst1 = np.nan
        nndst2 = np.nan
        lm_dist1 = [np.nan for i in range(7)]
        lm_dist2 = [np.nan for i in range(7)]

        megt = dt.Mesh(megt_file, annotatorID="gt")
        if modality == "US":
            assert magt.modality == "US", "The mask seems not to be for a US image"
        if modality == "CT":
            assert magt.modality == "CT", "The mask seems not to be for a CT image"

        ID = megt.individual_name

        print("Processing ", ID)
        me1 = dt.Mesh(me1_files[i], annotatorID="1")
        me2 = dt.Mesh(me2_files[i], annotatorID="2")
        ld1 = dt.Landmarks(ld1_files[i], annotatorID="1")
        ld2 = dt.Landmarks(ld2_files[i], annotatorID="2")
        ldgt = dt.Landmarks(ldgt_files[i], annotatorID="gt")

        # Evaluate Hausdorf metric between meshes:
        try:
            haus = HausMesh(95)
            haus1 = haus.evaluate_mesh(me1.o3dmesh, megt.o3dmesh)
            haus2 = haus.evaluate_mesh(me2.o3dmesh, megt.o3dmesh)
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
            nndst1 = d_nn.evaluate_mesh(me1.o3dmesh, megt.o3dmesh)
            nndst2 = d_nn.evaluate_mesh(me2.o3dmesh, megt.o3dmesh)
        except ValueError:
            error_message = (
                "There is an error when computing nn_dist for individual " + ID + ". \n"
            )
            print(error_message)

        # Evaluate landmark inter-annotator agreement:
        try:
            for t in range(7):
                lm_dist1[t] = np.linalg.norm(ld1.nparray[t, :] - ldgt.nparray[t, :])
                lm_dist2[t] = np.linalg.norm(ld2.nparray[t, :] - ldgt.nparray[t, :])
        except ValueError:
            error_message = (
                "There is an error when computing landmark inter-annotator agreement for individual "
                + ID
                + ". \n"
            )
            print(error_message)

        # Results saving
        values = {
            "kidney_id": ID,
            "h95mesh1": haus1,
            "h95mesh2": haus2,
            "nndst1": nndst1,
            "nndst2": nndst2,
            "lm1_dist": lm_dist1[0],
            "lm2_dist": lm_dist1[1],
            "lm3_dist": lm_dist1[2],
            "lm4_dist": lm_dist1[3],
            "lm5_dist": lm_dist1[4],
            "lm6_dist": lm_dist1[5],
            "lm7_dist": lm_dist1[6],
        }
        df_othermetric = df_othermetric.append(values, ignore_index=True)

    df_othermetric.to_csv(csv_othermetric_file, index=False)

    return


def main(config):
    usdata_analysis = bool(config["usdata_analysis"])
    ctdata_analysis = bool(config["ctdata_analysis"])

    allct = natsorted(
        config["CTfold"]["cv1"]
        + config["CTfold"]["cv2"]
        + config["CTfold"]["cv3"]
        + config["CTfold"]["cv4"]
        + config["CTfold"]["cv5"]
    )
    allct_with_side = natsorted([j + "L" for j in allct] + [j + "R" for j in allct])
    allus = natsorted(
        config["USfold"]["cv1"]
        + config["USfold"]["cv2"]
        + config["USfold"]["cv3"]
        + config["USfold"]["cv4"]
        + config["USfold"]["cv5"]
    )

    ctlist_with_side = allct_with_side

    ctlist = allct
    uslist = allus

    if usdata_analysis:
        modality = "US"
        usdatanalysis_folder = join(config["myDATA"], config["US_analysis_folder"])
        makedir(usdatanalysis_folder)

        USlike_IDlist = uslist
        CTlike_IDlist = None

        usma1_files, usma2_files, usmagt_files = build_many_mask_analist(
            modality, config, USlike_IDlist, CTlike_IDlist
        )

        (
            usme1_files,
            usme2_files,
            usmegt_files,
            usld1_files,
            usld2_files,
            usldgt_files,
        ) = build_many_me_ld_analist(modality, config, USlike_IDlist, CTlike_IDlist)

        gteval(
            modality,
            usdatanalysis_folder,
            usma1_files,
            usma2_files,
            usmagt_files,
            usme1_files,
            usme2_files,
            usmegt_files,
            usld1_files,
            usld2_files,
            usldgt_files,
        )

    if ctdata_analysis:
        modality = "CT"
        ctdatanalysis_folder = join(config["myDATA"], config["CT_analysis_folder"])
        makedir(ctdatanalysis_folder)

        # For CT Masks
        USlike_IDlist = None
        CTlike_IDlist = ctlist
        ctma1_files, ctma2_files, ctmagt_files = build_many_mask_analist(
            modality, config, USlike_IDlist, CTlike_IDlist
        )

        # For CT Meshes and Landmarks which are name like US ones
        USlike_IDlist = ctlist_with_side
        CTlike_IDlist = None
        (
            ctme1_files,
            ctme2_files,
            ctmegt_files,
            ctld1_files,
            ctld2_files,
            ctldgt_files,
        ) = build_many_me_ld_analist(modality, config, USlike_IDlist, CTlike_IDlist)

        gteval(
            modality,
            ctdatanalysis_folder,
            ctma1_files,
            ctma2_files,
            ctmagt_files,
            ctme1_files,
            ctme2_files,
            ctmegt_files,
            ctld1_files,
            ctld2_files,
            ctldgt_files,
        )


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    main(config)
