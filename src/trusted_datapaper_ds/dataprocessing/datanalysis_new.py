"""
In this file, we analyse the fused manual segmentations versus each annotator.
"""

from os.path import join

import numpy as np
import pandas as pd
from natsort import natsorted

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.metrics import Dice, Haus95Mask, HausMesh, MeanNNDistance


def datanalysis(
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
        haus1 = np.nan
        haus2 = np.nan

        magt = dt.Mask(magt_file, annotatorID="gt")
        if modality == "us":
            assert magt.modality == "US", "The mask seems not to be for a US image"
        if modality == "ct":
            assert magt.modality == "CT", "The mask seems not to be for a CT image"

        ID = magt.individual_name

        print("Processing ", ID)

        # Evaluate Dice score and Hausdorff95 metrics between masks:
        try:
            dice = Dice(ma1_files[i], magt_file)
            dice1 = dice.evaluate_overlap()
            dice = Dice(ma2_files[i], magt_file)
            dice2 = dice.evaluate_overlap()
        except ValueError:
            error_message = (
                "There is an error when computing Dice for individual " + ID + ". \n"
            )
            print(error_message)

        try:
            haus = Haus95Mask(ma1_files[i], magt_file)
            haus1 = haus.evaluate_overlap()
            haus = Haus95Mask(ma2_files[i], magt_file)
            haus2 = haus.evaluate_overlap()
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
            "dice1": dice1,
            "dice2": dice2,
            "h95mask1": haus1,
            "h95mask2": haus2,
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
        if modality == "us":
            assert magt.modality == "US", "The mask seems not to be for a US image"
        if modality == "ct":
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
