# """
# In this file, we analyse the fused manual segmentations versus each annotator.
# """

# import pickle

# import numpy as np
# import pandas as pd
# from natsort import natsorted

# from trusted_datapaper_ds.dataprocessing import data as dt
# from trusted_datapaper_ds.metrics import Dice, HausdorffMask, MeanNNDistance


# def usdatanalysis(
#     usma1_files,
#     usma2_files,
#     usmagt_files,
#     usme1_files,
#     usme2_files,
#     usmegt_files,
#     usld1_files,
#     usld2_files,
#     usldgt_files,
# ):
#     assert (
#         len(usmagt_files) == len(usma1_files)
#         and len(usmagt_files) == len(usma2_files)
#         and len(usmegt_files) == len(usme1_files)
#         and len(usmegt_files) == len(usme2_files)
#         and len(usldgt_files) == len(usld1_files)
#         and len(usldgt_files) == len(usld2_files)
#     ), "There is an incompatibility about the number of files in the lists you give me."

#     usma1_files = natsorted(usma1_files)
#     usma2_files = natsorted(usma2_files)
#     usmagt_files = natsorted(usmagt_files)
#     usme1_files = natsorted(usme1_files)
#     usme2_files = natsorted(usme2_files)
#     usmegt_files = natsorted(usmegt_files)
#     usld1_files = natsorted(usld1_files)
#     usld2_files = natsorted(usld2_files)
#     usldgt_files = natsorted(usldgt_files)

#     for i, file in enumerate(usmagt_files):
#         dice_us1 = np.nan
#         dice_us2 = np.nan
#         haus_us1 = np.nan
#         haus_us2 = np.nan
#         dst_us1 = np.nan
#         dst_us2 = np.nan

#         usmagt = dt.Mask(file)
#         assert usmagt.modality == "US", "The mask seems not to be for a US image"
#         ID = usmagt.individual_name
#         usma1 = dt.Mask(usma1_files[i], annotatorID="1")
#         usma2 = dt.Mask(usma2_files[i], annotatorID="2")
#         usme1 = dt.Landmarks(usme1_files[i])
#         usme2 = dt.Landmarks(usme2_files[i])
#         usmegt = dt.Landmarks(usmegt_files[i])
#         dt.Landmarks(usld1_files[i])
#         dt.Landmarks(usld2_files[i])
#         dt.Landmarks(usldgt_files[i])

#         # Evaluate Dice score metric over US masks:
#         try:
#             dice_us = Dice(usmagt.nparray, usma1.nparray)
#             dice_us1 = dice_us.evaluate_overlap()
#             print("dice_us1: ", dice_us1)

#             dice_us = Dice(usmagt.nparray, usma2.nparray)
#             dice_us2 = dice_us.evaluate_overlap()
#             print("dice_us2: ", dice_us2)
#         except:
#             error_message = (
#                 "There is an error when computing Dice for US, individual "
#                 + ID
#                 + ". \n"
#             )
#             print(error_message)

#         # Evaluate Hausdorf metric over US masks:
#         try:
#             haus_us = HausdorffMask(usmagt.nparray, usma1.nparray)
#             haus_us1 = haus_us.evaluate_overlap()
#             print("haus_us1: ", haus_us1)

#             haus_us = HausdorffMask(usmagt.nparray, usma2.nparray)
#             haus_us2 = haus_us.evaluate_overlap()
#             print("haus_us2: ", haus_us2)
#         except:
#             error_message = (
#                 "There is an error when computing HausdorffMask for US, individual "
#                 + ID
#                 + ". \n"
#             )
#             print(error_message)

#         # Evaluate US mean surface-to-surface nearest neighbour distance:
#         try:
#             d_nn = MeanNNDistance()
#             dst_us1 = d_nn.evaluate_mesh(usmegt, usme1)
#             print("dst_us1 =", dst_us1)
#             dst_us2 = d_nn.evaluate_mesh(usmegt, usme2)
#             print("dst_us2 =", dst_us2)
#         except:
#             error_message = (
#                 "There is an error when computing  US nn_dist for individual "
#                 + ID
#                 + ". \n"
#             )
#             print(error_message)

#     return


# # def ctdatanalysis(
# #     ctma1_files, ctma2_files, ctmagt_files, ctld1_files, ctld2_files, ctldgt_files
# # ):
# #     assert (
# #         len(ctmagt_files) == len(ctma1_files)
# #         and len(ctmagt_files) == len(ctma2_files)
# #         and len(ctldgt_files) == len(ctld1_files)
# #         and len(ctldgt_files) == len(ctld2_files)
# #     ), "There is an incompatibility about the number of files in the lists you give me."

# #     ctma1_files = natsorted(ctma1_files)
# #     ctma2_files = natsorted(ctma2_files)
# #     ctmagt_files = natsorted(ctmagt_files)
# #     ctld1_files = natsorted(ctld1_files)
# #     ctld2_files = natsorted(ctld2_files)
# #     ctldgt_files = natsorted(ctldgt_files)

# #     for i, file in enumerate(ctmagt_files):
# #         np.nan
# #         np.nan
# #         np.nan
# #         np.nan
# #         np.nan
# #         np.nan

# #         ctmagt = dt.Mask(file)
# #         assert ctmagt.modality == "CT", "The mask seems not to be for a CT image"
# #         ctmagt.individual_name
# #         dt.Mask(ctma1_files[i])
# #         dt.Mask(ctma2_files[i])
# #         dt.Mask(ctld1_files[i])
# #         dt.Mask(ctld2_files[i])
# #         dt.Mask(ctldgt_files[i])


# def write_csv_files(files, results_pth):
#     """
#     Writes registration results as a CSV file (which is then cted to compute registration statistics)
#     :return: None
#     """

#     df = pd.DataFrame()
#     for f in files:
#         ID = get_id_from_filename(f)
#         plkf = results_pth + "/" + ID + ".pkl"
#         if os.path.exists(plkf):
#             values = pickle.load(open(plkf, "rb"))
#             df = df.append(values, ignore_index=True)
#         else:
#             print(plkf + " does not exist")
#     df.to_csv(results_pth + "/" + "results.csv", index=False)
