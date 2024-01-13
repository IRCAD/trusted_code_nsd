import math

import numpy as np
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete, Compose, EnsureChannelFirst, LoadImage

from trusted_datapaper_ds import geometry_utils as gu

# from monai.metrics import DiceMetric
# from monai.transforms import AddChannel, AsDiscrete, Compose, ToTensor
# process = Compose([AddChannel(), ToTensor()])
# binarizing = AsDiscrete(threshold_values=True)
# dice_metric = DiceMetric(include_background=True, reduction="mean")
# haus_mask_metric = HausdorffDistanceMetric(
#     include_background=False,
#     distance_metric="euclidean",
#     percentile=100,
#     directed=False,
#     reduction="mean",
#     get_not_nans=False,
# )

process = Compose([LoadImage(), EnsureChannelFirst(), AsDiscrete(threshold=0.5)])

dice_metric = DiceMetric(include_background=True, reduction="mean")
haus_mask_metric = HausdorffDistanceMetric(
    include_background=True,
    distance_metric="euclidean",
    percentile=95,
    directed=False,
    reduction="mean",
    get_not_nans=False,
)


class Dice:
    def __init__(self, pred_file, gt_file):
        self.pred_file = pred_file
        self.gt_file = gt_file
        return

    def evaluate_overlap(self):
        pred = process(self.pred_file)
        gt = process(self.gt_file)

        dice_metric(y_pred=pred, y=gt)
        dice = dice_metric.aggregate().item()
        dice_metric.reset()
        return dice


# class HausdorffMask:
#     def __init__(self, pred_file, gt_file):
#         self.pred_file = pred_file
#         self.gt_file = gt_file
#         return

#     def evaluate_overlap(self):
#         pred = process(self.pred_file)
#         gt = process(self.gt_file)

#         haus_mask_metric(y_pred=pred, y=gt, spacing=)
#         haus_mask = haus_mask_metric.aggregate().item()
#         haus_mask_metric.reset()
#         return haus_mask


class MeanNNDistance:
    def evaluate_mesh(self, msh1, msh2):
        d1, indx1 = gu.nearest_neighbor(
            np.asarray(msh1.vertices), np.asarray(msh2.vertices)
        )
        d2, indx2 = gu.nearest_neighbor(
            np.asarray(msh2.vertices), np.asarray(msh1.vertices)
        )
        d_total = (np.average(d1) + np.average(d2)) / 2.0

        return d_total

    def evaluate_pcd(self, pcd1, pcd2):
        d1, indx1 = gu.nearest_neighbor(
            np.asarray(pcd1.points), np.asarray(pcd2.points)
        )
        d2, indx2 = gu.nearest_neighbor(
            np.asarray(pcd2.points), np.asarray(pcd1.points)
        )
        d_total = (np.average(d1) + np.average(d2)) / 2.0

        return d_total


class HaussDistance:
    def __init__(self, percent):
        self.percent = percent

    def evaluate_mesh(self, msh1, msh2):
        d1, indx = gu.nearest_neighbor(
            np.asarray(msh1.vertices), np.asarray(msh2.vertices)
        )
        d2, indx = gu.nearest_neighbor(
            np.asarray(msh2.vertices), np.asarray(msh1.vertices)
        )
        d1 = np.sort(d1)
        d2 = np.sort(d2)
        inx1 = int(math.floor(float(self.percent) / 100.0 * float(d1.shape[0])) - 1.0)
        d1val = d1[inx1]
        inx2 = int(math.floor(float(self.percent) / 100.0 * float(d2.shape[0])) - 1.0)
        d2val = d2[inx2]

        d_total = max(np.average(d1val), np.average(d2val))

        return d_total

    def evaluate_pcd(self, pcd1, pcd2):
        d1, indx = gu.nearest_neighbor(np.asarray(pcd1.points), np.asarray(pcd2.points))
        d2, indx = gu.nearest_neighbor(np.asarray(pcd2.points), np.asarray(pcd1.points))
        d1 = np.sort(d1)
        d2 = np.sort(d2)
        inx1 = int(math.floor(float(self.percent) / 100.0 * float(d1.shape[0])) - 1.0)
        d1val = d1[inx1]
        inx2 = int(math.floor(float(self.percent) / 100.0 * float(d2.shape[0])) - 1.0)
        d2val = d2[inx2]

        d_total = max(np.average(d1val), np.average(d2val))

        return d_total
