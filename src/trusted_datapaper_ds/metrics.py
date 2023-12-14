import math

import numpy as np
from geometry_utils import geometry_utils as gu
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AddChannel, AsDiscrete, Compose, ToTensor
from numpy import linalg as LA

process = Compose([AddChannel(), ToTensor()])
binarizing = AsDiscrete(threshold_values=True)
dice_metric = DiceMetric(include_background=True, reduction="mean")
haus_mask_metric = HausdorffDistanceMetric(
    include_background=False,
    distance_metric="euclidean",
    percentile=95,
    directed=False,
    reduction="mean",
    get_not_nans=False,
)


class Dice:
    def __init__(self, map1, map2):
        self.map1 = map1
        self.map2 = map2
        return

    def evaluate_overlap(self):
        map1 = process(self.map1)
        map1 = binarizing(map1)
        map2 = process(self.map2)
        map2 = binarizing(map2)

        dice_metric(y_pred=map1, y=map2)
        dice = dice_metric.aggregate().item()
        dice_metric.reset()
        return dice


class HausdorffMask:
    def __init__(self, map1, map2):
        self.map1 = map1
        self.map2 = map2
        return

    def evaluate_overlap(self):
        map1 = process(self.map1)
        map1 = binarizing(map1)
        map2 = process(self.map2)
        map2 = binarizing(map2)

        haus_mask_metric(y_pred=map1, y=map2)
        haus_mask = haus_mask_metric.aggregate().item()
        haus_mask_metric.reset()
        return haus_mask


class AsymmetricNNDistance:
    def evaluate_nesh(self, msh1, msh2):
        d, indx = gu.nearest_neighbor(
            np.asarray(msh1.vertices), np.asarray(msh2.vertices)
        )
        d_total = np.average(d)
        return d_total

    def evaluate_pcd(self, pcd1, pcd2):
        d, indx = gu.nearest_neighbor(np.asarray(pcd1.points), np.asarray(pcd2.points))
        d_total = np.average(d)
        return d_total


class AsymmetricNNmaxDistance:
    def evaluate_pcd(self, pcd1, pcd2):
        d, indx = gu.nearest_neighbor(np.asarray(pcd1.points), np.asarray(pcd2.points))
        d_max = np.max(d)
        return d_max


class AsymmetricNNminDistance:
    def evaluate_pcd(self, pcd1, pcd2):
        d, indx = gu.nearest_neighbor(np.asarray(pcd1.points), np.asarray(pcd2.points))
        d_min = np.min(d)
        return d_min


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

        d_total = (np.average(d1val) + np.average(d2val)) / 2.0

        # d = np.concatenate((d1, d2))
        # d_sorted = np.sort(d)
        # inx = int(math.floor(float(self.percent) / 100.0 * float(d_sorted.shape[0])) - 1.0)
        # d_total = d_sorted[inx]
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

        # d_total = (np.average(d1val) + np.average(d2val)) / 2.0
        d_total = max(np.average(d1val), np.average(d2val))

        # d = np.concatenate((d1, d2))
        # d_sorted = np.sort(d)
        # inx = int(math.floor(float(self.percent) / 100.0 * float(d_sorted.shape[0])) - 1.0)
        # d_total = d_sorted[inx]
        return d_total


class MeanTRE:
    def evaluate_pcd(self, pcd1, pcd2):
        a = np.asarray(pcd1.points)
        b = np.asarray(pcd2.points)
        d = LA.norm(a - b, axis=1)
        return np.mean(d)


class AsymmetricMeanNNDistance:
    def evaluate_mesh(self, msh1, msh2):
        d1, indx = gu.nearest_neighbor(
            np.asarray(msh1.vertices), np.asarray(msh2.vertices)
        )
        d2, indx = gu.nearest_neighbor(
            np.asarray(msh2.vertices), np.asarray(msh1.vertices)
        )
        d1 = np.sort(d1)
        d2 = np.sort(d2)
        inx1 = int(math.floor(float(100) / 100.0 * float(d1.shape[0])) - 1.0)
        d1val = d1[inx1]
        inx2 = int(math.floor(float(100) / 100.0 * float(d2.shape[0])) - 1.0)
        d2[inx2]
        d_total = np.average(d1val)
        return d_total

        # # d_total = (np.average(d1val) + np.average(d2val)) / 2.0
        #
        # d1, indx1 = gu.nearest_neighbor(np.asarray(msh1.vertices), np.asarray(msh2.vertices))
        # print(np.asarray(msh1.vertices).shape[0])
        #
        # d_total = np.average(d1)
        # d_total = np.median(d1)
        # return d_total

    def evaluate_pcd(self, pcd1, pcd2):
        d1, indx = gu.nearest_neighbor(np.asarray(pcd1.points), np.asarray(pcd2.points))
        d2, indx = gu.nearest_neighbor(np.asarray(pcd2.points), np.asarray(pcd1.points))
        d1 = np.sort(d1)
        d2 = np.sort(d2)
        inx1 = int(math.floor(float(100) / 100.0 * float(d1.shape[0])) - 1.0)
        d1val = d1[inx1]
        inx2 = int(math.floor(float(100) / 100.0 * float(d2.shape[0])) - 1.0)
        d2val = d2[inx2]

        d_total = (np.average(d1val) + np.average(d2val)) / 2.0
        # d_total = np.average(d1val)
        return d_total

        #
        # d1, indx1 = gu.nearest_neighbor(np.asarray(pcd1.points), np.asarray(pcd2.points))
        # d_total = np.average(d1)
        # d_total = np.median(d1)
        #
        # # d = np.concatenate((d1, d2))
        # # d_total = np.average(d)
        #
        # return d_total


class MeanNNDistance:
    def evaluate_mesh(self, msh1, msh2):
        d1, indx1 = gu.nearest_neighbor(
            np.asarray(msh1.vertices), np.asarray(msh2.vertices)
        )
        d2, indx2 = gu.nearest_neighbor(
            np.asarray(msh2.vertices), np.asarray(msh1.vertices)
        )
        d_total = (np.average(d1) + np.average(d2)) / 2.0

        # d = np.concatenate((d1, d2))
        # d_total = np.average(d)
        return d_total

    def evaluate_pcd(self, pcd1, pcd2):
        d1, indx1 = gu.nearest_neighbor(
            np.asarray(pcd1.points), np.asarray(pcd2.points)
        )
        d2, indx2 = gu.nearest_neighbor(
            np.asarray(pcd2.points), np.asarray(pcd1.points)
        )
        d_total = (np.average(d1) + np.average(d2)) / 2.0

        # d = np.concatenate((d1, d2))
        # d_total = np.average(d)

        return d_total
