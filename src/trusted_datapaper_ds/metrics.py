"""
    Copyright (C) 2022-2024 IRCAD France - All rights reserved. *
    This file is part of Disrumpere. *
    Disrumpere can not be copied, modified and/or distributed without
    the express permission of IRCAD France.
"""

import math

import numpy as np
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose, EnsureType
from numpy import linalg as LA

from trusted_datapaper_ds import geometry_utils as gu

process = Compose(
    [EnsureType(data_type="tensor"), AsDiscrete(threshold=0.5, threshold_values=True)]
)

dice_metric = DiceMetric(include_background=False, reduction="mean")


class Dice:
    def __init__(self, prednparray, gtnparray):
        self.prednparray = prednparray
        self.gtnparray = gtnparray
        self.pred = process(self.prednparray)
        self.gt = process(self.gtnparray)
        return

    def evaluate_dice(self):
        dice_metric(y_pred=self.pred, y=self.gt)
        dice = dice_metric.aggregate().item()
        dice_metric.reset()
        return dice


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


class HausMesh:
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


class MeanTRE:
    def evaluate_pcd(self, pcd1, pcd2):
        a = np.asarray(pcd1.points)
        b = np.asarray(pcd2.points)
        d = LA.norm(a - b, axis=1)
        return np.mean(d)
