# import numpy as np
# from geometry_metrics import geometry_utils as gu
# from monai.metrics import DiceMetric, HausdorffDistanceMetric
# from monai.transforms import AddChannel, AsDiscrete, Compose, ToTensor

# process = Compose([AddChannel(), ToTensor()])
# binarizing = AsDiscrete(threshold_values=True)
# dice_metric = DiceMetric(include_background=True, reduction="mean")
# haus_mask_metric = HausdorffDistanceMetric(
#     include_background=False,
#     distance_metric="euclidean",
#     percentile=95,
#     directed=False,
#     reduction="mean",
#     get_not_nans=False,
# )


# class Dice:
#     def __init__(self, map1, map2):
#         self.map1 = map1
#         self.map2 = map2
#         return

#     def evaluate_overlap(self):
#         map1 = process(self.map1)
#         map1 = binarizing(map1)
#         map2 = process(self.map2)
#         map2 = binarizing(map2)

#         dice_metric(y_pred=map1, y=map2)
#         dice = dice_metric.aggregate().item()
#         dice_metric.reset()
#         return dice


# class HausdorffMask:
#     def __init__(self, map1, map2):
#         self.map1 = map1
#         self.map2 = map2
#         return

#     def evaluate_overlap(self):
#         map1 = process(self.map1)
#         map1 = binarizing(map1)
#         map2 = process(self.map2)
#         map2 = binarizing(map2)

#         haus_mask_metric(y_pred=map1, y=map2)
#         haus_mask = haus_mask_metric.aggregate().item()
#         haus_mask_metric.reset()
#         return haus_mask


# class MeanNNDistance:
#     def evaluate_mesh(self, msh1, msh2):
#         d1, indx1 = gu.nearest_neighbor(
#             np.asarray(msh1.vertices), np.asarray(msh2.vertices)
#         )
#         d2, indx2 = gu.nearest_neighbor(
#             np.asarray(msh2.vertices), np.asarray(msh1.vertices)
#         )
#         d_total = (np.average(d1) + np.average(d2)) / 2.0

#         return d_total

#     def evaluate_pcd(self, pcd1, pcd2):
#         d1, indx1 = gu.nearest_neighbor(
#             np.asarray(pcd1.points), np.asarray(pcd2.points)
#         )
#         d2, indx2 = gu.nearest_neighbor(
#             np.asarray(pcd2.points), np.asarray(pcd1.points)
#         )
#         d_total = (np.average(d1) + np.average(d2)) / 2.0

#         return d_total
