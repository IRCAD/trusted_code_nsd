import importlib
import os

import numpy as np
import torch
from batchgenerators.augmentations.utils import (
    convert_seg_image_to_one_hot_encoding_batched,
)
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from termcolor import colored
from torch.nn.functional import avg_pool3d


class CustomDice:
    def __init__(self, log=None):
        self.log = log

    def __call__(self, input, target):
        log = self.log
        smooth = 1.0

        log.debug("input", input.shape)
        log.debug("target", target.shape)

        iflat = input.view(-1)
        log.debug("iflat", iflat.shape)
        log.debug("iflat sum", iflat.sum())

        tflat = target.view(-1)
        log.debug("tflat", tflat.shape)
        log.debug("tflat sum", tflat.sum())

        intersection = iflat * tflat
        log.debug("intersection", intersection.shape)
        log.debug("intersection sum", intersection.sum())

        intersection = intersection.sum()
        log.debug("intersection_sum", intersection.shape)
        log.debug("intersection_sum item", intersection.item())

        ret = 1 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
        log.debug("ret", ret.shape)
        log.debug("ret item", ret.item())

        return ret


def _to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype, device=y.device)

    return zeros.scatter(scatter_dim, y_tensor, 1)


def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs) ** exponent


def downsample_seg_for_ds_transform3(
    seg,
    ds_scales=((1, 1, 1), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
    classes=None,
    log=None,
):
    output = []
    one_hot = torch.from_numpy(
        convert_seg_image_to_one_hot_encoding_batched(seg[:, 0], classes)
    )  # b, c,

    for s in ds_scales:
        if all([i == 1 for i in s]):
            # if log != None:
            # log.debug("here we are", seg.shape)
            output.append(seg[:, 0])
            # output.append(one_hot[0,...])
        else:
            kernel_size = tuple(int(1 / i) for i in s)
            stride = kernel_size
            pad = tuple((i - 1) // 2 for i in kernel_size)
            pooled = avg_pool3d(
                one_hot,
                kernel_size,
                stride,
                pad,
                count_include_pad=False,
                ceil_mode=False,
            )

            output.append(pooled[0, ...])
    return output


def get_loss(net_num_pool_op_kernel_sizes):
    # Here we wrap the loss for deep supervision
    # we need to know the number of outputs of the network
    loss = DC_and_CE_loss(
        {"batch_dice": True, "smooth": 1e-5, "do_bg": False}, {}
    )  # maybe batch dice false
    net_numpool = len(net_num_pool_op_kernel_sizes)

    # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
    # this gives higher resolution outputs more weight in the loss
    weights = np.array([1 / (2**i) for i in range(net_numpool)])

    # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
    mask = np.array(
        [True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)]
    )
    weights[~mask] = 0
    weights = weights / weights.sum()
    ds_loss_weights = weights
    # now wrap the loss
    loss = MultipleOutputLoss2(loss, ds_loss_weights)

    return loss


def import_model(name, *args, **kwargs):
    return importlib.import_module("models." + name).model(**kwargs)


def create_split_v3(
    im_pth, seg_pth, val=False, cv="cv1", log=None, data="us", *args, **kwargs
):
    if data == "us":
        all_splits = {
            "cv1": [
                "84R",
                "84L",
                "116R",
                "116L",
                "114R",
                "98R",
                "98L",
                "118R",
                "118L",
                "04L",
                "44R",
                "44L",
            ],
            "cv2": [
                "47R",
                "01R",
                "01L",
                "74R",
                "14R",
                "14L",
                "33L",
                "72L",
                "51R",
                "51L",
            ],
            "cv3": [
                "05R",
                "57R",
                "57L",
                "43R",
                "43L",
                "94R",
                "131L",
                "11R",
                "11L",
                "54R",
                "12L",
                "15R",
                "17L",
            ],
            "cv4": [
                "21R",
                "21L",
                "03R",
                "03L",
                "97R",
                "97L",
                "27R",
                "27L",
                "23R",
                "23L",
                "53L",
                "119R",
            ],
            "cv5": [
                "125R",
                "125L",
                "20R",
                "42R",
                "99R",
                "02R",
                "02L",
                "87R",
                "87L",
                "37R",
                "38R",
                "38L",
            ],
        }
        teuteu = ""

    else:
        all_splits = {
            "cv1": ["84", "116", "114", "98", "118", "04", "44", "101", "88"],
            "cv2": ["47", "01", "74", "14", "33", "24", "72", "51", "64"],
            "cv3": ["05", "57", "43", "94", "131", "11", "54", "12", "15", "17"],
            "cv4": ["21", "03", "97", "27", "23", "09", "61", "32", "53", "119"],
            "cv5": ["125", "20", "42", "99", "02", "87", "37", "38", "107", "31"],
        }
        teuteu = "_"

    if val:
        split = all_splits[cv]
    else:
        split = []
        for c in list(all_splits.keys()):
            if c != cv:
                split += all_splits[c]

    splits = []
    for spl in split:
        log.debug(
            "file 	", os.path.join(im_pth, spl + "_img" + data.upper() + ".nii.gz")
        )
        if os.path.exists(
            os.path.join(im_pth, spl + "_img" + data.upper() + ".nii.gz")
        ):
            tmp = {
                "image": os.path.join(im_pth, spl + "_img" + data.upper() + ".nii.gz"),
                "label1": os.path.join(
                    seg_pth[0], spl + teuteu + "1_mask" + data.upper() + ".nii.gz"
                ),
                "label2": os.path.join(
                    seg_pth[1], spl + teuteu + "2_mask" + data.upper() + ".nii.gz"
                ),
                "label": os.path.join(
                    seg_pth[2], spl + "_mask" + data.upper() + ".nii.gz"
                ),
                "id": spl,
            }
            # log.debug('paths and ids', tmp)
            splits.append(tmp)
    return splits


class Log(object):
    """docstring for Log"""

    def __init__(self, log):
        super(Log, self).__init__()
        self.log = log

    def debug(self, msg, info=""):
        self.log.debug(colored(msg + "\n", "red") + colored(str(info), "magenta"))

    def info(self, msg, info):
        self.log.info(colored(msg + "\n", "blue") + colored(str(info), "green"))

    def start(self, msg):
        self.log.info(colored("Start " + msg + " ...\n", "blue"))

    def end(self, msg):
        self.log.info(colored("... " + msg + " Done\n", "blue"))
