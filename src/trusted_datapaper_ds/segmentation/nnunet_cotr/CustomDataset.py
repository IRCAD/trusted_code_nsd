import random

import monai
import numpy as np
import torch
from monai.transforms import LoadImage, apply_transform
from torch.utils.data import Dataset
from utils.tools import downsample_seg_for_ds_transform3

from trusted_datapaper_ds.dataprocessing import data as dt


class CustomDataset(Dataset):
    def __init__(
        self,
        data,
        transform=None,
        iterations=250,
        log=None,
        net_num_pool_op_kernel_sizes=[],
        type_="train",
        multi_anno=True,
        num_classes=2,
        name="us",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # We use our own Custom dataset wich with we can keep track of sub volumes position.
        self.data = monai.data.Dataset(data)
        self.iterations = iterations
        self.loader = LoadImage()
        self.n_data = len(data)
        self.transform = transform
        self.log = log
        self.type = type_
        self.multi_anno = multi_anno
        self.num_classes = num_classes
        self.name = name
        self.net_num_pool_op_kernel_sizes = net_num_pool_op_kernel_sizes
        self.idx = -1

    def __len__(self):
        if self.type == "train":
            if self.iterations == 0:
                return len(self.data)
            return self.iterations
        else:
            return len(self.data)

    def __getitem__(self, index):
        self.log
        if self.type == "train":
            if self.iterations == 0:
                self.idx += 1
                i = self.idx
            else:
                i = random.randint(0, self.n_data - 1)
        else:
            self.idx += 1
            i = self.idx

        data_i = {}

        img = dt.Image(self.data[i]["image"])
        mask = dt.Mask(self.data[i]["label"], annotatorID="gt")
        data_i["image"] = np.expand_dims(img.nparray, axis=0)
        data_i["label"] = np.expand_dims(mask.nparray, axis=0)

        if self.multi_anno:
            mask1 = dt.Mask(self.data[i]["label1"], annotatorID="1")
            mask2 = dt.Mask(self.data[i]["label2"], annotatorID="2")
            data_i["label1"] = np.expand_dims(mask1.nparray, axis=0)
            data_i["label2"] = np.expand_dims(mask2.nparray, axis=0)

        data_i["id"] = [self.data[i]["image"].split("/")[-1].replace("_img", "_xxx")]
        data_i["affine"] = img.nibaffine
        data_i["size"] = img.size

        if self.transform is not None:
            tmp = apply_transform(self.transform, data_i)
            if isinstance(tmp, list):
                data_i = tmp[0] if self.transform is not None else data_i
            else:
                data_i = tmp if self.transform is not None else data_i

            data_i["image"] = torch.from_numpy(data_i["image"])

        if self.net_num_pool_op_kernel_sizes != []:
            deep_supervision_scales = [[1, 1, 1]] + list(
                list(i)
                for i in 1
                / np.cumprod(np.vstack(self.net_num_pool_op_kernel_sizes), axis=0)
            )[:-1]

            data_i["label"] = downsample_seg_for_ds_transform3(
                data_i["label"][None, ...],
                deep_supervision_scales,
                classes=[i for i in range(self.num_classes)],
                log=self.log,
            )

            if self.multi_anno:
                data_i["label1"] = downsample_seg_for_ds_transform3(
                    data_i["label1"][None, ...],
                    deep_supervision_scales,
                    classes=[i for i in range(self.num_classes)],
                    log=self.log,
                )
                data_i["label2"] = downsample_seg_for_ds_transform3(
                    data_i["label2"][None, ...],
                    deep_supervision_scales,
                    classes=[i for i in range(self.num_classes)],
                    log=self.log,
                )

        return data_i
