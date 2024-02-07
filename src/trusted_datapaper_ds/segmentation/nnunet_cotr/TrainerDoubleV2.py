import gc
import os

import monai.transforms as mt
import nibabel as nib
import numpy as np
import telegram_send as ts
import torch
from CustomDataset import CustomDataset
from einops import rearrange
from monai.metrics import compute_meandice
from monai.transforms import (
    Compose,
    RandAffined,
    RandCropByLabelClassesd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
)
from monai.utils import set_determinism
from nnunet.utilities.nd_softmax import softmax_helper
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.tools import (
    _to_one_hot,
    create_path_if_not_exists,
    create_split_v3,
    get_loss,
    import_model,
    poly_lr,
)

"""## Set deterministic training for reproducibility"""
set_determinism(seed=0)


class Trainer:
    def __init__(self, cfg, log, *args, **kwargs):
        # Logs
        self.log = log
        self.dbg = cfg.training.dbg
        self.writer = SummaryWriter(
            log_dir="tensorboard/"
            + cfg.dataset.name
            + "_"
            + cfg.training.name
            + "_"
            + cfg.model.name
            + "_"
            + cfg.dataset.cv
        )
        self.dataset_name = cfg.dataset.name
        self.training_name = cfg.training.name
        self.model_name = cfg.model.name
        self.path = create_path_if_not_exists(
            os.path.join(
                cfg.training.pth,
                cfg.dataset.name,
                cfg.training.name,
                cfg.model.name,
                cfg.dataset.cv,
            )
        )

        # Device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.training.gpu)
        # torch.cuda.set_device(cfg.training.gpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_gpu = cfg.training.use_gpu
        torch.backends.cudnn.benchmark = True

        # Hyperparameters
        log.debug("Hyperparameters")
        self.epochs = cfg.training.epochs
        self.start_epoch = 0
        self.initial_lr = cfg.training.lr
        self.batch_size = cfg.training.batch_size
        self.num_workers = cfg.training.num_workers
        self.crop_size = cfg.training.crop_size
        self.iterations = cfg.training.iter
        self.weight_decay = cfg.training.weight_decay
        self.net_num_pool_op_kernel_sizes = cfg.model.net_num_pool_op_kernel_sizes
        self.net_conv_kernel_sizes = cfg.model.net_conv_kernel_sizes
        self.do_clip = cfg.training.do_clip
        self.do_schedul = cfg.training.do_schedul
        self._loss = cfg.training.loss

        # Dataset
        log.debug("Dataset")
        self.online_validation = cfg.training.online_validation
        self.eval_step = cfg.training.eval_step
        self.img_size = cfg.dataset.im_size

        self.seg_path = [
            cfg.dataset.path.seg1,
            cfg.dataset.path.seg2,
            cfg.dataset.path.seg12,
        ]
        if "ct" in cfg.dataset.name:
            self.train_split = create_split_v3(
                cfg.dataset.path.im,
                self.seg_path,
                cv=cfg.dataset.cv,
                log=log,
                data="ct",
            )
            self.val_split = create_split_v3(
                cfg.dataset.path.im,
                self.seg_path,
                cv=cfg.dataset.cv,
                val=True,
                log=log,
                data="ct",
            )
        else:
            self.train_split = create_split_v3(
                cfg.dataset.path.im, self.seg_path, cv=cfg.dataset.cv, log=log
            )
            self.val_split = create_split_v3(
                cfg.dataset.path.im, self.seg_path, cv=cfg.dataset.cv, val=True, log=log
            )

        train_transforms = None
        test_transforms = None
        val_transforms = None

        if "full" not in cfg.training.name:
            test_transforms = Compose(
                [mt.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)]
            )

            val_transforms = Compose(
                [mt.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)]
            )

            train_transforms = Compose(
                [
                    RandFlipd(
                        keys=["image", "label", "label1", "label2"],
                        prob=0.25,
                        spatial_axis=0,
                    ),
                    RandFlipd(
                        keys=["image", "label", "label1", "label2"],
                        prob=0.25,
                        spatial_axis=1,
                    ),
                    RandFlipd(
                        keys=["image", "label", "label1", "label2"],
                        prob=0.25,
                        spatial_axis=2,
                    ),
                    RandAffined(
                        keys=["image", "label", "label1", "label2"],
                        rotate_range=(np.pi, np.pi, np.pi),
                        translate_range=(50, 50, 50),
                        padding_mode="border",
                        scale_range=(0.25, 0.25, 0.25),
                        mode=("bilinear", "nearest", "nearest", "nearest"),
                        prob=1.0,
                    ),
                    Resized(
                        keys=["image", "label", "label1", "label2"],
                        spatial_size=self.img_size,
                        mode="trilinear",
                    ),
                    mt.NormalizeIntensityd(
                        keys="image", nonzero=True, channel_wise=True
                    ),
                    RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                    mt.RandGaussianNoised(keys="image", prob=0.1, mean=0.0, std=0.1),
                    mt.RandGaussianSmoothd(
                        keys="image",
                        sigma_x=(0.5, 1),
                        sigma_y=(0.5, 1),
                        sigma_z=(0.5, 1),
                        prob=0.2,
                    ),
                    mt.RandAdjustContrastd(keys="image", prob=0.15),
                ]
            )
        else:
            test_transforms = Compose(
                [
                    mt.NormalizeIntensityd(
                        keys="image", nonzero=True, channel_wise=True
                    ),
                ]
            )

            val_transforms = Compose(
                [
                    RandCropByLabelClassesd(
                        keys=["image", "label", "label1", "label2"],
                        label_key="label",
                        spatial_size=self.crop_size,
                        num_classes=cfg.dataset.classes + 1,
                        num_samples=1,
                    ),
                    mt.NormalizeIntensityd(
                        keys="image", nonzero=True, channel_wise=True
                    ),
                ]
            )

            train_transforms = Compose(
                [
                    mt.NormalizeIntensityd(
                        keys="image", nonzero=True, channel_wise=True
                    ),
                    RandCropByLabelClassesd(
                        keys=["image", "label", "label1", "label2"],
                        label_key="label",
                        spatial_size=self.crop_size,
                        num_classes=cfg.dataset.classes + 1,
                        num_samples=1,
                    ),
                    RandFlipd(
                        keys=["image", "label", "label1", "label2"],
                        prob=0.25,
                        spatial_axis=0,
                    ),
                    RandFlipd(
                        keys=["image", "label", "label1", "label2"],
                        prob=0.25,
                        spatial_axis=1,
                    ),
                    RandFlipd(
                        keys=["image", "label", "label1", "label2"],
                        prob=0.25,
                        spatial_axis=2,
                    ),
                    RandAffined(
                        keys=["image", "label", "label1", "label2"],
                        rotate_range=(np.pi, np.pi, np.pi),
                        translate_range=(50, 50, 50),
                        padding_mode="border",
                        scale_range=(0.25, 0.25, 0.25),
                        mode=("bilinear", "nearest", "nearest", "nearest"),
                        prob=1.0,
                    ),
                    RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                    mt.RandGaussianNoised(keys="image", prob=0.1, mean=0.0, std=0.1),
                    mt.RandGaussianSmoothd(
                        keys="image",
                        sigma_x=(0.5, 1),
                        sigma_y=(0.5, 1),
                        sigma_z=(0.5, 1),
                        prob=0.2,
                    ),
                    mt.RandAdjustContrastd(keys="image", prob=0.15),
                ]
            )

        trainData = CustomDataset(
            self.train_split,
            transform=train_transforms,
            iterations=self.iterations,
            log=log,
            net_num_pool_op_kernel_sizes=self.net_num_pool_op_kernel_sizes,
        )
        testData = CustomDataset(
            self.val_split,
            transform=test_transforms,
            iterations=0,
            log=log,
            type_="test",
        )

        if self.online_validation:
            valData = CustomDataset(
                self.val_split,
                transform=val_transforms,
                iterations=0,
                log=log,
                type_="val",
            )

        self.train_loader = DataLoader(
            trainData,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        log.debug("train_loader", len(self.train_loader))
        self.test_loader = DataLoader(
            testData,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=torch.cuda.is_available(),
        )
        if self.online_validation:
            self.val_loader = DataLoader(
                valData,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                pin_memory=torch.cuda.is_available(),
            )

        self.stride = cfg.training.inference.stride
        self.classes = cfg.dataset.classes

        self.classes += 1

        # Models
        log.debug("Model")
        self.feature_size = cfg.model.feature_size
        self.save_path = create_path_if_not_exists(
            os.path.join(self.path, "checkpoint")
        )
        self.n_save = cfg.training.checkpoint.save
        self.do_load_checkpoint = cfg.training.checkpoint.load
        self.load_path = os.path.join(self.path, "checkpoint", "latest.pt")

        self.model = import_model(
            cfg.model.model,
            dataset="US",
            num_classes=self.classes,
            num_pool=len(self.net_num_pool_op_kernel_sizes),
            pool_op_kernel_sizes=self.net_num_pool_op_kernel_sizes,
            conv_kernel_sizes=self.net_conv_kernel_sizes,
            cfg=cfg.model,
            log=log,
            img_size=self.crop_size,
            feature_size=self.feature_size,
        )

        if torch.cuda.is_available() and self.use_gpu:
            self.model.cuda()

        self.model.inference_apply_nonlin = softmax_helper

        self.lr = self.initial_lr

        if cfg.training.optim == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                self.lr,
                weight_decay=self.weight_decay,
                momentum=0.99,
                nesterov=True,
            )
        elif cfg.training.optim == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), self.lr, weight_decay=self.weight_decay
            )

        self.loss = get_loss(self.net_num_pool_op_kernel_sizes)

        log.debug("Loss", self._loss)
        self.infer_path = self.path

        if self.do_load_checkpoint:
            log.debug("Checkpoint")
            self.load_checkpoint()

    def run_training(self, *args, **kwargs):
        log = self.log
        if not self.dbg:
            ts.send(
                messages=[
                    "Training: "
                    + self.dataset_name
                    + "_"
                    + self.training_name
                    + "_"
                    + self.model_name
                ]
            )

        best_metric = -1

        log.debug("run_training")

        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()

            self.optimizer.param_groups[0]["lr"] = self.lr
            btc = 0
            l_train = 0
            log.debug("epoch{}".format(epoch))

            for batch_data in tqdm(self.train_loader):
                btc += 1
                self.optimizer.zero_grad()

                inputs = batch_data["image"]
                labels = batch_data["label"]
                labels1 = batch_data["label1"]
                labels2 = batch_data["label2"]

                if torch.cuda.is_available() and self.use_gpu:
                    inputs = inputs.float().cuda(0)
                    for lab in range(len(labels)):
                        labels[lab] = labels[lab].cuda(0)
                        labels1[lab] = labels1[lab].cuda(0)
                        labels2[lab] = labels2[lab].cuda(0)

                output = self.model(inputs)

                del inputs
                if len(self.net_num_pool_op_kernel_sizes) == 0:
                    labels = labels.cuda(0)
                    labels1 = labels1.cuda(0)
                    labels2 = labels2.cuda(0)

                gc.collect()

                ltorch = (self.loss(output, labels1) + self.loss(output, labels2)) * 0.5
                l_train += ltorch.detach().cpu().numpy()

                gc.collect()
                ltorch.backward()

                if self.do_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                self.optimizer.step()

                for out in output:
                    del out
                del output
                for lab in labels:
                    del lab
                for lab in labels1:
                    del lab
                for lab in labels2:
                    del lab
                del labels
                del labels1
                del labels2

                gc.collect()

            l_train = l_train / btc
            l_val = 0
            if self.online_validation and ((epoch + 1) % self.eval_step == 0):
                self.model.eval()
                len_val = 0

                with torch.no_grad():
                    for batch_data in tqdm(self.val_loader):
                        inputs = batch_data["image"]
                        labels = batch_data["label"]
                        if torch.cuda.is_available() and self.use_gpu:
                            inputs = inputs.float().cuda(0)
                            labels = labels.long().cuda(0)
                        output = self.model(inputs)

                        output = output[0]
                        output = torch.argmax(output, dim=1)
                        labels = _to_one_hot(
                            labels[0, 0, ...], num_classes=self.classes
                        )
                        output = _to_one_hot(output[0, ...], num_classes=self.classes)
                        labels = rearrange(labels, "z x y c -> c z x y")[None, ...]
                        output = rearrange(output, "z x y c -> c z x y")[None, ...]
                        ltorch = compute_meandice(output, labels, ignore_empty=False)
                        l_val += np.mean(ltorch.cpu().numpy()[0][1:])

                        len_val += 1

                l_val = l_val / len_val
                if l_val > best_metric:
                    best_metric = l_val
                    self.save_chekpoint(epoch, "best.pt")

                self.writer.add_scalar("Val Dice", l_val, epoch)

            saved_txt = ""
            if (epoch + 1) % self.n_save == 0:
                self.save_chekpoint(epoch)
                saved_txt = " :: Saved!"
                log.info(
                    "Epoch: {}".format(epoch),
                    "Val Dice: {}, lr: {}{}".format(l_val, self.lr, saved_txt),
                )
            log.info("Epoch: {}".format(epoch), "Train Loss: {}".format(l_train))
            self.writer.add_scalar("Loss", l_train, epoch)

            self.writer.add_scalar("lr", self.lr, epoch)
            if self.do_schedul:
                self.lr = poly_lr(epoch, self.epochs, self.initial_lr, 0.9)
            torch.cuda.empty_cache()

        if not self.dbg:
            ts.send(
                messages=[
                    "Training END: "
                    + self.dataset_name
                    + "_"
                    + self.training_name
                    + "_"
                    + self.model_name
                ]
            )

    def run_eval(self, *args, **kwargs):
        self.load_checkpoint(os.path.join(self.path, "checkpoint", "best.pt"))
        print("loaded checkpoint: ", os.path.join(self.path, "checkpoint", "best.pt"))

        self.model.eval()

        for batch_data in tqdm(self.test_loader):
            inputs = batch_data["image"]
            affine = batch_data["affine"][0, ...].numpy()

            prediction = self.inference(inputs)

            prediction = torch.argmax(prediction, dim=1)[0, ...]
            idx = batch_data["id"][0][0]

            name = idx.replace("xxx", "mask")
            file = os.path.join(self.infer_path, name)

            prediction = prediction.numpy().astype(np.float32)

            pred_nib = nib.Nifti1Image(prediction, affine)
            nib.save(pred_nib, file)

        return pred_nib

    def save_chekpoint(self, epoch, txt="latest.pt"):
        state_dict = self.model.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        save_this = {
            "epoch": epoch + 1,
            "state_dict": state_dict,
            "optimizer_state_dict": optimizer_state_dict,
        }
        torch.save(save_this, os.path.join(self.save_path, txt))

    def load_checkpoint(self, txt=None):
        if txt is None:
            txt = self.load_path
            print("checkpoint loaded: ", txt)

        checkpoint = torch.load(txt)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.start_epoch = checkpoint["epoch"]
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def inference(self, inputs):
        D_crop, H_crop, W_crop = self.crop_size
        B, C, D, H, W = inputs.shape

        nD, nH, nW = (
            int(D // (D_crop * self.stride[2])),
            int(H // (H_crop * self.stride[0])),
            int(W // (W_crop * self.stride[1])),
        )

        output = torch.zeros((B, self.classes, D, H, W))
        count = torch.zeros((B, self.classes, D, H, W))

        for k in range(nD):
            for i in range(nH):
                for j in range(nW):
                    idx_d = int(k * D_crop * self.stride[0])
                    idx_h = int(i * H_crop * self.stride[1])
                    idx_w = int(j * W_crop * self.stride[2])

                    if idx_d + D_crop > D:
                        idx_d = D - D_crop
                    if idx_h + H_crop > H:
                        idx_h = H - H_crop
                    if idx_w + W_crop > W:
                        idx_w = W - W_crop

                    crop = inputs[
                        :,
                        :,
                        idx_d : idx_d + D_crop,
                        idx_h : idx_h + H_crop,
                        idx_w : idx_w + W_crop,
                    ]

                    if torch.cuda.is_available() and self.use_gpu:
                        crop = crop.float().cuda(0)

                    with torch.no_grad():
                        out_crop = self.model(crop)

                    output[
                        :,
                        :,
                        idx_d : idx_d + D_crop,
                        idx_h : idx_h + H_crop,
                        idx_w : idx_w + W_crop,
                    ] = out_crop[0].cpu()

                    del crop, out_crop

                    count[
                        :,
                        :,
                        idx_d : idx_d + D_crop,
                        idx_h : idx_h + H_crop,
                        idx_w : idx_w + W_crop,
                    ] += 1

                    gc.collect()

        return output / count
