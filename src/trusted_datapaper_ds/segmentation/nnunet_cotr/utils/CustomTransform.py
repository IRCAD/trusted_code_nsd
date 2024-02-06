from copy import deepcopy
from itertools import chain
from math import ceil
from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms import RandSpatialCropd
from monai.transforms.croppad.array import BorderPad, RandCropByPosNegLabel, SpatialCrop
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform, Randomizable
from monai.transforms.utils import (
    generate_pos_neg_label_crop_centers,
    map_binary_to_indices,
)
from monai.utils import ImageMetaKey as Key
from monai.utils import ensure_tuple, ensure_tuple_rep, fall_back_tuple
from monai.utils.enums import TraceKeys

# from tools import log_debug, log_info, log_start, log_end


class CustomRandCropByPosNegLabeld(Randomizable, MapTransform, InvertibleTransform):
    # This is our own custom crop function wich give us the center position of each crop volumes
    """
    Dictionary-based version :py:class:`monai.transforms.RandCropByPosNegLabel`.
    Crop random fixed sized regions with the center being a foreground or background voxel
    based on the Pos Neg Ratio.
    Suppose all the expected fields specified by `keys` have same shape,
    and add `patch_index` to the corresponding meta data.
    And will return a list of dictionaries for all the cropped images.

    If a dimension of the expected spatial size is bigger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than the expected size,
    and the cropped results of several images may not have exactly the same shape.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        label_key: name of key for label image, this will be used for finding foreground/background.
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            if a dimension of ROI size is bigger than image size, will not crop that dimension of the image.
            if its components have non-positive values, the corresponding size of `data[label_key]` will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `spatial_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        image_key: if image_key is not None, use ``label == 0 & image > image_threshold`` to select
            the negative sample(background) center. so the crop center will only exist on valid image area.
        image_threshold: if enabled image_key, use ``image > image_threshold`` to determine
            the valid image content area.
        fg_indices_key: if provided pre-computed foreground indices of `label`, will ignore above `image_key` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices_key`
            and `bg_indices_key` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndicesd` transform first and cache the results.
        bg_indices_key: if provided pre-computed background indices of `label`, will ignore above `image_key` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices_key`
            and `bg_indices_key` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndicesd` transform first and cache the results.
        meta_keys: explicitly indicate the key of the corresponding meta data dictionary.
            used to add `patch_index` to the meta dict.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the meta data is a dictionary object which contains: filename, original_shape, etc.
            it can be a sequence of string, map to the `keys`.
            if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
        meta_key_postfix: if meta_keys is None, use `key_{postfix}` to to fetch the meta data according
            to the key data, default is `meta_dict`, the meta data is a dictionary object.
            used to add `patch_index` to the meta dict.
        allow_smaller: if `False`, an exception will be raised if the image is smaller than
            the requested ROI in any dimension. If `True`, any smaller dimensions will be set to
            match the cropped size (i.e., no cropping in that dimension).
        allow_missing_keys: don't raise exception if key is missing.

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """

    backend = RandCropByPosNegLabel.backend

    def __init__(
        self,
        keys: KeysCollection,
        label_key: str,
        spatial_size: Union[Sequence[int], int],
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image_key: Optional[str] = None,
        image_threshold: float = 0.0,
        fg_indices_key: Optional[str] = None,
        bg_indices_key: Optional[str] = None,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "meta_dict",
        allow_smaller: bool = False,
        allow_missing_keys: bool = False,
        log=None,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.label_key = label_key
        self.spatial_size: Union[Tuple[int, ...], Sequence[int], int] = spatial_size
        if pos < 0 or neg < 0:
            raise ValueError(
                f"pos and neg must be nonnegative, got pos={pos} neg={neg}."
            )
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.image_key = image_key
        self.image_threshold = image_threshold
        self.fg_indices_key = fg_indices_key
        self.bg_indices_key = bg_indices_key
        self.meta_keys = (
            ensure_tuple_rep(None, len(self.keys))
            if meta_keys is None
            else ensure_tuple(meta_keys)
        )
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.centers: Optional[List[List[int]]] = None
        self.allow_smaller = allow_smaller
        self.log = log

    def randomize(
        self,
        label: NdarrayOrTensor,
        fg_indices: Optional[NdarrayOrTensor] = None,
        bg_indices: Optional[NdarrayOrTensor] = None,
        image: Optional[NdarrayOrTensor] = None,
    ) -> None:
        # log_debug(log, "inputs.shape", inputs.shape)
        self.spatial_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
        if fg_indices is None or bg_indices is None:
            fg_indices_, bg_indices_ = map_binary_to_indices(
                label, image, self.image_threshold
            )
        else:
            fg_indices_ = fg_indices
            bg_indices_ = bg_indices
        self.centers = generate_pos_neg_label_crop_centers(
            self.spatial_size,
            self.num_samples,
            self.pos_ratio,
            label.shape[1:],
            fg_indices_,
            bg_indices_,
            self.R,
            self.allow_smaller,
        )

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> List[Dict[Hashable, NdarrayOrTensor]]:
        d = dict(data)
        label = d[self.label_key]
        image = d[self.image_key] if self.image_key else None
        fg_indices = (
            d.pop(self.fg_indices_key, None)
            if self.fg_indices_key is not None
            else None
        )
        bg_indices = (
            d.pop(self.bg_indices_key, None)
            if self.bg_indices_key is not None
            else None
        )

        self.randomize(label, fg_indices, bg_indices, image)
        if not isinstance(self.spatial_size, tuple):
            raise ValueError("spatial_size must be a valid tuple.")
        if self.centers is None:
            raise ValueError("no available ROI centers to crop.")

        # initialize returned list with shallow copy to preserve key ordering
        results: List[Dict[Hashable, NdarrayOrTensor]] = [
            dict(d) for _ in range(self.num_samples)
        ]

        for i, center in enumerate(self.centers):
            # fill in the extra keys with unmodified data
            for key in set(d.keys()).difference(set(self.keys)):
                results[i][key] = deepcopy(d[key])
            for key in self.key_iterator(d):
                img = d[key]
                cropper = SpatialCrop(
                    roi_center=tuple(center), roi_size=self.spatial_size
                )
                orig_size = img.shape[1:]
                results[i][key] = cropper(img)
                self.push_transform(
                    results[i], key, extra_info={"center": center}, orig_size=orig_size
                )
            # add `patch_index` to the meta data
            for key, meta_key, meta_key_postfix in self.key_iterator(
                d, self.meta_keys, self.meta_key_postfix
            ):
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key not in results[i]:
                    results[i][meta_key] = {}  # type: ignore
                results[i][meta_key][Key.PATCH_INDEX] = i  # type: ignore

        return results, self.centers

    def inverse(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            orig_size = np.asarray(transform[TraceKeys.ORIG_SIZE])
            current_size = np.asarray(d[key].shape[1:])
            center = transform[TraceKeys.EXTRA_INFO]["center"]
            cropper = SpatialCrop(roi_center=tuple(center), roi_size=self.spatial_size)  # type: ignore
            # get required pad to start and end
            pad_to_start = np.array(
                [s.indices(o)[0] for s, o in zip(cropper.slices, orig_size)]
            )
            pad_to_end = orig_size - current_size - pad_to_start
            # interleave mins and maxes
            pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
            inverse_transform = BorderPad(pad)
            # Apply inverse transform
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class CustomRandScaleCropd(RandSpatialCropd):
    """
    Dictionary-based version :py:class:`monai.transforms.RandScaleCrop`.
    Crop image with random size or specific size ROI.
    It can crop at a random position as center or at the image center.
    And allows to set the minimum and maximum scale of image size to limit the randomly generated ROI.
    Suppose all the expected fields specified by `keys` have same shape.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        roi_scale: if `random_size` is True, it specifies the minimum crop size: `roi_scale * image spatial size`.
            if `random_size` is False, it specifies the expected scale of image size to crop. e.g. [0.3, 0.4, 0.5].
            If its components have non-positive values, will use `1.0` instead, which means the input image size.
        max_roi_scale: if `random_size` is True and `roi_scale` specifies the min crop region size, `max_roi_scale`
            can specify the max crop region size: `max_roi_scale * image spatial size`.
            if None, defaults to the input image size. if its components have non-positive values,
            will use `1.0` instead, which means the input image size.
        random_center: crop at random position as center or the image center.
        random_size: crop with random size or specified size ROI by `roi_scale * image spatial size`.
            if True, the actual size is sampled from:
            `randint(roi_scale * image spatial size, max_roi_scale * image spatial size + 1)`.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = RandSpatialCropd.backend

    def __init__(
        self,
        keys: KeysCollection,
        roi_scale: Union[Sequence[float], float],
        max_roi_scale: Optional[Union[Sequence[float], float]] = None,
        random_center: bool = True,
        random_size: bool = True,
        allow_missing_keys: bool = False,
        prob: float = 0.2,
    ) -> None:
        super().__init__(
            keys=keys,
            roi_size=-1,
            max_roi_size=None,
            random_center=random_center,
            random_size=random_size,
            allow_missing_keys=allow_missing_keys,
        )
        self.roi_scale = roi_scale
        self.max_roi_scale = max_roi_scale
        self.prob = prob

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        if np.random.uniform() < self.prob:
            first_key: Union[Hashable, List] = self.first_key(data)  # type: ignore
            if first_key == []:
                return data  # type: ignore

            img_size = data[first_key].shape[1:]  # type: ignore
            ndim = len(img_size)
            self.roi_size = [
                ceil(r * s)
                for r, s in zip(ensure_tuple_rep(self.roi_scale, ndim), img_size)
            ]
            if self.max_roi_scale is not None:
                self.max_roi_size = [
                    ceil(r * s)
                    for r, s in zip(
                        ensure_tuple_rep(self.max_roi_scale, ndim), img_size
                    )
                ]
            else:
                self.max_roi_size = None
            return super().__call__(data=data)
        else:
            return data
