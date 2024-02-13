from os.path import join

import nibabel as nib
import yaml
from natsort import natsorted

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.utils import makedir, parse_args


def main(
    config,
    ctlist,
    uslist,
    usdata_resize,
    ctdata_resize,
):
    ann = config["annotator_dataresize"]

    if ann == "gt" or ann == "auto":
        word = ""
    else:
        word = ann

    if usdata_resize:
        print("RESIZING US")
        img_resized_dirname = join(config["myDATA"], config["US128imgfol"])
        makedir(img_resized_dirname)
        mask_resized_dirname = join(config["myDATA"], config["US128ma" + ann + "fol"])
        makedir(mask_resized_dirname)

        for ind in uslist:
            print("processing: ", ind)
            k_side = ind[-1]
            individual = ind[:-1]
            imgpath = join(
                config["data_location"],
                config["USimgfol"],
                individual + k_side + config["USimg_end"],
            )
            maskpath = join(
                config["data_location"],
                config["USma" + ann + "fol"],
                individual + k_side + word + config["USma_end"],
            )
            newsize = [int(size) for size in config["newsize"]]

            img = dt.Image(imgpath)
            mask = dt.Mask(maskpath, annotatorID=ann)

            resized_nibimg = dt.resiz_nib_data(
                img.nibimg, newsize, interpolmode="trilinear", binary=False
            )
            resized_img_path = join(img_resized_dirname, img.basename)
            nib.save(resized_nibimg, resized_img_path)

            resized_nibmask = dt.resiz_nib_data(
                mask.nibmask, newsize, interpolmode="trilinear", binary=True
            )
            resized_mask_path = join(mask_resized_dirname, mask.basename)
            nib.save(resized_nibmask, resized_mask_path)

    if ctdata_resize:
        print("RESIZING CT")
        img_resized_dirname = join(config["myDATA"], config["CT128imgfol"])
        makedir(img_resized_dirname)
        mask_resized_dirname = join(config["myDATA"], config["CT128ma" + ann + "fol"])
        makedir(mask_resized_dirname)

        for ind in ctlist:
            print("processing: ", ind)

            individual = ind
            imgpath = join(
                config["data_location"],
                config["CTimgfol"],
                individual + config["CTimg_end"],
            )
            maskpath = join(
                config["data_location"],
                config["CTma" + ann + "fol"],
                individual + "_" + word + config["CTma_end"],
            )
            newsize = [int(size) for size in config["newsize"]]

            img = dt.Image(imgpath)
            mask = dt.Mask(maskpath, annotatorID=ann)

            resized_nibimg = dt.resiz_nib_data(
                img.nibimg, newsize, interpolmode="trilinear", binary=False
            )
            resized_img_path = join(img_resized_dirname, img.basename)
            nib.save(resized_nibimg, resized_img_path)

            resized_nibmask = dt.resiz_nib_data(
                mask.nibmask, newsize, interpolmode="trilinear", binary=True
            )
            resized_mask_path = join(mask_resized_dirname, mask.basename)
            nib.save(resized_nibmask, resized_mask_path)

    return


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    allct = natsorted(
        config["CTfoldmask"]["cv1"]
        + config["CTfoldmask"]["cv2"]
        + config["CTfoldmask"]["cv3"]
        + config["CTfoldmask"]["cv4"]
        + config["CTfoldmask"]["cv5"]
    )

    allus = natsorted(
        config["USfold"]["cv1"]
        + config["USfold"]["cv2"]
        + config["USfold"]["cv3"]
        + config["USfold"]["cv4"]
        + config["USfold"]["cv5"]
    )

    ctlist = allct
    uslist = allus

    resizing = 0

    main(
        config,
        ctlist,
        uslist,
        usdata_resize=bool(config["usdata_resize"]),
        ctdata_resize=bool(config["ctdata_resize"]),
    )
