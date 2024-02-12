from os.path import join

import yaml
from natsort import natsorted

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.utils import makedir, parse_args


def main(
    config,
    ctlist,
    uslist,
    fuse_USmask,
    fuse_CTmask,
):
    if fuse_CTmask:
        fused_dirname = config["myCT_fusedmasks_location"]
        makedir(fused_dirname)
        for ind in ctlist:
            individual = ind
            print("PROCESSING: ", ind)
            imgpath = join(
                config["data_location"],
                config["CTimgfol"],
                individual + config["CTimg_end"],
            )
            mask1path = join(
                config["data_location"],
                config["CTma" + config["annotator1"] + "fol"],  # config["CTma1fol"],
                individual + "_" + config["annotator1"] + config["CTma_end"],
            )
            mask2path = join(
                config["data_location"],
                config["CTma2fol"],  # or config["CTma" + config["annotator2"] + "fol"],
                individual + "_" + config["annotator2"] + config["CTma_end"],
            )
            img = dt.Image(imgpath)
            mask1 = dt.Mask(mask1path, annotatorID=config["annotator1"])
            mask2 = dt.Mask(mask2path, annotatorID=config["annotator2"])
            list_of_masks = [mask1, mask2]

            dt.fuse_masks(
                list_of_trusted_masks=list_of_masks,
                trusted_img=img,
                npmaxflow_lamda=2.5,
                img_intensity_scaling="normal",  # "normal" or "scale"
                resizing=None,  # I reduce the data shape to increase the speed of the process. Can be None
                fused_dirname=fused_dirname,
            )

    # Fuse list of masks from annotator1 and annotator2 (here US masks) ###
    # Note: "fused_dirname" is the directory to save the fused mask.
    if fuse_USmask:
        fused_dirname = config["myUS_fusedmasks_location"]
        makedir(fused_dirname)
        for ind in uslist:
            print("PROCESSING: ", ind)
            k_side = ind[-1]
            individual = ind[:-1]
            imgpath = join(
                config["data_location"],
                config["USimgfol"],
                individual + k_side + config["USimg_end"],
            )
            mask1path = join(
                config["data_location"],
                config["USma1fol"],  # or config["USma" + config["annotator1"] + "fol"]
                individual + k_side + config["annotator1"] + config["USma_end"],
            )
            mask2path = join(
                config["data_location"],
                config["USma2fol"],  # or config["USma" + config["annotator2"] + "fol"]
                individual + k_side + config["annotator2"] + config["USma_end"],
            )
            img = dt.Image(imgpath)
            mask1 = dt.Mask(mask1path, annotatorID=config["annotator1"])
            mask2 = dt.Mask(mask2path, annotatorID=config["annotator2"])
            list_of_masks = [mask1, mask2]

            dt.fuse_masks(
                list_of_trusted_masks=list_of_masks,
                trusted_img=img,
                npmaxflow_lamda=2.5,
                img_intensity_scaling="normal",  # "normal" or "scale"
                resizing=[
                    384,
                    256,
                    256,
                ],  # I reduce the data shape to increase the speed of the process. Can be None
                fused_dirname=fused_dirname,
            )

    return


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    allct = natsorted(
        config["CTfold"]["cv1"]
        + config["CTfold"]["cv2"]
        + config["CTfold"]["cv3"]
        + config["CTfold"]["cv4"]
        + config["CTfold"]["cv5"]
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

    main(
        config,
        ctlist,
        uslist,
        fuse_CTmask=bool(config["fuse_CTmask"]),
        fuse_USmask=bool(config["fuse_USmask"]),
    )
