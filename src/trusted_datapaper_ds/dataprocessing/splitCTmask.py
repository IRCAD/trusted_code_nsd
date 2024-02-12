from os.path import join

import yaml
from natsort import natsorted

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.utils import build_many_mask_analist, makedir, parse_args


def main(
    config,
    ctlist,
    splitCTmask,
):
    if splitCTmask:
        ann = config["annotator_splitCTmask"]

        split_dirname = join(config["myDATA"], config["CTspma" + ann + "fol"])
        makedir(split_dirname)

        USlike_IDlist = None
        CTlike_IDlist = ctlist

        ctma1_files, ctma2_files, ctmagt_files = build_many_mask_analist(
            "CT", config, USlike_IDlist, CTlike_IDlist
        )
        if ann == "1":
            ctmask_files = ctma1_files
        elif ann == "2":
            ctmask_files = ctma2_files
        elif ann == "gt":
            ctmask_files = ctmagt_files

        for ctmask_file in ctmask_files:
            ctmask = dt.Mask(ctmask_file, annotatorID=ann)
            nibL, nibR = ctmask.split(split_dirname=split_dirname)

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

    ctlist = allct

    main(config, ctlist, splitCTmask=bool(config["splitCTmask"]))
