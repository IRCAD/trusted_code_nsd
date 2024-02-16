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
    """
    Splits CT segmentation masks.

    This function reads a list of individual IDs for CT images (`ctlist`). If the
    `splitCTmask` flag is True and an annotator is specified in the configuration,
    it performs the following:

    1. Identifies mask files for the specified annotator (`annotator_splitCTmask`)
    2. Creates a directory to store the split masks.
    3. Splits each mask into left and right kidney segmentations.
    4. Saves the split masks in the designated directory with appropriate naming.

    Args:
        config (dict): Configuration dictionary containing paths and parameters.
        ctlist (List[str]): List of individual IDs for CT images.
        splitCTmask (bool): Flag indicating whether to split CT masks.

    Returns:
        None
    """

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
        config["CTfoldmask"]["cv1"]
        + config["CTfoldmask"]["cv2"]
        + config["CTfoldmask"]["cv3"]
        + config["CTfoldmask"]["cv4"]
        + config["CTfoldmask"]["cv5"]
    )

    ctlist = allct

    main(config, ctlist, splitCTmask=bool(config["splitCTmask"]))
