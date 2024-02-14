from os.path import join

import yaml
from natsort import natsorted

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.utils import makedir, parse_args


def main(
    config,
    ctlist,
    shiftCTimg_origin,
):
    """
    Shifts the origin of CT images.

    This function reads a list of individual IDs for CT images (`ctlist`). If the
    `shiftCTimg_origin` flag is True, it iterates through each ID and:

    1. Loads the corresponding CT image using the provided configuration paths.
    2. Shifts the image origin to the specified location ([0,0,0]).
    3. Saves the shifted image in a designated directory provided in the configuration.

    Args:
        config (dict): Configuration dictionary containing paths and parameters.
        ctlist (List[str]): List of individual IDs for CT images.
        shiftCTimg_origin (bool): Flag indicating whether to shift the origin of CT images.

    Returns:
        None
    """

    if shiftCTimg_origin:
        shifted_dirname = join(config["myDATA"], config["CT0imgfol"])
        makedir(shifted_dirname)
        for ind in ctlist:
            individual = ind
            imgpath = join(
                config["data_location"],
                config["CTimgfol"],
                individual + config["CTimg_end"],
            )
            ctimg = dt.Image(imgpath)
            ctimg.shift_origin(shifted_dirname=shifted_dirname)
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

    main(
        config,
        ctlist,
        shiftCTimg_origin=config["shiftCTimg_origin"],
    )
