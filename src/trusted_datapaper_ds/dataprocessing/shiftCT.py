"""
Here you can see some examples (NOT EXHAUSTIVE) of
running common data processings you could have to do with a list of individuals the TRUSTED dataset.
Based on them, you could run those you want.

IMPORTANT: You could adapt the config_file.yml file

# Example of command to run the tutorial ####
# python src/trusted_datapaper_ds/dataprocessing/tutorial_for_list.py --config_path configs/config_file.yml

"""
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
    # Shift the origin of list of images or masks (here CT images) ###
    # Note: "shifted_dirname" is the directory to save the shifted data.
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
