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
    This function execute CT image origin shifting operations.
    It handles the following tasks:

    1. Checks if CT image origin shifting is enabled based on the `shiftCTimg_origin` flag.
    2. Creates necessary directories to store shifted images and transformation matrices for shifting back.
    3. Iterates through a list of CT images.
    4. For each CT image:
        a. Loads the image using the `Image` class.
        b. Shifts the origin of the image using the `shift_origin` method.
        c. Optionally saves the shifted image and transformation matrixfor shifting back.

    Args:
    config (dict): A configuration dictionary containing relevant paths and settings.
    ctlist (list): A list of individual identifiers for the CT images to be processed.
    shiftCTimg_origin (bool): A flag indicating whether to perform CT image origin shifting.

    Returns:
    None
    """
    if shiftCTimg_origin:
        shifted_dirname = join(config["myDATA"], config["CT0imgfol"])
        makedir(shifted_dirname)
        shiftback_mesh_transforms_dirname = join(
            config["myDATA"], config["CT0tbackmeshfol"]
        )
        makedir(shiftback_mesh_transforms_dirname)
        shiftback_ldks_transforms_dirname = join(
            config["myDATA"], config["CT0tbackldkfol"]
        )
        makedir(shiftback_ldks_transforms_dirname)
        for ind in ctlist:
            individual = ind
            imgpath = join(
                config["data_location"],
                config["CTimgfol"],
                individual + config["CTimg_end"],
            )
            ctimg = dt.Image(imgpath)
            ctimg.shift_origin(
                shifted_dirname=shifted_dirname,
                shiftback_mesh_transforms_dirname=shiftback_mesh_transforms_dirname,
                shiftback_ldks_transforms_dirname=shiftback_ldks_transforms_dirname,
            )
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
