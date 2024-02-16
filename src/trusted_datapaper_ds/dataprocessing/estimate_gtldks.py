from os.path import join

import yaml
from natsort import natsorted

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.utils import makedir, parse_args


def main(
    config,
    ctlist,
    uslist,
    fuse_CTlandmark,
    fuse_USlandmark,
):
    """
    Fuses landmark sets for CT and US images based on configuration settings.

    Args:
        config (dict): Configuration dictionary containing paths and parameters.
        ctlist (List[str]): List of individual IDs for CT images.
        uslist (List[str]): List of individual IDs for US images.
        fuse_CTlandmark (bool): Flag indicating whether to fuse CT landmarks.
        fuse_USlandmark (bool): Flag indicating whether to fuse US landmarks.

    Returns:
        None
    """
    if fuse_CTlandmark:
        fused_dirname = config["myCT_fusedlandmarks_location"]
        makedir(fused_dirname)
        for ind in ctlist:
            individual = ind
            print("Processing ", ind)
            for k_side in ["L", "R"]:
                ldk1path = join(
                    config["data_location"],
                    config["CTld1fol"],  # or config["CTld" + "1" + "fol"]
                    individual + k_side + "1" + config["CTld_end"],
                )
                ldk2path = join(
                    config["data_location"],
                    config["CTld2fol"],  # or config["CTld" + "2" + "fol"]
                    individual + k_side + "2" + config["CTld_end"],
                )
                ldks1 = dt.Landmarks(ldk1path, annotatorID="1")
                ldks2 = dt.Landmarks(ldk2path, annotatorID="2")
                list_of_ldks = [ldks1, ldks2]
                dt.fuse_landmarks(
                    list_of_trusted_ldks=list_of_ldks,
                    fused_dirname=fused_dirname,
                )

    # Fuse list of landmark set from 1 and 2 (here U landmarks) ###
    # Note: "fused_dirname" is the directory to save the fused mask.
    if fuse_USlandmark:
        fused_dirname = config["myUS_fusedlandmarks_location"]
        makedir(fused_dirname)
        for ind in uslist:
            print("Processing ", ind)
            k_side = ind[-1]
            individual = ind[:-1]
            ldk1path = join(
                config["data_location"],
                config["USld1fol"],  # or config["USld" + "1" + "fol"]
                individual + k_side + "1" + config["USld_end"],
            )
            ldk2path = join(
                config["data_location"],
                config["USld2fol"],  # or config["USld" + "2" + "fol"]
                individual + k_side + "2" + config["USld_end"],
            )
            ldks1 = dt.Landmarks(ldk1path, annotatorID="1")
            ldks2 = dt.Landmarks(ldk2path, annotatorID="2")
            list_of_ldks = [ldks1, ldks2]
            dt.fuse_landmarks(
                list_of_trusted_ldks=list_of_ldks,
                fused_dirname=fused_dirname,
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
        fuse_CTlandmark=config["fuse_CTlandmark"],
        fuse_USlandmark=config["fuse_USlandmark"],
    )
