import argparse
import os

import yaml
from tools import set_nnunet_dir


def parse_args():
    PARSER = argparse.ArgumentParser(description="Launch nnUnet training")
    PARSER.add_argument(
        "--config_path", type=str, required=True, help="path to the parameters yml file"
    )
    ARGS = PARSER.parse_args()
    return ARGS


def main():
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        data = yaml.safe_load(yaml_file)

    modality = data["modality"]
    data_location = data["data_location"]
    modality_specific = os.path.join(modality.upper() + "_DATA", data["nnunet_data"])
    root_dir = os.path.join(data_location, modality_specific, "processed/Dataset")

    set_nnunet_dir(root_dir)
    command0 = (
        "nnUNetv2_train " + str(data["dataset_id"]) + " 3d_fullres" + " 0" + " --npz"
    )
    command1 = (
        "nnUNetv2_train " + str(data["dataset_id"]) + " 3d_fullres" + " 1" + " --npz"
    )
    command2 = (
        "nnUNetv2_train " + str(data["dataset_id"]) + " 3d_fullres" + " 2" + " --npz"
    )
    command3 = (
        "nnUNetv2_train " + str(data["dataset_id"]) + " 3d_fullres" + " 3" + " --npz"
    )
    command4 = (
        "nnUNetv2_train " + str(data["dataset_id"]) + " 3d_fullres" + " 4" + " --npz"
    )

    os.system(command0)
    os.system(command1)
    os.system(command2)
    os.system(command3)
    os.system(command4)
    return


if __name__ == "__main__":
    main()
