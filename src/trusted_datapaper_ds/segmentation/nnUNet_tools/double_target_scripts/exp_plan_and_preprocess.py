import argparse
import os

import yaml
from tools import set_nnunet_dir


def parse_args():
    PARSER = argparse.ArgumentParser(description="Convert dataset into nnUnet format")
    PARSER.add_argument(
        "--config_path", type=str, required=True, help="path to the parameters yml file"
    )
    PARSER.add_argument(
        "--num_processors",
        type=int,
        default=2,
        help="number of processors used \
                       for preprocessing",
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

    os.environ["nnUNet_def_n_proc"] = str(args.num_processors)
    os.environ["nnUNet_n_proc_DA"] = str(args.num_processors)
    command = (
        "nnUNetv2_plan_and_preprocess -np "
        + str(args.num_processors)
        + " -d "
        + str(data["dataset_id"])
        + " --verify_dataset_integrity"
    )
    os.system(command)
    return


if __name__ == "__main__":
    main()
