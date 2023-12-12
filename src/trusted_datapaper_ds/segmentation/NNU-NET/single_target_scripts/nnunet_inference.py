import argparse
import os
import time

import yaml


def parse_args():
    PARSER = argparse.ArgumentParser(description="Convert dataset into nnUnet format")
    PARSER.add_argument(
        "--config_path",
        type=str,
        required=False,
        help="path to the parameters yml file",
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
    raw_data_dir = os.path.join(root_dir, "nnUNet_raw")
    specific_dataset_dir = "Dataset%03.0d_%s" % (data["dataset_id"], data["task_name"])
    pred_folder = os.path.join(root_dir, "Inference_Kidney_output")

    if not os.path.exists(pred_folder):
        os.makedirs(pred_folder)
    input_inference = os.path.join(raw_data_dir, specific_dataset_dir, "imagesTs")
    if not os.path.exists(input_inference):
        os.makedirs(input_inference)

    os.environ["nnUNet_raw"] = os.path.join(root_dir, "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = os.path.join(root_dir, "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = os.path.join(root_dir, "nnUNet_results")

    start_time = time.time()
    command = (
        "nnUNetv2_predict -i "
        + input_inference
        + " -o "
        + pred_folder
        + " -d"
        + specific_dataset_dir
        + " -c 3d_fullres -f 0 -device cpu -npp 1 -nps 1"
    )
    os.system(command)
    end_time = time.time()

    single_running_time = (end_time - start_time) / 12

    print("single_running_time: ", single_running_time)

    # command = ("nnUNetv2_predict -h ")
    # os.system(command)

    return


if __name__ == "__main__":
    args = parse_args()
    main()
