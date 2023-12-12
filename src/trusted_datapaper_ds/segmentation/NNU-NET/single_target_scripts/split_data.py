import argparse
import json
import os

import yaml
from natsort import natsorted


def parse_args():
    PARSER = argparse.ArgumentParser(description="create five fold split JSON file")
    PARSER.add_argument(
        "--config_path", type=str, required=True, help="path to the parameters yml file"
    )
    ARGS = PARSER.parse_args()
    return ARGS


def create_five_fold(img_file_list, us_cv):
    """
    Function that create five fold

    Args:
       -img_file_list (list): list to  all images
       -us_cv: image in the fold to be used as valid set.
    Return:
       fold: fold created

    """
    us_cv = ["kidney_" + name for name in us_cv]
    train = []
    val = []
    for i, img_file in enumerate(img_file_list):
        if img_file[:-12] not in us_cv:
            train.append(img_file[:-12])
        else:
            val.append(img_file[:-12])

    fold = {"train": train, "val": val}
    return fold


def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    """
     Function that save dataset JSON file

    Args:
         obj: object to be saved
         file (str): filename of the JSON is going to be saved.
         indent (int): indent needed.
         sort_keys (bool): choosing to sort object or not
    """
    with open(file, "w") as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


def main():
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        data = yaml.safe_load(yaml_file)

    modality = data["modality"]
    data_location = data["data_location"]
    modality_specific = os.path.join(modality.upper() + "_DATA", data["nnunet_data"])
    root_dir = os.path.join(data_location, modality_specific, "processed/Dataset")
    raw_data_dir = os.path.join(root_dir, "nnUNet_raw")
    spefic_dataset_dir = "Dataset%03.0d_%s" % (data["dataset_id"], data["task_name"])
    dataset_dir = os.path.join(raw_data_dir, spefic_dataset_dir)
    img_dir = os.path.join(dataset_dir, "imagesTr")

    json_dir = os.path.join(
        data_location,
        modality_specific,
        "processed",
        "Dataset",
        "nnUNet_preprocessed",
        spefic_dataset_dir,
    )
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    json_filename = os.path.join(json_dir, "splits_final.json")

    img_file_list = natsorted(os.listdir(img_dir))
    fold1 = create_five_fold(img_file_list, data["fold"]["cv1"])
    fold2 = create_five_fold(img_file_list, data["fold"]["cv2"])
    fold3 = create_five_fold(img_file_list, data["fold"]["cv3"])
    fold4 = create_five_fold(img_file_list, data["fold"]["cv4"])
    fold5 = create_five_fold(img_file_list, data["fold"]["cv5"])
    json_data = [fold1, fold2, fold3, fold4, fold5]

    save_json(json_data, json_filename)
    print("cross validation jsons successfully created into :", json_dir)
    return


if __name__ == "__main__":
    main()
