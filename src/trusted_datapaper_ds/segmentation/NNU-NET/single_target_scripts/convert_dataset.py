import argparse
import json
import os
import shutil
from typing import Tuple

import yaml
from natsort import natsorted


def parse_args():
    PARSER = argparse.ArgumentParser(description="Convert dataset into nnUnet format")
    PARSER.add_argument(
        "--config_path", type=str, required=True, help="path to the parameters yml file"
    )

    ARGS = PARSER.parse_args()
    return ARGS


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


def generate_dataset_json(
    output_folder: str,
    channel_names: dict,
    labels: dict,
    num_training_cases: int,
    file_ending: str,
    regions_class_order: Tuple[int, ...] = None,
    dataset_name: str = None,
    reference: str = None,
    release: str = None,
    license: str = None,
    description: str = None,
    overwrite_image_reader_writer: str = None,
    **kwargs
):
    """
    Generates a dataset.json file in the output folder

     Argument:
           -channel_names:
                  Channel names must map the index to the name of the channel, example:
                    {
                        0: 'T1',
                        1: 'CT'
                        }
                Note that the channel names may influence the normalization scheme!! Learn more in the documentation.

            -labels:
                This will tell nnU-Net what labels to expect. Important: This will also determine whether you use
                region-based training or not.
                Example regular labels:
                {
                    'background': 0,
                    'left atrium': 1,
                    'some other label': 2
                }
               Example region-based training:
               {
                'background': 0,
                'whole tumor': (1, 2, 3),
                'tumor core': (2, 3),
                'enhancing tumor': 3
                }
                Remember that nnU-Net expects consecutive values for labels! nnU-Net also expects 0 to be background!

            -num_training_cases: is used to double check all cases are there!
            -file_ending: needed for finding the files correctly. IMPORTANT! File endings must match between
               images and segmentations!

            -dataset_name, reference, release, license, description: self-explanatory and not used by nnU-Net. Just for
                completeness and as a reminder that these would be great!

            - overwrite_image_reader_writer: If you need a special IO class for your dataset you can derive it from
                BaseReaderWriter, place it into nnunet.imageio and reference it here by name

            -kwargs: whatever you put here will be placed in the dataset.json as well

    """
    has_regions: bool = any(
        [isinstance(i, (tuple, list)) and len(i) > 1 for i in labels.values()]
    )
    if has_regions:
        assert regions_class_order is not None, (
            "You have defined regions but regions_class_order is not set. "
            "You need that."
        )
    # channel names need strings as keys
    keys = list(channel_names.keys())
    for k in keys:
        if not isinstance(k, str):
            channel_names[str(k)] = channel_names[k]
            del channel_names[k]

    # labels need ints as values
    for lab in labels.keys():
        value = labels[lab]
        if isinstance(value, (tuple, list)):
            value = tuple([int(i) for i in value])
            labels[lab] = value
        else:
            labels[lab] = int(labels[lab])

    dataset_json = {
        "channel_names": channel_names,  # previously this was called 'modality'. I didnt like this so this is
        # channel_names now. Live with it.
        "labels": labels,
        "numTraining": num_training_cases,
        "file_ending": file_ending,
    }

    if dataset_name is not None:
        dataset_json["name"] = dataset_name
    if reference is not None:
        dataset_json["reference"] = reference
    if release is not None:
        dataset_json["release"] = release
    if license is not None:
        dataset_json["licence"] = license
    if description is not None:
        dataset_json["description"] = description
    if overwrite_image_reader_writer is not None:
        dataset_json["overwrite_image_reader_writer"] = overwrite_image_reader_writer
    if regions_class_order is not None:
        dataset_json["regions_class_order"] = regions_class_order

    dataset_json.update(kwargs)

    save_json(
        dataset_json, os.path.join(output_folder, "dataset.json"), sort_keys=False
    )


def main(config_path, modality):
    with open(config_path, "r") as yaml_file:
        data = yaml.safe_load(yaml_file)
    data_location = data["data_location"]
    modality_specific = os.path.join(modality.upper() + "_DATA", data["nnunet_data"])
    root_dir = os.path.join(data_location, modality_specific, "processed/Dataset")
    raw_data_dir = os.path.join(root_dir, "nnUNet_raw")
    preprocessed_data_dir = os.path.join(root_dir, "nnUNet_preprocessed")
    results_dir = os.path.join(root_dir, "nnUNet_results")
    spefic_dataset_dir = "Dataset%03.0d_%s" % (data["dataset_id"], data["task_name"])
    dataset_dir = os.path.join(raw_data_dir, spefic_dataset_dir)
    train_img_dir = os.path.join(dataset_dir, "imagesTr")
    test_img_dir = os.path.join(dataset_dir, "imagesTs")
    train_labels = os.path.join(dataset_dir, "labelsTr")
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(preprocessed_data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_labels, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    img_dir = os.path.join(
        data_location, modality.upper() + "_DATA", data["nnunet_data"], "raw", "input"
    )
    img_files_list = natsorted(os.listdir(img_dir))
    mask_dir = os.path.join(
        data_location,
        modality.upper() + "_DATA",
        data["nnunet_data"],
        "raw",
        "ground_truth",
    )
    mask_files_list = natsorted(os.listdir(mask_dir))

    for i, (img_file, mask_file) in enumerate(zip(img_files_list, mask_files_list)):
        new_img_filename = (
            data["img_prefix"]
            + img_file[: img_file.find(data["file_ending"])]
            + "_"
            + data["channel"]
            + data["file_ending"]
        )
        new_mask_filename = (
            data["img_prefix"]
            + mask_file[: mask_file.find(data["file_ending"])]
            + data["file_ending"]
        )
        shutil.copyfile(
            os.path.join(img_dir, img_file),
            os.path.join(train_img_dir, new_img_filename),
        )
        shutil.copyfile(
            os.path.join(mask_dir, mask_file),
            os.path.join(train_labels, new_mask_filename),
        )

    generate_dataset_json(
        dataset_dir,
        data["channel_names"],
        labels=data["labels"],
        num_training_cases=len(img_files_list),
        file_ending=data["file_ending"],
        dataset_name=data["task_name"],
        reference=data["reference"],
        overwrite_image_reader_writer=data["image_reader_writer"],
        description=data["description"],
    )

    return root_dir


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        data = yaml.safe_load(yaml_file)
    modality = data["modality"]

    root_dir = main(args.config_path, modality=modality)
    print("Data successfully preprocessed into: ", root_dir)
