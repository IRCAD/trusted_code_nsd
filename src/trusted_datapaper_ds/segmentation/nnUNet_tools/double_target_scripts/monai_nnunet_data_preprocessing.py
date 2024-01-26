import argparse
import os

from spatiotemporal_us_organ_segmentation.tools import monai_nnunet_data_preprocessing


def parse_args(root_dir, msd_task, num_crossfold, dataset_name):
    """
    Parse command-line arguments for data preprocessing using MONAI and nnUNet.

    Args:
        root_dir (str): Root directory for the data preprocessing.
        msd_task (str): Name of the msd task.
        num_crossfold: Number of cross-fold for training.
        dataset_name: Dataset name used for training MONAI nnunet.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    PARSER = argparse.ArgumentParser(description="monai nnunet data preprocessing")
    PARSER.add_argument(
        "--root_dir",
        type=str,
        help="Root directory for reference for nnunet output",
        default=root_dir,
    )
    PARSER.add_argument(
        "--msd_task", type=str, help="Name of the msd task", default=msd_task
    )
    PARSER.add_argument(
        "--number_folds", type=int, help="Number of cross-folds", default=num_crossfold
    )
    PARSER.add_argument(
        "--dataset_id", type=int, help="dataset_name for nnunet", default=dataset_name
    )

    ARGS = PARSER.parse_args()

    return ARGS


if __name__ == "__main__":
    if (
        os.path.join(os.path.normpath(os.getcwd())).split("/")[-1]
        != "spatiotemporal-us-organ-segmentation"
    ):
        temp_folder = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    else:
        temp_folder = os.path.normpath(os.getcwd())
    root_dir = os.path.join(temp_folder, "data/processed/nnunet")
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    data_root = os.path.join(root_dir, "Task02_Kidney")
    msd_task = "Task02_Kidney"
    num_crossfold = 5
    dataset_id = 2

    ARGS = parse_args(root_dir, msd_task, num_crossfold, dataset_id)
    monai_nnunet_data_preprocessing(
        root_dir=ARGS.root_dir,
        msd_task=ARGS.msd_task,
        num_crossfold=ARGS.number_folds,
        dataset_name=ARGS.dataset_id,
    )
