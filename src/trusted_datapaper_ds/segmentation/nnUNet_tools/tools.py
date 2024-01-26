import os


def set_nnunet_dir(root_dir):
    """
    Function that set nnUNet required directories
    Argument:
        -root_dir(str): root directory of these nnUNet directories"
    """
    raw_data_dir = os.path.join(root_dir, "nnUNet_raw")
    preprocessed_data_dir = os.path.join(root_dir, "nnUNet_preprocessed")
    results_dir = os.path.join(root_dir, "nnUNet_results")
    nnUNet_raw = raw_data_dir
    nnUNet_preprocessed = preprocessed_data_dir
    nnUNet_results = results_dir
    os.environ["nnUNet_raw"] = nnUNet_raw
    os.environ["nnUNet_preprocessed"] = nnUNet_preprocessed
    os.environ["nnUNet_results"] = nnUNet_results
