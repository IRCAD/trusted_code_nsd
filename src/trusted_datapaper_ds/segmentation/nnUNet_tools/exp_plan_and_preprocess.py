import os

import yaml
from nnUNet_tools.tools import set_nnunet_dir

from trusted_datapaper_ds.utils import parse_args


def run(args):
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    data_location = config["trained_models_location"]
    assert config["ntarget"] in ["1", "2"], ' "ntarget" must be  in ["1", "2"]'
    if config["ntarget"] == "1":
        target = "single"
    if config["ntarget"] == "2":
        target = "double"

    root_dir = os.path.join(
        data_location, config["nnunet_data"], "processed", target, "Dataset"
    )

    set_nnunet_dir(root_dir)

    os.environ["nnUNet_def_n_proc"] = str(args.num_processors)
    os.environ["nnUNet_n_proc_DA"] = str(args.num_processors)
    command = (
        "nnUNet_plan_and_preprocess -np "
        + str(args.num_processors)
        + " -d "
        + str(config["dataset_id"])
        + " --verify_dataset_integrity"
    )
    os.system(command)
    return
