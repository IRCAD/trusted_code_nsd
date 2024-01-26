import yaml
from nnUNet_tools import exp_plan_and_preprocess

from trusted_datapaper_ds.utils import parse_args

if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # training_folder.create(config)
    # convert_dataset.convert(config)
    exp_plan_and_preprocess.run(args)
