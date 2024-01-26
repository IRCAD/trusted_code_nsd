import argparse
import os


def parse_args():
    PARSER = argparse.ArgumentParser(
        description="Run nnUNet integrated within MONAI entire pipeline"
    )
    PARSER.add_argument(
        "--root_dir",
        type=str,
        default=os.path.join(os.getcwd(), "monai_ct_dataset"),
        help="path to directory containing input yaml file",
    )

    ARGS = PARSER.parse_args()
    return ARGS


if __name__ == "__main__":
    args = parse_args()
    input_path_yaml = os.path.join(args.root_dir, "input.yaml")
    command = (
        "python -m monai.apps.nnunet nnUNetV2Runner run --input_config="
        + input_path_yaml
    )
    os.system(command)
