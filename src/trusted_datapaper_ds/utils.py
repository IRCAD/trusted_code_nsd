import argparse


def parse_args():
    PARSER = argparse.ArgumentParser(description="Launch nnUnet training")
    PARSER.add_argument(
        "--config_path", type=str, required=True, help="path to the parameters yml file"
    )
    ARGS = PARSER.parse_args()
    return ARGS
