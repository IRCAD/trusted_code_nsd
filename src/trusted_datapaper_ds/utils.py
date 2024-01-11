import argparse
import os


def parse_args():
    PARSER = argparse.ArgumentParser(description="")
    PARSER.add_argument(
        "--config_path", type=str, required=True, help="path to the parameters yml file"
    )
    ARGS = PARSER.parse_args()
    return ARGS


def makedir(folder):
    try:
        os.makedirs(folder)
        print("Directory ", folder, " Created ")
    except FileExistsError:
        print("Directory ", folder, " already exists")
    return
