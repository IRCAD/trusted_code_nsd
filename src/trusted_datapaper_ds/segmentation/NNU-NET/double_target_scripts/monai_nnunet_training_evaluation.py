import os

if (
    os.path.join(os.path.normpath(os.getcwd())).split("/")[-1]
    != "spatiotemporal-us-organ-segmentation"
):
    temp_folder = os.path.normpath(os.getcwd() + os.sep + os.pardir)
else:
    temp_folder = os.path.normpath(os.getcwd())
root_dir = os.path.join(temp_folder, "data/processed/nnunet")
data_root = os.path.join(root_dir, "Task02_Kidney")
output_path_yaml = os.path.join(root_dir, "input.yaml")
command = (
    "python -m monai.apps.nnunet nnUNetV2Runner run --input_config=" + output_path_yaml
)
os.system(command)
