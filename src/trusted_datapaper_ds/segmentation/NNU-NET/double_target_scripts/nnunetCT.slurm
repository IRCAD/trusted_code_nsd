#!/bin/bash
#SBATCH -p long
#SBATCH -k
#SBATCH --job-name="nnunet_seg_2tg_CT_kidney128"
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=32
#SBATCH --time 7-00:00:00

source /opt/server-env.sh

# Display info about the current commit (reproducibility)
echo "git: $(git log -1 --format='%H - %cd - %an - %s')"

# Activate conda base environement
conda activate nnunet
echo "Using conda env: $CONDA_DEFAULT_ENV"

python convert_dataset.py --config_path configs/segCT_config_file.yml
python exp_plan_and_preprocess.py --config_path configs/segCT_config_file.yml
python split_data.py --config_path configs/segCT_config_file.yml
python train_script.py --config_path configs/segCT_config_file.yml

# python nnunet_inference.py --config_path configs/segCT_config_file.yml
