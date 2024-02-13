[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
<!-- These are examples of badges you might also want to add to your README. Update the URLs accordingly.
[![Built Status](https://api.cirrus-ci.com/github/<USER>/trusted_datapaper_ds.svg?branch=main)](https://cirrus-ci.com/github/<USER>/trusted_datapaper_ds)
[![ReadTheDocs](https://readthedocs.org/projects/trusted_datapaper_ds/badge/?version=latest)](https://trusted_datapaper_ds.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/trusted_datapaper_ds/main.svg)](https://coveralls.io/r/<USER>/trusted_datapaper_ds)
[![PyPI-Server](https://img.shields.io/pypi/v/trusted_datapaper_ds.svg)](https://pypi.org/project/trusted_datapaper_ds/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/trusted_datapaper_ds.svg)](https://anaconda.org/conda-forge/trusted_datapaper_ds)
[![Monthly Downloads](https://pepy.tech/badge/trusted_datapaper_ds/month)](https://pepy.tech/project/trusted_datapaper_ds)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/trusted_datapaper_ds)
-->

# trusted_datapaper_ds

> Add a short description here!

A longer description of your project goes here...

# Installation

### 1. Clone the repo:
   ```
   git clone https://git.ircad.fr/wndzimbong/trusted_datapaper_ds.git

   cd trusted_datapaper_ds
   ```

### 2. Environment setting:
   ```
   conda create -n trusted_env python=3.9

   conda activate trusted_env

   pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

   pip install monai[all]==0.9.0

   pip install SimpleITK==2.3.0

   pip install pandas==1.3.0

   pip install lxml

   pip install connected-components-3d==3.12.4

   pip install statsmodels==0.14.1

   pip install natsort==8.4.0

   pip install vtk==9.1.0

   pip install jupyterlab

   pip install open3d==0.15.2

   pip install opencv-contrib-python==4.9.0.80

   pip install plyfile==1.0.3

   pip install numpymaxflow==0.0.6

   pip install hydra-core==1.3.1

   pip install telegram-send==0.35

   pip install batchgenerators==0.23

   pip install nnunet==1.7.1

   pip install -e .

   ```

# Data checking
   - Download the data folder "TRUSTED_submission"
   - Check the folder structure which is given in the file README.txt in the folder "TRUSTED_submission"

# Data processing and analysis

### 3. Notes:

   - The config file used i this section is "configs/anaconfig.yml"
   - In this file, set "data_location" properly
   - Check that the structure of the folder "TRUSTED_submission" and the convension naming used matches with the one used in "configs/anaconfig.yml" for the "PROVIDED" subfolders
   - In out python scripts, an annotator is represented by the values: "1" for annotator 1, "2" for annotator 2, "gt" for ground-truth segmentations, and "auto" for automatic segmentations


### 4. To estimate the ground-truth masks

   In the file configs/anaconfig.yml, set the values of:
   myUS_fusedmasks_location, myCT_fusedmasks_location, fuse_USmask, fuse_CTmask, as you want, and run the command
   ```
   python src/trusted_datapaper_ds/dataprocessing/estimate_gtmasks.py --config_path configs/anaconfig.yml
   ```
   Note: in the file "src/trusted_datapaper_ds/dataprocessing/estimate_gtmasks.py", line 89, there are some resizing option parameters, to avoid memory overload. I choose by default [512, 384, 384] just for the US data which are quite big. Depending to your memory, you can set different values.


### 5. To estimate the ground-truth landmarks

   In the file configs/anaconfig.yml, set the values of:
   myUS_fusedlandmarks_location, myCT_fusedlandmarks_location, fuse_USlandmark, fuse_CTlandmark, as you want, and run the command
   ```
   python src/trusted_datapaper_ds/dataprocessing/estimate_gtldks.py --config_path configs/anaconfig.yml
   ```

### 6. To convert masks to meshes

   In the file configs/anaconfig.yml, set the values of:
   myDATA, CTmask_to_mesh, USmask_to_mesh, annotator_mask_to_mesh, as you want, and run the command
   ```
   python src/trusted_datapaper_ds/dataprocessing/convert_mask_to_mesh.py --config_path configs/anaconfig.yml
   ```


### 7. To split CT masks

   In the file configs/anaconfig.yml, set the values of:
   myDATA, splitCTmask, annotator_splitCTmask, as you want, and run the command
   ```
   python src/trusted_datapaper_ds/dataprocessing/splitCTmask.py --config_path configs/anaconfig.yml
   ```


### 8. To compare ground-truth estimated masks with annotator segmentations

   In the file configs/anaconfig.yml, set the values of:
   US_analysis_folder, CT_analysis_folder, usdata_eval and ctdata_eval, as you want, and run the command

   ```
   python src/trusted_datapaper_ds/dataprocessing/groundtruth_eval.py --config_path configs/anaconfig.yml
   ```

### 9. To compute the statistical summary of the comparison in 8-

   In the file configs/anaconfig.yml, set the values of:
   US_analysis_folder, CT_analysis_folder, usdata_analysis and ctdata_analysis, as you want, and run the command

   ```
   python src/trusted_datapaper_ds/dataprocessing/dataanalysis.py --config_path configs/anaconfig.yml
   ```

### 10. Data resizing
   In the file configs/anaconfig.yml, set the values of:
   usdata_resize, ctdata_resize, newsize, annotator_dataresize, as you want, and run the command

   ```
   python src/trusted_datapaper_ds/dataprocessing/data_resizing.py --config_path configs/anaconfig.yml
   ```

# Automatic segmentation

## Processes in CT data

### 11. 3D UNet or VNet models training

   In the file configs/ctsegconfig.yml, set properly the values of:
   - data_location, the path to the folder "TRUSTED_submission"
   - img_location, the path to the folder containing the original CT images
   - output_location, the main folder where you want to save your different training outputs (models weight, training graphs, ...), depending to the modality, the models, ... etc
   The other meanings are given as comments in that .yml file
   - The learning rate, weight_decay, number of epochs are fixed in the file "src/trusted_datapaer_ds/segmentation/U_or_V_Net_training.py" (line 82-84)

   Run the command:

   ```
   python src/trusted_datapaper_ds/segmentation/U_or_V_Net_training.py  --config_path configs/ctsegconfig.yml
   ```

### 12. 3D UNet or VNet inference

   In the file configs/ctsegconfig.yml, set properly the values of:
   - trained_models_location, where your training folders are located
   - list_UVnetmodels and list_training_target, depending on what you want to infer
   - mask_seglocation, where the outputs masks in the original size are save by model and training target

   Run the command:
   ```
   python src/trusted_datapaper_ds/segmentation/U_or_V_Net_inference.py  --config_path configs/ctsegconfig.yml
   ```
   The outputs of these inferences have their original sizes

### 13. nnUNet or CoTr models training

   - Manually create a specific folder to train the model: ~/MedSeg/CT_DATA
   - Create 4 subfolders in ~/MedSeg/CT_DATA, named:
      CTUSimg_128_t, CTmask_a1_128, CTmask_a2_128 and CTmask_mf_128,
      containing respectively the resized images, resized masks from annotator1, resized masks from annotator2 and resized ground-truth masks
   - In the folder src/trusted_datapaper_ds/segmentation/nnunet_cotr/configs/dataset, there is four .yml to set properly depending on what you want to train. Particularly, set the values of:
      - path.pth
      - cv
   corresponding to what you want
   - In the folder src/trusted_datapaper_ds/segmentation/nnunet_cotr/configs/training, there is a single file "training_128_jz_v2.yaml" to set properly depending on how you want to train. Particularly, set the values of:
      - only_val and checkpoint.load to False, to launch a training
      - pth, to set the training results location (Remember the change it if you change the data modality)
   - **Important note about only_val and checkpoint.load:**
      - when only_val==False and checkpoint.load==False , a new training is launched
      - when only_val==False and checkpoint.load==True , the last training continues using the "latest.pt", if it exists
      - when only_val==True, the evaluation in run using the "best.pt" if it exists


   Run the command:
   ```
   cd src/trusted_datapaper_ds/segmentation/nnunet_cotr
   ```

   To launch the training of nnunet with single training target, with testing fold cv1, run the command:
   ```
   python mainV2.py -m model=nnunet dataset=ct_128_simple_jz_v2 training=training_128_jz_v2 dataset.cv=cv1
   ```

   To launch the training of nnunet with double training target, with testing fold cv1, run the command:
   ```
   python mainDoubleV2.py -m model=nnunet dataset=ct_128_double_jz_v2 training=training_128_jz_v2 dataset.cv=cv1
   ```

   In the command above, use: mainV2.py or mainDoubleV2.py, the model's value nnunet or cotr, and dataset's value ct_128_simple_jz_v2 or ct_128_double_jz_v2 depending on what you need


### 14. nnUNet or CoTr models inference

   For inference of these models, in the file "/src/trusted_datapaper_ds/segmentation/nnunet_cotr/configs/training/training_128_jz_v2.yaml", set the values of only_val and checkpoint.load to True and run the same command as for training.
   The segmentation masks with the size 128-128-128 will be located into the folder: "~/MedSeg/CT_DATA/medseg_results/~".
   For evaluation, you should move them into a folder corresponding to "seg128location" into "src/trusted_datapaper_ds/segmentation/configs/ctsegconfig.yml" and organized in the following structure:

   ```
   ├── CT_DATA/
   │   ├── CToutput_mask128s/
   │   │   ├── nnunet/
   │   │   │   ├── single_target_training/
   │   │   │   │   ├── 01_maskCT.nii.gz
   │   │   │   │   ├── 02_maskCT.nii.gz
   │   │   │   │   ├── ...
   │   │   │   ├── double_targets_training/
   │   │   │   │   ├── 01_maskCT.nii.gz
   │   │   │   │   ├── 02_maskCT.nii.gz
   │   │   │   │   ├── ...
   │   │   ├── cotr/
   │   │   │   ├── single_target_training/
   │   │   │   │   ├── 01_maskCT.nii.gz
   │   │   │   │   ├── 02_maskCT.nii.gz
   │   │   │   │   ├── ...
   │   │   │   ├── double_targets_training/
   │   │   │   │   ├── 01_maskCT.nii.gz
   │   │   │   │   ├── 02_maskCT.nii.gz
   │   │   │   │   ├── ...
   ```
   The outputs of these inferences have the size [128,128,128], and will be upsampled in the next step

**Note: After finishing with training or/and inference with nnunet and cotr, go back project folder**
   ```
   cd ../../../..
   ```
   The current dir must be the project folder


### 15. Automatic segmentation post-processing
Apply different post-processings (upsampling, meshing, splitCT) to the masks obtain in inference. Set the values of list_othermodels, list_training_target, upsampling, meshing, splitCT  in the file configs/ctsegconfig.yml, depending on what you have and want to post-process.
Then run the command:
   ```
   python src/trusted_datapaper_ds/segmentation/autoseg_postprocess.py  --config_path configs/ctsegconfig.yml
   ```

### 16. Segmentation evaluation
Set the values of segresults_folder, list_othermodels, list_UVnetmodels, list_training_target in the file configs/ctsegconfig.yml, depending on what you have and want to post-process.
Then run the command:
   ```
   python src/trusted_datapaper_ds/segmentation/segmentation_evaluation.py  --config_path configs/ctsegconfig.yml
   ```

### 17. Segmentation statistical analysis
Set the values of modality, refmodel, reftarget, list_segmodels, segresults_folder in thefile configs/seganalysis.yml, and run:

   ```
   python src/trusted_datapaper_ds/segmentation/segmentation_analysis.py  --config_path configs/seganalysis.yml
   ```


# Registration





## Project Organization

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── CONTRIBUTING.md         <- Guidelines for contributing to this project.
├── Dockerfile              <- Build a docker container with `docker build .`.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── pyproject.toml          <- Build configuration. Don't change! Use `pip install -e .`
│                              to install for development or to build `tox -e build`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- [DEPRECATED] Use `python setup.py develop` to install for
│                              development or `python setup.py bdist_wheel` to build.
├── src
│   └── trusted_datapaper_ds <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `pytest`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using [PyScaffold] 4.5 and the [dsproject extension] 0.7.2.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[PyScaffold]: https://pyscaffold.org/
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
