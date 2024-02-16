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

### The goal of the python project
This is the script developed to analyse the TRUSTED dataset, as well as to evaluate some baselines of deep-learning-based segmentations models, and registration models.
Here is how to use it for "reproductibility".
Each point is organized as:
- config file: the configuration file to set before running the specific commands.
   The main fields you will have to update, are the paths. I recommend to not change them too much
- config variables to set: the fields to check in the configuration file before running the specific commands
- command: the specific command to run


# Installation of the python project

### 1. Clone the repo:
   Commands:

   ```
   git clone https://git.ircad.fr/wndzimbong/trusted_datapaper_ds.git

   cd trusted_datapaper_ds
   ```

### 2. Environment setting:
   Commands:

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

   pip install seaborn==0.12.2

   pip install -e .

   ```

# Data checking
   - Download the data folder "TRUSTED_submission"
   - Check the folder structure which is given in the file README.txt in the folder "TRUSTED_submission"

# Data processing and analysis
This step describes the operations to run to estimate the ground-truth (GT) annotations, to quantitatively compare the estimated GT and the annotations provided by human experts, and to prepare the data used for baselines evaluation.

### 3. Notes:
   - config file: configs/anaconfig.yml
   - config variables to set: data_location
   - Check that the structure of the folder "TRUSTED_submission" and the convension naming used matches with the one used in "configs/anaconfig.yml" for the "PROVIDED" subfolders
   - In out python scripts, an annotator is represented by the values: "1" for annotator 1, "2" for annotator 2, "gt" for ground-truth segmentations, and "auto" for automatic segmentations


### 4. To estimate the ground-truth masks
   - config file: configs/anaconfig.yml
   - config variables to set: myUS_fusedmasks_location, myCT_fusedmasks_location, fuse_USmask, fuse_CTmask
   - command:
   ```
   python src/trusted_datapaper_ds/dataprocessing/estimate_gtmasks.py --config_path configs/anaconfig.yml
   ```
   Note: in the file "src/trusted_datapaper_ds/dataprocessing/estimate_gtmasks.py", line 89, there are some resizing option parameters, to avoid memory overload. I choose by default [512, 384, 384] just for the US data which are quite big. Depending to your memory, you can set different values.


### 5. To estimate the ground-truth landmarks
   - config file: configs/anaconfig.yml
   - config variables to set: myUS_fusedlandmarks_location, myCT_fusedlandmarks_location, fuse_USlandmark, fuse_CTlandmark
   - command:
   ```
   python src/trusted_datapaper_ds/dataprocessing/estimate_gtldks.py --config_path configs/anaconfig.yml
   ```

   **Note:** To obtain CT landmarks in the CT device coordinate systems (US landmarks are in the US device coordinate system)
   - config file: configs/anaconfig.yml
   - config variables to set: CTldks_over_images, annotator_mov_ldks
   - command:
   ```
   python src/trusted_datapaper_ds/dataprocessing/put_landmarks_in_image_space.py --config_path configs/anaconfig.yml
   ```

### 6. To convert masks to meshes
   - config file: configs/anaconfig.yml
   - config variables to set: myDATA, CTmask_to_mesh, USmask_to_mesh, annotator_mask_to_mesh
   - command:
   ```
   python src/trusted_datapaper_ds/dataprocessing/convert_mask_to_mesh.py --config_path configs/anaconfig.yml
   ```
   **Note:** To obtain CT Meshes in the CT or US device coordinate systems
   - config file: configs/anaconfig.yml
   - config variables to set: CTmesh_over_images, USmesh_over_images, annotator_mov_mesh
   - command:
   ```
   python src/trusted_datapaper_ds/dataprocessing/put_meshes_in_image_space.py --config_path configs/anaconfig.yml
   ```


### 7. To split CT masks
   - config file: configs/anaconfig.yml
   - config variables to set: myDATA, splitCTmask, annotator_splitCTmask
   - command:
   ```
   python src/trusted_datapaper_ds/dataprocessing/splitCTmask.py --config_path configs/anaconfig.yml
   ```

### 8. To compare ground-truth estimated masks with annotator segmentations
   - config file: configs/anaconfig.yml
   - config variables to set: US_analysis_folder, CT_analysis_folder, usdata_eval and ctdata_eval
   - command:
   ```
   python src/trusted_datapaper_ds/dataprocessing/groundtruth_eval.py --config_path configs/anaconfig.yml
   ```

### 9. To compute the statistical summary of the comparison in 8-
   - config file: configs/anaconfig.yml
   - config variables to set: US_analysis_folder, CT_analysis_folder, usdata_analysis and ctdata_analysis
   - command:
   ```
   python src/trusted_datapaper_ds/dataprocessing/dataanalysis.py --config_path configs/anaconfig.yml
   ```

### 10. Data resizing
   - config file: configs/anaconfig.yml
   - config variables to set: usdata_resize, ctdata_resize, newsize, annotator_dataresize
   - command:
   ```
   python src/trusted_datapaper_ds/dataprocessing/data_resizing.py --config_path configs/anaconfig.yml
   ```

# Automatic segmentation

## Segmentation processes in CT data

### 11. 3D UNet or VNet models training
   - config file: configs/ctsegconfig.yml
   - config variables to set:
      - data_location, the path to the folder "TRUSTED_submission"
      - img_location, the path to the folder containing the original CT images
      - output_location, the main folder where you want to save your different training outputs (models weight, training graphs, ...), depending to the modality, the models, ... etc
   - Note: the learning rate, weight_decay, number of epochs are fixed in the file "src/trusted_datapaer_ds/segmentation/U_or_V_Net_training.py" (line 82-84)
   - command:
   ```
   python src/trusted_datapaper_ds/segmentation/U_or_V_Net_training.py  --config_path configs/ctsegconfig.yml
   ```

### 12. 3D UNet or VNet inference
   - config file: configs/ctsegconfig.yml
   - config variables to set:
      - trained_models_location, where your training folders are located
      - list_UVnetmodels and list_training_target, depending on what you want to infer
      - mask_seglocation, where the outputs masks in the original size are save by model and training target
   - command:
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
   corresponding to what you want.
   - config file: src/trusted_datapaper_ds/segmentation/nnunet_cotr/configs/training/training_128_jz_v2.yaml
   - config variables to set:
      - only_val and checkpoint.load to False, to launch a training
      - pth, to set the training results location (Remember the change it if you change the data modality)
   - **Important note about only_val and checkpoint.load:**
      - when only_val==False and checkpoint.load==False , a new training is launched
      - when only_val==False and checkpoint.load==True , the last training continues using the "latest.pt", if it exists
      - when only_val==True, the evaluation in run using the "best.pt" if it exists
   - command:
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
   - config file: /src/trusted_datapaper_ds/segmentation/nnunet_cotr/configs/training/training_128_jz_v2.yaml
   - config variables to set: only_val and checkpoint.load to True
   - command: same command as for training

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
Apply different post-processings (upsampling, meshing, splitCT) to the masks obtain in inference.
   - config file: configs/ctsegconfig.yml
   - config variables to set: ist_othermodels, list_training_target, upsampling, meshing, splitCT
   - command:
   ```
   python src/trusted_datapaper_ds/segmentation/autoseg_postprocess.py  --config_path configs/ctsegconfig.yml
   ```

### 16. Segmentation evaluation
   - config file: configs/ctsegconfig.yml
   - config variables to set: segresults_folder, list_othermodels, list_UVnetmodels, list_training_target
   - command:
   ```
   python src/trusted_datapaper_ds/segmentation/segmentation_evaluation.py  --config_path configs/ctsegconfig.yml
   ```

### 17. Segmentation results statistical analysis
   - config file: configs/seganalysis.yml
   - config variables to set: modality, refmodel, reftarget, list_segmodels, segresults_folder
   - command:
   ```
   python src/trusted_datapaper_ds/segmentation/segmentation_analysis.py  --config_path configs/seganalysis.yml
   ```

## Segmentation processes in US data

**Follow the same steps with the corresponding command and config file obtained by changing "CT" to "US" or "ct" to "us"**



# Registration

### 18. Landmarks registration (global)
   - config file: configs/regconfig.yml
   - config variables to set: USldks_location, CTldks_location, ldks_model, transfo_location, noise_std, iternumb
   - command:
   ```
   python src/trusted_datapaper_ds/registration/landmarks_registration.py  --config_path configs/regconfig.yml
   ```
   It produces the global (or initial) registration matrices

### 19. Landmarks registration (global) + Surface-based registration (refinement)
   - config file: configs/regconfig.yml
   - config variables to set: regmethod, ldks_model, USldks_location, CTldks_location, noise_std, iternumb, refine_model, USautomesh_location, CTautomesh_location, transfo_location, BCPD_temp_folder, regpack_dir
   - command:
   ```
   python src/trusted_datapaper_ds/registration/surface_registration.py  --config_path configs/regconfig.yml
   ```
   It produces the refinement registration matrices computed with surface-based methods

### 20. Landmarks registration (global) + Intensity-based registration (refinement)

   **Install ImFusion Suite (You can use a trial version)**
      - Get the installer file "BaseImFusionSuite-2.44.1-22.04.deb" and
      the License key(s) for Base ImFusion Suite, from ImFusion,
      - Open a terminal window and navigate to the directory containing the .deb file
      - run the command: sudo apt-get install -f
      - run the installation command: sudo dpkg --install BaseImFusionSuite-2.44.1-22.04.deb
      - launch the ImFusionSuite interface with the command: ImFusionSuite
      - activate your License by entering your License key
      - now you can use ImFusionSuite by launching the interface (command: ImFusionSuite), of with our python script provided

   **Running**
      - config file: configs/regconfig.yml
      - config variables to set: similarity_metric, ldks_model, USldks_location, CTldks_location, noise_std, iternumb, refine_model, USimg_location, CTimg_location, transfo_location, regpack_dir, imf_temp_folder
      - command:
      ```
      python src/trusted_datapaper_ds/registration/intensity_registration.py  --config_path configs/regconfig.yml
      ```
      It produces the refinement registration matrices computed with intensity-based methods


### 21. Shift origin of CT images
   I explained above that the CT image origins are not [0,0,0], while CT mesh origins are always [0,0,0] because of the Marching Cubes implementation used to create them, which does not keep their initial location (it just keeps their orientation). So, to have the CT images (particularly) and their corresponding meshes, point clouds and landmarks, in the same space, then to be able to transfert the estimated transforms by landmarks or point clouds registration to the images, and vice-versa, one way is to also applied this localization (origin) shifting to the CT images, so translate them from their original origin to the [0,0,0] origin in the RAS coordinates system. This is what I describe here.
   **Note:**  Another way to have CT images and meshes (landmarks and point clouds) in the same coordinates system is to keep the CT images in their initial location, and apply the specific transform located in "/home/TRUSTED_dataset_for_submission/CT_DATA/CT_tback_transforms/", to the US moving data AFTER the registration (first apply the registration transform, second apply the shiftback transform).

   For CT origin shifting:
   - config file: configs/anaconfig.yml
   - config variables to set: shiftCTimg_origin, myDATA, data_location, CTimg_origin0_location
   - command:
   ```
   python src/trusted_datapaper_ds/dataprocessing/shiftCT.py  --config_path configs/anaconfig.yml
   ```

### 22. Registration evaluation
   - config file: configs/regconfig.yml
   - config variables to set: CTimg_origin0_location, refinement_methods, transform_models, std_cases, transfo_location, CTgtmesh_location, USgtmesh_location, USldks_location, CTldks_location
   - command:
   ```
   python src/trusted_datapaper_ds/registration/registration_evaluation.py  --config_path configs/regconfig.yml
   ```

### 23. Registration results statistical analysis
   - config file: configs/reganalysis.yml
   - config variables to set: refmethod, reftransform, list_regmethods, list_regtransforms, regresults_folder
   - command:
   ```
    python src/trusted_datapaper_ds/registration/registration_analysis.py  --config_path configs/reganalysis.yml
   ```

### 24. Boxplots of registration results statistical analysis

   - config file: configs/reganalysis.yml
   - config variables to set: regresults_folder, list_regmethods, list_regtransforms, list_std
   - command:
   ```
   python src/trusted_datapaper_ds/registration/analysis_of_noising_on_registration.py  --config_path configs/reganalysis.yml
   ```





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
