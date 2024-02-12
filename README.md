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

## Installation

1. Clone the repo:
   ```
   git clone https://git.ircad.fr/wndzimbong/trusted_datapaper_ds.git
   cd trusted_datapaper_ds
   ```

2. Environment setting:
   ```
   conda create -n trusted_env python=3.9

   conda activate trusted_env

   pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

   pip install monai[all]==0.9.0
   pip install pandas==1.3.0
   pip install lxml
   pip install connected-components-3d==3.12.4
   pip install statsmodels==0.14.1
   pip install natsort==8.4.0
   pip install vtk==9.3.0
   pip install jupyterlab (lastest version)
   pip install open3d==0.15.2
   pip install opencv-contrib-python==4.9.0.80
   pip install plyfile==1.0.3
   pip install numpymaxflow==0.0.6

   pip install -e .

   ```

3. Note about the naming convension

   An annotator could be: "1", "2" or "gt" for data analysis, and: "1", "2", "gt" or "auto" for the other modules

4. Estimate the ground-truth masks

   In the file configs/anaconfig.yml, set the values of:
   myUS_fusedmasks_location, myCT_fusedmasks_location, fuse_USmask, fuse_CTmask, as you want, and run the command
   ```
   python src/trusted_datapaper_ds/dataprocessing/estimate_gtmasks.py --config_path configs/anaconfig.yml
   ```
   Note: in the file "src/trusted_datapaper_ds/dataprocessing/estimate_gtmasks.py", line 89, there are some resizing option parameters, to avoid memory overload. I choose by default [384, 256, 256] just for the US data which are quite big. Depending to your memory, you can set different values.


5. Estimate the ground-truth landmarks

   In the file configs/anaconfig.yml, set the values of:
   myUS_fusedlandmarks_location, myCT_fusedlandmarks_location, fuse_USlandmark, fuse_CTlandmark, as you want, and run the command
   ```
   python src/trusted_datapaper_ds/dataprocessing/estimate_gtldks.py --config_path configs/anaconfig.yml
   ```

6. Convert masks to meshes

   In the file configs/anaconfig.yml, set the values of:
   myDATA, CTmask_to_mesh, USmask_to_mesh, annotator_mask_to_mesh, as you want, and run the command
   ```
   python src/trusted_datapaper_ds/dataprocessing/convert_mask_to_mesh.py --config_path configs/anaconfig.yml
   ```


7. Split CT masks

   In the file configs/anaconfig.yml, set the values of:
   myDATA, splitCTmask, annotator_splitCTmask, as you want, and run the command
   ```
   python src/trusted_datapaper_ds/dataprocessing/splitCTmask.py --config_path configs/anaconfig.yml
   ```


8. Ground-truth comparison with annotators

   In the file configs/anaconfig.yml, set the values of:
   US_analysis_folder, CT_analysis_folder, usdata_analysis and ctdata_analysis, as you want, and run the command

   ```
   python src/trusted_datapaper_ds/dataprocessing/groundtruth_eval.py --config_path configs/anaconfig.yml
   ```

9. Ground-truth statistical analysis
   ```

   ```


10.
   ```

   ```


11.
   ```

   ```



4. install [nbstripout] git hooks to remove the output cells of committed notebooks with:
   ```bash
   nbstripout --install --attributes notebooks/.gitattributes
   ```
   This is useful to avoid large diffs due to plots in your notebooks.
   A simple `nbstripout --uninstall` will revert these changes.


Then take a look into the `scripts` and `notebooks` folders.

## Dependency Management & Reproducibility

1. Always keep your abstract (unpinned) dependencies updated in `environment.yml` and eventually
   in `setup.cfg` if you want to ship and install your package via `pip` later on.
2. Create concrete dependencies as `environment.lock.yml` for the exact reproduction of your
   environment with:
   ```bash
   conda env export -n trusted_datapaper_ds -f environment.lock.yml
   ```
   For multi-OS development, consider using `--no-builds` during the export.
3. Update your current environment with respect to a new `environment.lock.yml` using:
   ```bash
   conda env update -f environment.lock.yml --prune
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
