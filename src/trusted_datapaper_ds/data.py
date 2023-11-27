"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
``[options.entry_points]`` section in ``setup.cfg``::

    console_scripts =
         fibonacci = trusted.skeleton:run

Then run ``pip install .`` (or ``pip install -e .`` for editable mode)
which will install the command ``fibonacci`` inside your current environment.

Besides console scripts, the header (i.e. until ``_logger``...) of this file can
also be used as template for Python modules.

Note:
    This file can be renamed depending on your needs or safely removed if not needed.

References:
    - https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    - https://pip.pypa.io/en/stable/reference/pip_install
"""

import argparse
import logging
import os
import sys

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from trusted_datapaper_ds import __version__

__author__ = "William NDZIMBONG"
__copyright__ = "William NDZIMBONG"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


# ---- Python API ----
# The functions defined in this section can be imported by users in their
# Python scripts/interactive interpreter, e.g. via
# `from trusted.skeleton import loadimg`,
# when using this Python module as a library.


class Image:
    def __init__(self, imgpath) -> None:
        """load a .nii.gz file and set its basic useful infos
        Args:
        imgpath (str): .nii.gz image file path

        Returns:
        Image object: SimpleITK.Image
        """
        assert imgpath[-7:] == ".nii.gz"

        self.path = imgpath
        self.itkimg = sitk.ReadImage(self.path)
        self.nibimg = nib.load(self.path)
        self.modality = None
        self.basename = os.path.basename(self.path)
        if "US" in self.basename:
            self.modality = "US"
        if "CT" in self.basename:
            self.modality = "CT"
        self.size = np.array(self.itkimg.GetSize())
        self.origin = np.array(self.itkimg.GetOrigin())
        self.orientation = np.array(self.itkimg.GetDirection()).reshape((3, 3))
        self.spacing = np.array(self.itkimg.GetSpacing())
        self.nibaffine = self.nibimg.affine
        self.nparray = self.nibimg.get_fdata()

    def resize(self, newsize, interpolmode, resized_nibimg_path=None):
        assert interpolmode == "trilinear"

        resized_nparray = F.interpolate(
            torch.unsqueeze(torch.unsqueeze(torch.from_numpy(self.nparray), 0), 0),
            size=newsize,
            mode=interpolmode,
            align_corners=True,
        )
        resized_nparray = (torch.squeeze(torch.squeeze(resized_nparray, 0), 0)).numpy()

        if resized_nibimg_path is not None:
            resized_nibimg = nib.Nifti1Image(resized_nparray, self.nibaffine)
            nib.save(resized_nibimg, resized_nibimg_path)

        return resized_nparray


class Mask(Image):
    def resize(self, newsize, interpolmode, resized_nibimg_path=None):
        assert interpolmode == "nearest"

        resized_nparray = F.interpolate(
            torch.unsqueeze(torch.unsqueeze(torch.from_numpy(self.nparray), 0), 0),
            size=newsize,
            mode=interpolmode,
            align_corners=False,
        )
        resized_nparray = (torch.squeeze(torch.squeeze(resized_nparray, 0), 0)).numpy()

        if resized_nibimg_path is not None:
            resized_nibimg = nib.Nifti1Image(resized_nparray, self.nibaffine)
            nib.save(resized_nibimg, resized_nibimg_path)

        return resized_nparray

    def tomesh():
        return


class Landmarks:
    def __init__(self) -> None:
        pass


# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Just a Fibonacci demonstration")
    parser.add_argument(
        "--version",
        action="version",
        version=f"trusted_datapaper_ds {__version__}",
    )
    parser.add_argument(dest="n", help="n-th Fibonacci number", type=int, metavar="INT")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


# def main(args):
#     """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

#     Instead of returning the value from :func:`fib`, it prints the result to the
#     ``stdout`` in a nicely formatted message.

#     Args:
#       args (List[str]): command line parameters as list of strings
#           (for example  ``["--verbose", "42"]``).
#     """
#     args = parse_args(args)
#     setup_logging(args.loglevel)
#     _logger.debug("Starting crazy calculations...")
#     print(f"The {args.n}-th Fibonacci number is {fib(args.n)}")
#     _logger.info("Script ends here")


def main(args):
    return


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::

    # python -m trusted_datapaper_ds.data 42

    run()
