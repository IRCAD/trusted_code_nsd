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
import re
import sys
from os.path import join

import cc3d
import nibabel as nib
import numpy as np
import open3d as o3d
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from monai.transforms import AsDiscrete
from skimage import measure

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
    """
    Represents a medical image.

    Attributes:
        path (str): The path to the image file (.nii.gz).
        itkimg (SimpleITK.Image): The SimpleITK image object.
        nibimg (nibabel.Nifti1Image): The NiBabel image object.
        modality (str): The modality of the image (e.g., "US", "CT").
        basename (str): The base name of the image file.
        size (np.ndarray): The size of the image (width, height, depth).
        origin (np.ndarray): The origin of the image in voxel coordinates.
        orientation (np.ndarray): The orientation of the image as a 3x3 matrix.
        spacing (np.ndarray): The voxel spacing of the image.
        nibaffine (np.ndarray): The affine transform of the image.
        nparray (np.ndarray): The NumPy array representation of the image data.
    """

    def __init__(self, imgpath) -> None:
        """
        Initializes an Image object from an image file (.nii.gz).

        Args:
            imgpath (str): The path to the image file (.nii.gz).
        """

        assert imgpath[-7:] == ".nii.gz"

        self.path = imgpath
        self.basename = os.path.basename(self.path)
        self.itkimg = sitk.ReadImage(self.path)
        self.nibimg = nib.load(self.path)

        # Determine the modality of the image
        self.modality = None

        if "US" in self.basename:
            self.modality = "US"
        if "CT" in self.basename:
            self.modality = "CT"

        # Extract image information
        self.size = np.array(self.itkimg.GetSize())
        self.origin = np.array(self.itkimg.GetOrigin())
        self.orientation = np.array(self.itkimg.GetDirection()).reshape((3, 3))
        self.spacing = np.array(self.itkimg.GetSpacing())
        self.nibaffine = self.nibimg.affine
        self.nparray = self.nibimg.get_fdata()

    def setmodality(self, modality):
        assert modality in ["US", "CT"], "trusted modalities are US or CT"
        if self.modality is None:
            self.modality = modality
        else:
            print(
                "The modality ",
                self.modality,
                " has already been found. Check the data used",
            )
        return

    def getmodality(self):
        print("The modality ", self.modality, " has been found.")
        if self.modality is None:
            print("Please set the modality to 'US' or 'CT' with .setmodality(modality)")
        return self.modality

    def setsuffix(self, suffix=None):
        if suffix is None:
            if self.modality is None:
                print(
                    "Please, before, set the modality to 'US' or 'CT' with .setmodality(modality)"
                )
            else:
                suffix = "_img" + self.modality + ".nii.gz"
                self.suffix = suffix
        else:
            self.suffix = suffix
        return

    def resize(self, newsize, interpolmode="trilinear", resized_dirname=None):
        """
        Resizes the image to the specified new size.

        Args:
            newsize (tuple): New size of the image (width, height, depth).
            interpolmode (str): Interpolation mode to use for resizing ("trilinear" is recommended).
            resized_dirname (str, optional): Path to the folder where the resized images are saved (optional).

        Returns:
            np.ndarray: The resized NumPy array representation of the image data.
        """
        print("Image resizing: ", self.basename)
        resized_nparray = F.interpolate(
            torch.unsqueeze(torch.unsqueeze(torch.from_numpy(self.nparray), 0), 0),
            size=newsize,
            mode=interpolmode,
            align_corners=interpolmode == "trilinear",
        )
        resized_nparray = (torch.squeeze(torch.squeeze(resized_nparray, 0), 0)).numpy()

        # Save the resized image if specified
        if resized_dirname is not None:
            resized_nibimg_path = join(resized_dirname, self.basename)
            resized_nibimg = nib.Nifti1Image(resized_nparray, self.nibaffine)
            nib.save(resized_nibimg, resized_nibimg_path)
            print("resized image saved as: ", resized_nibimg_path)

        return resized_nparray

    def shift_origin(self, new_origin=[0, 0, 0], shifted_dirname=None):
        assert self.modality == "CT", "Needed only for CT volume"

        print("CT data origin shifting to ", new_origin, " for ", self.basename)
        self.setsuffix()
        a = re.search(self.suffix, self.basename).start()
        self.individual_name = self.basename[:a]

        itk_nparray = sitk.GetArrayFromImage(self.itkimg)
        itk_ref_img = sitk.GetImageFromArray(itk_nparray)

        itk_ref_img.SetSpacing(self.spacing)
        itk_ref_img.SetDirection(self.orientation.flatten().tolist())
        itk_ref_img.SetOrigin(new_origin)

        tx = sitk.AffineTransform(3)
        tx.SetMatrix(np.eye(3).flatten().tolist())

        if "img" in self.suffix:
            interpolator = sitk.sitkLinear
        elif "mask" in self.suffix:
            interpolator = sitk.sitkNearestNeighbor
        else:
            raise ValueError(
                "The suffix file basename must content 'img' or 'mask'. Use self.setsuffix(suffix)."
            )

        itk_shifed = sitk.Resample(
            itk_ref_img,
            itk_ref_img,
            tx,
            interpolator=interpolator,
            defaultPixelValue=0.0,
        )

        if shifted_dirname is not None:
            itk_shifed_path = join(shifted_dirname, self.basename)
            sitk.WriteImage(itk_shifed, itk_shifed_path)
            print("itk_shifed data saved as: ", itk_shifed_path)

        return itk_shifed


class Mask(Image):
    def __init__(self, imgpath, annotatorID=None):
        super().__init__(imgpath)
        self.annotatorID = annotatorID

    def resize(self, newsize, interpolmode="trilinear", resized_dirname=None):
        """
        Resizes the image to the specified new size.

        Args:
            ...
        Returns:
            ...
        """
        print("Mask resizing: ", self.basename)
        resized_nparray = F.interpolate(
            torch.unsqueeze(torch.unsqueeze(torch.from_numpy(self.nparray), 0), 0),
            size=newsize,
            mode=interpolmode,
            align_corners=interpolmode == "trilinear",
        )
        transform = AsDiscrete(threshold_values=True, logit_thresh=0.5)
        resized_nparray = transform(resized_nparray)
        resized_nparray = (torch.squeeze(torch.squeeze(resized_nparray, 0), 0)).numpy()

        # Save the resized image if specified
        if resized_dirname is not None:
            resized_nibimg_path = join(resized_dirname, self.basename)
            resized_nibimg = nib.Nifti1Image(resized_nparray, self.nibaffine)
            nib.save(resized_nibimg, resized_nibimg_path)
            print("resized mask saved as: ", resized_nibimg_path)

        return resized_nparray

    def setsuffix(self, suffix=None):
        if suffix is None:
            if self.modality is None:
                print(
                    "Please, before, set the modality to 'US' or 'CT' with .setmodality(modality)"
                )
            else:
                suffix = "_mask" + self.modality + ".nii.gz"
                self.suffix = suffix
        else:
            self.suffix = suffix
        print("suffix ", self.suffix, "has been set")
        return

    def to_mesh_and_pcd(
        self,
        mesh_dirname=None,
        pcd_dirname=None,
        ok_with_suffix=False,
        mask_cleaning=True,
    ):
        assert (
            self.modality is not None
        ), "Please set the modality to 'US' or 'CT' with self.setmodality(modality)"

        if not ok_with_suffix:
            self.setsuffix()
            print("NOTE:")
            print(
                "The suffix is the part of the basename (including the annotator ID) after the individual name"
            )
            print("Example of individual name:'01L' for US or '01' for CT ")
            print("Actually the suffix is: ", self.suffix)
            print("Make sure it is what you must have.")
            print("Or, set the appropriate suffix with self.setsuffix(suffix).")
            print(
                "Then set the argument 'ok_with_suffix' to True when running self.tomesh()"
            )
        else:
            self.setsuffix()
            a = re.search(self.suffix, self.basename).start()
            self.individual_name = self.basename[:a]

            affine = self.nibaffine
            new_affine = np.linalg.solve(np.sign(affine), affine)
            sign_affine = np.sign(affine)
            sign_affine[sign_affine == 0.0] = 1.0
            sign_new_affine = np.sign(new_affine)
            sign_new_affine[sign_new_affine == 0.0] = 1.0
            self.mesh_orientation = np.diag(new_affine)[:3] @ np.sign(affine[:3, :3])

            print("*** mesh and pcd creation for: ", self.basename, " ***")
            out = cc3d.connected_components(self.nparray)
            bins_origin = np.bincount(out.flatten())
            bins_copy = np.ndarray.tolist(np.bincount(out.flatten()))
            ind0 = 0
            bins_copy.remove(bins_origin[ind0])
            ind1 = np.where(bins_origin == max(bins_copy))[0][0]
            bins_copy.remove(bins_origin[ind1])

            if self.modality == "CT":
                ind2 = np.where(bins_origin == max(bins_copy))[0][0]
                bins_copy.remove(bins_origin[ind2])
                out1 = out.copy()
                out1[out1 != ind1] = 0
                out2 = out.copy()
                out2[out2 != ind2] = 0
                out1[out1 > 0] = 1
                out2[out2 > 0] = 1
                out_both = out1 + out2
                del out
                (
                    vertexsCT1,
                    faceCT1,
                    normalsCT1,
                    valuesCT1,
                ) = measure.marching_cubes_lewiner(
                    out1, spacing=self.mesh_orientation, step_size=1
                )
                (
                    vertexsCT2,
                    faceCT2,
                    normalsCT2,
                    valuesCT2,
                ) = measure.marching_cubes_lewiner(
                    out2, spacing=self.mesh_orientation, step_size=1
                )

                o3d_CT_1 = o3d.geometry.TriangleMesh()
                o3d_CT_1.triangles = o3d.utility.Vector3iVector(faceCT1)
                o3d_CT_1.vertices = o3d.utility.Vector3dVector(vertexsCT1)

                o3d_CT_2 = o3d.geometry.TriangleMesh()
                o3d_CT_2.triangles = o3d.utility.Vector3iVector(faceCT2)
                o3d_CT_2.vertices = o3d.utility.Vector3dVector(vertexsCT2)

                x_center1 = np.asarray(o3d_CT_1.get_center())[0]
                x_center2 = np.asarray(o3d_CT_2.get_center())[0]

                if x_center1 < x_center2:
                    o3d_meshCT_L = o3d.geometry.TriangleMesh()
                    o3d_meshCT_L.triangles = o3d.utility.Vector3iVector(faceCT1)
                    o3d_meshCT_L.vertices = o3d.utility.Vector3dVector(vertexsCT1)

                    o3d_meshCT_R = o3d.geometry.TriangleMesh()
                    o3d_meshCT_R.triangles = o3d.utility.Vector3iVector(faceCT2)
                    o3d_meshCT_R.vertices = o3d.utility.Vector3dVector(vertexsCT2)

                if x_center1 > x_center2:
                    o3d_meshCT_R = o3d.geometry.TriangleMesh()
                    o3d_meshCT_R.triangles = o3d.utility.Vector3iVector(faceCT1)
                    o3d_meshCT_R.vertices = o3d.utility.Vector3dVector(vertexsCT1)

                    o3d_meshCT_L = o3d.geometry.TriangleMesh()
                    o3d_meshCT_L.triangles = o3d.utility.Vector3iVector(faceCT2)
                    o3d_meshCT_L.vertices = o3d.utility.Vector3dVector(vertexsCT2)

                o3d_pcdCT_L = o3d.geometry.PointCloud()
                o3d_pcdCT_L.points = o3d.utility.Vector3dVector(
                    np.asarray(o3d_meshCT_L.vertices)
                )

                o3d_pcdCT_R = o3d.geometry.PointCloud()
                o3d_pcdCT_R.points = o3d.utility.Vector3dVector(
                    np.asarray(o3d_meshCT_R.vertices)
                )

                if mesh_dirname is not None:
                    if self.annotatorID is None:
                        meshL_path = join(
                            mesh_dirname, self.individual_name + "L" + "meshfaceCT.obj"
                        )
                        meshR_path = join(
                            mesh_dirname, self.individual_name + "R" + "meshfaceCT.obj"
                        )
                    else:
                        meshL_path = join(
                            mesh_dirname,
                            self.individual_name
                            + "L"
                            + self.annotatorID
                            + "meshfaceCT.obj",
                        )
                        meshR_path = join(
                            mesh_dirname,
                            self.individual_name
                            + "R"
                            + self.annotatorID
                            + "meshfaceCT.obj",
                        )

                    o3d.io.write_triangle_mesh(meshL_path, o3d_meshCT_L)
                    print("o3d_meshCT_L saved as: ", meshL_path)
                    o3d.io.write_triangle_mesh(meshR_path, o3d_meshCT_R)
                    print("o3d_meshCT_R saved as: ", meshR_path)

                if pcd_dirname is not None:
                    if self.annotatorID is None:
                        pcdL_path = join(
                            pcd_dirname, self.individual_name + "L" + "pcdCT.txt"
                        )
                        pcdR_path = join(
                            pcd_dirname, self.individual_name + "R" + "pcdCT.txt"
                        )
                    else:
                        pcdL_path = join(
                            pcd_dirname,
                            self.individual_name + "L" + self.annotatorID + "pcdCT.txt",
                        )
                        pcdR_path = join(
                            pcd_dirname,
                            self.individual_name + "R" + self.annotatorID + "pcdCT.txt",
                        )

                    np.savetxt(
                        pcdL_path, np.asarray(o3d_meshCT_L.vertices), delimiter=", "
                    )
                    print("pcdCT_L_txt saved as: ", pcdL_path)
                    np.savetxt(
                        pcdR_path, np.asarray(o3d_meshCT_R.vertices), delimiter=", "
                    )
                    print("pcdCT_R_txt saved as: ", pcdR_path)

                print("case done")
                return o3d_meshCT_L, o3d_meshCT_R, o3d_pcdCT_L, o3d_pcdCT_R

            if self.modality == "US":
                out1 = out.copy()
                out1[out1 != ind1] = 0
                out1[out1 > 0] = 1
                out_both = out1
                del out
                (
                    vertexsUS,
                    faceUS,
                    normalsUS,
                    valuesUS,
                ) = measure.marching_cubes_lewiner(
                    out1, spacing=self.mesh_orientation, step_size=1
                )
                o3d_meshUS = o3d.geometry.TriangleMesh()
                o3d_meshUS.triangles = o3d.utility.Vector3iVector(faceUS)
                o3d_meshUS.vertices = o3d.utility.Vector3dVector(vertexsUS)

                o3d_pcdUS = o3d.geometry.PointCloud()
                o3d_pcdUS.points = o3d.utility.Vector3dVector(
                    np.asarray(o3d_meshUS.vertices)
                )

                if mesh_dirname is not None:
                    if self.annotatorID is None:
                        mesh_path = join(
                            mesh_dirname, self.individual_name + "meshfaceUS.obj"
                        )
                    else:
                        mesh_path = join(
                            mesh_dirname,
                            self.individual_name + self.annotatorID + "meshfaceUS.obj",
                        )

                    o3d.io.write_triangle_mesh(mesh_path, o3d_meshUS)
                    print("o3d_meshUS saved as: ", mesh_path)

                if pcd_dirname is not None:
                    if self.annotatorID is None:
                        pcd_path = join(pcd_dirname, self.individual_name + "pcdUS.txt")
                    else:
                        pcd_path = join(
                            pcd_dirname,
                            self.individual_name + self.annotatorID + "pcdUS.txt",
                        )
                    np.savetxt(
                        pcd_path, np.asarray(o3d_meshUS.vertices), delimiter=", "
                    )
                    print("pcdUS_txt saved as: ", pcd_path)

                print("case done")
                return o3d_meshUS, o3d_pcdUS

            """mask_cleaning"""
            if mask_cleaning:
                mask_cleaned_nib = nib.Nifti1Image(out_both, self.affine)
                nib.save(mask_cleaned_nib, self.path)
                print("cleaning done")

    def split(self, split_dirname=None):
        assert self.modality == "CT", "Applicable only on a CT mask"

        print("*** CT mask splitting for: ", self.basename, " ***")

        self.setsuffix()
        a = re.search(self.suffix, self.basename).start()
        self.individual_name = self.basename[:a]

        len0 = self.nparray.shape[0]
        mid0 = int(len0 / 2)

        nparrayR = np.zeros_like(self.nparray)
        nparrayR[:mid0, :, :] = self.nparray[:mid0, :, :].copy()

        nparrayL = np.zeros_like(self.nparray)
        nparrayL[mid0:, :, :] = self.nparray[mid0:, :, :].copy()

        if split_dirname is not None:
            splitR_path = join(
                split_dirname,
                self.basename.replace(self.individual_name, self.individual_name + "R"),
            )
            splitR_nib = nib.Nifti1Image(nparrayR, self.nibaffine)
            nib.save(splitR_nib, splitR_path)
            print("splitR_nib saved as: ", splitR_path)

            splitL_path = join(
                split_dirname,
                self.basename.replace(self.individual_name, self.individual_name + "L"),
            )
            splitL_nib = nib.Nifti1Image(nparrayL, self.nibaffine)
            nib.save(splitL_nib, splitL_path)
            print("splitL_nib saved as: ", splitL_path)

        return nparrayL, nparrayR


class Landmarks:
    def __init__(self) -> None:
        pass


class Mesh:
    def __init__(self) -> None:
        pass


def fuse_masks(
    list_of_trusted_masks: list[Mask],
    trusted_img: Image,
    resizing=None,
    npmaxflow_lamda=2.5,
):
    print("*** masks fusion with staple+maxflow ***")

    seg_stack = []

    for trusted_mask in list_of_trusted_masks:
        # STAPLE requires we cast into int16 arrays
        # trusted_mask_int16 = sitk.GetImageFromArray(trusted_mask.nparray.astype(np.int16))

        itk_int16 = sitk.Cast(trusted_mask.itkimg, sitk.sitkInt16)
        seg_stack.append(itk_int16)

    # Run STAPLE algorithm
    itk_STAPLE_prob = sitk.STAPLE(seg_stack, 1.0)  # 1.0 specifies the foreground value

    # convert back to numpy array
    nparray_itkprob = sitk.GetArrayFromImage(itk_STAPLE_prob)
    nparray_itkimg = sitk.GetArrayFromImage(trusted_img.itkimg)

    nparray_itkprob = np.asarray(nparray_itkprob, np.float32)
    nparray_itkimg = np.asarray(nparray_itkimg, np.float32)

    if resizing is not None:
        resized_nparray_itkprob = F.interpolate(
            torch.unsqueeze(torch.unsqueeze(torch.from_numpy(nparray_itkprob), 0), 0),
            size=resizing,
            mode="nearest",
        )
        resized_nparray_itkimg = F.interpolate(
            torch.unsqueeze(torch.unsqueeze(torch.from_numpy(nparray_itkimg), 0), 0),
            size=resizing,
            mode="trilinear",
            align_corners=True,
        )

        nparray_itkprob = (
            torch.squeeze(torch.squeeze(resized_nparray_itkprob, 0), 0)
        ).numpy()
        nparray_itkimg = (
            torch.squeeze(torch.squeeze(resized_nparray_itkimg, 0), 0)
        ).numpy()

    nparray_itkprob = np.asarray(nparray_itkprob, np.float32)
    nparray_itkimg = np.asarray(nparray_itkimg, np.float32)

    print(nparray_itkprob.shape)
    print(nparray_itkimg.shape)

    return


def fuse_landmarks():
    return


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
