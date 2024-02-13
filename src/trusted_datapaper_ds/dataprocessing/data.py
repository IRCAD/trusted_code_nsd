"""
This file defines the TRUSTED data processing main classes and methods:
    - Image
    - Mask
    - Landmarks
    - Mesh

    - fuse_masks()
    - fuse_landmarks()
    - plot_arrays()
    - resiz_nparray()
"""

import os
import re
from os.path import join

import cc3d
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import numpymaxflow as maxflow
import open3d as o3d
import SimpleITK as sitk
import torch
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureType,
    NormalizeIntensity,
    Resize,
    ScaleIntensity,
)
from skimage import measure

__author__ = "William NDZIMBONG"
__copyright__ = "William NDZIMBONG"
__license__ = "MIT"

# np.random.seed(42)


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

        self.modality = "CT" if "CT" in self.basename else "US"

        self.suffix = f"_img{self.modality}.nii.gz"

        a = re.search(self.suffix, self.basename).start()
        self.individual_name = self.basename[:a]

        # Extract image information
        self.size = np.array(self.itkimg.GetSize())
        self.origin = np.array(self.itkimg.GetOrigin())
        self.orientation = np.array(self.itkimg.GetDirection()).reshape((3, 3))
        self.spacing = np.array(self.itkimg.GetSpacing())
        self.nibaffine = self.nibimg.affine
        self.nparray = self.nibimg.get_fdata()

    def shift_origin(self, new_origin=[0, 0, 0], shifted_dirname=None):
        assert self.modality == "CT", "Needed only for CT volume"

        print("CT data origin shifting to ", new_origin, " for ", self.basename)

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
            raise ValueError("The suffix file basename must content 'img' or 'mask'.")

        itk_shifted = sitk.Resample(
            itk_ref_img,
            itk_ref_img,
            tx,
            interpolator=interpolator,
            defaultPixelValue=0.0,
        )

        if shifted_dirname is not None:
            itk_shifted_path = join(shifted_dirname, self.basename)
            sitk.WriteImage(itk_shifted, itk_shifted_path)
            print("itk_shifted data saved as: ", itk_shifted_path)

        return itk_shifted


class Mask:
    def __init__(self, maskpath, annotatorID, split=False):
        """
        Initializes an Mask object from an mask file (.nii.gz).

        Args:
            maskpath (str): The pathmask to the mask file (.nii.gz).
        """

        assert maskpath[-7:] == ".nii.gz"

        self.path = maskpath
        self.basename = os.path.basename(self.path)
        self.itkmask = sitk.ReadImage(self.path)
        self.nibmask = nib.load(self.path)
        self.modality = "CT" if "CT" in self.basename else "US"

        self.suffix = "_mask" + self.modality + ".nii.gz"

        if annotatorID == "gt" or annotatorID == "auto":
            self.annotatorID = ""
            b = self.suffix
        else:
            self.annotatorID = str(annotatorID)
            if self.modality == "CT":
                if split:
                    b = self.annotatorID + self.suffix
                else:
                    b = "_" + self.annotatorID + self.suffix
            else:
                b = self.annotatorID + self.suffix

        a = re.search(b, self.basename).start()

        self.individual_name = self.basename[:a]

        # Extract image information
        self.size = np.array(self.itkmask.GetSize())
        self.origin = np.array(self.itkmask.GetOrigin())
        self.orientation = np.array(self.itkmask.GetDirection()).reshape((3, 3))
        self.spacing = np.array(self.itkmask.GetSpacing())
        self.nibaffine = self.nibmask.affine
        self.nparray = self.nibmask.get_fdata()

    def to_mesh_and_pcd(
        self,
        mesh_dirname=None,
        pcd_dirname=None,
        mask_cleaning=True,
    ):
        nib_data = self.nibmask
        modality = self.modality
        individual_name = self.individual_name
        annotatorID = self.annotatorID

        if self.modality == "CT":
            (
                o3d_meshCT_L,
                o3d_meshCT_R,
                o3d_pcdCT_L,
                o3d_pcdCT_R,
                mask_cleaned_nib,
            ) = convert_to_mesh_and_pcd(
                nib_data,
                modality,
                individual_name,
                annotatorID,
                mesh_dirname,
                pcd_dirname,
                mask_cleaning,
            )
            return (
                o3d_meshCT_L,
                o3d_meshCT_R,
                o3d_pcdCT_L,
                o3d_pcdCT_R,
                mask_cleaned_nib,
            )

        if self.modality == "US":
            o3d_meshUS, o3d_pcdUS, mask_cleaned_nib = convert_to_mesh_and_pcd(
                nib_data,
                modality,
                individual_name,
                annotatorID,
                mesh_dirname,
                pcd_dirname,
                mask_cleaning,
            )
            return o3d_meshUS, o3d_pcdUS, mask_cleaned_nib

    def split(self, split_dirname=None):
        nib_data = self.nibmask
        modality = self.modality
        individual_name = self.individual_name
        annotatorID = self.annotatorID

        splitL_nib, splitR_nib = convert_to_split(
            nib_data, modality, individual_name, annotatorID, split_dirname
        )

        return splitL_nib, splitR_nib


class Landmarks:
    def __init__(self, ldkspath, annotatorID) -> None:
        assert ldkspath[-4:] == ".txt"
        self.path = ldkspath
        self.basename = os.path.basename(self.path)
        self.nparray = np.loadtxt(self.path)
        assert self.nparray.shape[1] == 3
        self.number_of_ldks = self.nparray.shape[0]
        # Determine the modality of the image
        self.modality = None
        self.modality = "CT" if "CT" in self.basename else "US"

        self.suffix = "_ldk" + self.modality + ".txt"

        if annotatorID == "gt" or annotatorID == "auto":
            self.annotatorID = ""
            b = self.suffix
        else:
            self.annotatorID = str(annotatorID)
            b = self.annotatorID + self.suffix

        a = re.search(b, self.basename).start()

        self.individual_name = self.basename[:a]

    def to_o3d(self, ply_dirname=None):
        if len(self.nparray.shape) == 1:
            self.nparray = self.nparray.reshape(1, self.nparray.shape[0])

        o3dldks = o3d.geometry.PointCloud()
        o3dldks.points = o3d.utility.Vector3dVector(self.nparray)

        plymark_path = join(ply_dirname, self.basename)
        o3d.io.write_point_cloud(plymark_path.replace(".txt", ".ply"), o3dldks)
        return o3dldks

    def noising(self, std):
        noised_ldks = np.random.normal(
            loc=self.nparray, scale=std * np.ones_like(self.nparray), size=None
        )
        self.nparray = noised_ldks
        return


class Mesh:
    def __init__(self, meshpath, annotatorID) -> None:
        assert meshpath[-4:] == ".obj"
        self.path = meshpath
        self.basename = os.path.basename(self.path)
        self.o3dmesh = o3d.io.read_triangle_mesh(self.path)

        # Determine the modality of the image
        self.modality = None
        self.modality = "CT" if "CT" in self.basename else "US"

        self.suffix = "meshface" + self.modality + ".obj"

        if annotatorID == "gt" or annotatorID == "auto":
            self.annotatorID = ""
            b = self.suffix
        else:
            self.annotatorID = str(annotatorID)
            b = self.annotatorID + self.suffix

        a = re.search(b, self.basename).start()

        self.individual_name = self.basename[:a]

    def to_nparraypcd(self):
        nparray = np.array(self.o3dmesh.vertices)
        return nparray

    def to_o3dpcd(self):
        nparray = self.to_nparraypcd()
        o3dpcd = o3d.geometry.PointCloud()
        o3dpcd.points = o3d.utility.Vector3dVector(nparray)
        return o3dpcd


def convert_to_mesh_and_pcd(
    nib_data,
    modality,
    individual_name,
    annotatorID,
    mesh_dirname=None,
    pcd_dirname=None,
    mask_cleaning=True,
):
    affine = nib_data.affine
    new_affine = np.linalg.solve(np.sign(affine), affine)
    sign_affine = np.sign(affine)
    sign_affine[sign_affine == 0.0] = 1.0
    sign_new_affine = np.sign(new_affine)
    sign_new_affine[sign_new_affine == 0.0] = 1.0
    mesh_orientation = np.diag(new_affine)[:3] @ np.sign(affine[:3, :3])

    mask_cleaned_nib = None  # initialization of this variable

    if annotatorID == "gt" or annotatorID == "auto":
        annotatorID = ""

    # print("*** mesh and pcd creation for: ", self.basename, " ***")

    out = cc3d.connected_components(nib_data.get_fdata())
    bins_origin = np.bincount(out.flatten())
    bins_copy = np.ndarray.tolist(np.bincount(out.flatten()))
    ind0 = 0
    bins_copy.remove(bins_origin[ind0])
    ind1 = np.where(bins_origin == max(bins_copy))[0][0]
    bins_copy.remove(bins_origin[ind1])

    if modality == "CT":
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
        ) = measure.marching_cubes(
            out1, spacing=mesh_orientation, step_size=1, method="lewiner"
        )
        (
            vertexsCT2,
            faceCT2,
            normalsCT2,
            valuesCT2,
        ) = measure.marching_cubes(
            out2, spacing=mesh_orientation, step_size=1, method="lewiner"
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
            meshL_path = join(
                mesh_dirname,
                individual_name + "L" + annotatorID + "meshfaceCT.obj",
            )
            meshR_path = join(
                mesh_dirname,
                individual_name + "R" + annotatorID + "meshfaceCT.obj",
            )

            o3d.io.write_triangle_mesh(meshL_path, o3d_meshCT_L)
            print("o3d_meshCT_L saved as: ", meshL_path)
            o3d.io.write_triangle_mesh(meshR_path, o3d_meshCT_R)
            print("o3d_meshCT_R saved as: ", meshR_path)

        if pcd_dirname is not None:
            pcdL_path = join(
                pcd_dirname,
                individual_name + "L" + annotatorID + "pcdCT.txt",
            )
            pcdR_path = join(
                pcd_dirname,
                individual_name + "R" + annotatorID + "pcdCT.txt",
            )

            np.savetxt(pcdL_path, np.asarray(o3d_meshCT_L.vertices), delimiter=", ")
            print("pcdCT_L_txt saved as: ", pcdL_path)
            np.savetxt(pcdR_path, np.asarray(o3d_meshCT_R.vertices), delimiter=", ")
            print("pcdCT_R_txt saved as: ", pcdR_path)

        """mask_cleaning"""
        if mask_cleaning:
            mask_cleaned_nib = nib.Nifti1Image(out_both, affine)
            print("cleaning done")

        print("case done")
        return o3d_meshCT_L, o3d_meshCT_R, o3d_pcdCT_L, o3d_pcdCT_R, mask_cleaned_nib

    if modality == "US":
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
        ) = measure.marching_cubes(
            out1, spacing=mesh_orientation, step_size=1, method="lewiner"
        )
        o3d_meshUS = o3d.geometry.TriangleMesh()
        o3d_meshUS.triangles = o3d.utility.Vector3iVector(faceUS)
        o3d_meshUS.vertices = o3d.utility.Vector3dVector(vertexsUS)

        o3d_pcdUS = o3d.geometry.PointCloud()
        o3d_pcdUS.points = o3d.utility.Vector3dVector(np.asarray(o3d_meshUS.vertices))

        if mesh_dirname is not None:
            mesh_path = join(
                mesh_dirname,
                individual_name + annotatorID + "meshfaceUS.obj",
            )

            o3d.io.write_triangle_mesh(mesh_path, o3d_meshUS)
            print("o3d_meshUS saved as: ", mesh_path)

        if pcd_dirname is not None:
            pcd_path = join(
                pcd_dirname,
                individual_name + annotatorID + "pcdUS.txt",
            )
            np.savetxt(pcd_path, np.asarray(o3d_meshUS.vertices), delimiter=", ")
            print("pcdUS_txt saved as: ", pcd_path)

        """mask_cleaning"""
        if mask_cleaning:
            mask_cleaned_nib = nib.Nifti1Image(out_both, affine)
            print("cleaning done")

        print("case done")
        return o3d_meshUS, o3d_pcdUS, mask_cleaned_nib


def convert_to_split(
    nib_data, modality, individual_name, annotatorID, split_dirname=None
):
    assert modality == "CT", "Applicable only on a CT mask"

    if annotatorID == "gt" or annotatorID == "auto":
        annotatorID = ""

    print("*** CT mask splitting ***")

    suffix = "_mask" + modality + ".nii.gz"

    nparray = nib_data.get_fdata()
    len0 = nparray.shape[0]
    mid0 = int(len0 / 2)

    nparrayR = np.zeros_like(nparray)
    nparrayR[:mid0, :, :] = nparray[:mid0, :, :].copy()

    nparrayL = np.zeros_like(nparray)
    nparrayL[mid0:, :, :] = nparray[mid0:, :, :].copy()

    if split_dirname is not None:
        splitR_path = join(
            split_dirname,
            individual_name + "R" + annotatorID + suffix,
        )
        splitR_nib = nib.Nifti1Image(nparrayR, nib_data.affine)
        nib.save(splitR_nib, splitR_path)
        print("splitR_nib saved as: ", splitR_path)

        splitL_path = join(
            split_dirname,
            individual_name + "L" + annotatorID + suffix,
        )
        splitL_nib = nib.Nifti1Image(nparrayL, nib_data.affine)
        nib.save(splitL_nib, splitL_path)
        print("splitL_nib saved as: ", splitL_path)

    return splitL_nib, splitR_nib


def fuse_masks(
    list_of_trusted_masks: list[Mask],
    trusted_img: Image,
    resizing,
    npmaxflow_lamda,
    img_intensity_scaling="normal",  # "normal" or "scale"
    fused_dirname=None,
):
    print(
        "*** Fusion of masks with staple (from Sitk) + maxflow (from numpymaxflow) ***"
    )

    seg_stack = []

    for trusted_mask in list_of_trusted_masks:
        # STAPLE requires we cast into int16 arrays
        itk_int16 = sitk.Cast(trusted_mask.itkmask, sitk.sitkInt16)
        seg_stack.append(itk_int16)

    # Run STAPLE algorithm ###
    itk_STAPLE_prob = sitk.STAPLE(seg_stack, 1.0)  # 1.0 specifies the foreground value

    nparray_itkprob = sitk.GetArrayFromImage(itk_STAPLE_prob)
    nparray_nibprob = np.transpose(nparray_itkprob)
    del nparray_itkprob
    nparray_nibimg = trusted_img.nparray

    # Data resizing to increase the running speed, if needed
    if resizing is not None:
        nparray_nibimg = resiz_nparray(
            nparray_nibimg, resizing, interpolmode="trilinear", binary=False
        )
        nparray_nibprob = resiz_nparray(
            nparray_nibprob, resizing, interpolmode="trilinear", binary=False
        )

    # Normalise intensity of nparray_nibimg
    if img_intensity_scaling == "normal":
        norm = NormalizeIntensity(nonzero=True, channel_wise=True)
    if img_intensity_scaling == "scale":
        norm = ScaleIntensity()
    nparray_nibimg = norm(nparray_nibimg)

    # convert values into np.float32
    nparray_nibimg = np.asarray(nparray_nibimg, np.float32)
    nparray_nibprob = np.asarray(nparray_nibprob, np.float32)

    # Run numpy maxflow
    fP = nparray_nibprob
    bP = 1.0 - fP
    Prob = np.asarray([bP, fP])
    sigma = np.std(nparray_nibimg)
    connectivity = 6
    nparray_nibimg = np.expand_dims(nparray_nibimg, axis=0)

    nparray_fused = maxflow.maxflow(
        nparray_nibimg, Prob, npmaxflow_lamda, sigma, connectivity
    )
    nparray_fused = np.squeeze(nparray_fused, axis=0)
    nparray_fused = np.asarray(nparray_fused, np.float32)

    # Turn back to the initial shape, if a resizing has been done
    if resizing is not None:
        init_size = [
            int(trusted_img.size[0]),
            int(trusted_img.size[1]),
            int(trusted_img.size[2]),
        ]
        nparray_fused = resiz_nparray(
            nparray_fused, init_size, interpolmode="trilinear", binary=True
        )

    fused_nib = nib.Nifti1Image(nparray_fused, trusted_img.nibaffine)

    if fused_dirname is not None:
        img_suffix = "_img" + trusted_img.modality + ".nii.gz"
        a = re.search(img_suffix, trusted_img.basename).start()
        individual_name = trusted_img.basename[:a]

        fused_path = join(
            fused_dirname, individual_name + "_mask" + trusted_img.modality + ".nii.gz"
        )
        nib.save(fused_nib, fused_path)
        print("fused_nib saved as: ", fused_path)

    return fused_nib


def fuse_landmarks(
    list_of_trusted_ldks: list[Landmarks],
    fused_dirname=None,
):
    print("*** Fusion of landmarks ***")
    sum_nparray = np.zeros((7, 3))

    i = 0
    for trusted_ldks in list_of_trusted_ldks:
        sum_nparray += trusted_ldks.nparray
        i += 1

    fused_nparray = sum_nparray / i

    if fused_dirname is not None:
        ldks_suffix = "_ldk" + list_of_trusted_ldks[0].modality + ".txt"
        a = re.search(ldks_suffix, list_of_trusted_ldks[0].basename).start()
        individual_name = list_of_trusted_ldks[0].basename[: a - 1]

        fused_path = join(
            fused_dirname,
            individual_name + "_ldk" + list_of_trusted_ldks[0].modality + ".txt",
        )
        np.savetxt(fused_path, fused_nparray)

    return fused_nparray


def plot_arrays(arrays):
    # Create a figure and subplots
    fig, axes = plt.subplots(1, len(arrays), figsize=(15, 5))

    # Plot each image on a separate subplot
    for i, array in enumerate(arrays):
        axes[i].imshow(array)
        axes[i].set_title(f"Image {i + 1}")

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def resiz_nparray(input_nparray, newsize, interpolmode, binary):
    """
    Resize a numpy array into a specified new size.

    Args:
        ...
    Returns:
        ...
    """
    binarizing = Compose(
        [
            EnsureType(data_type="tensor"),
            AsDiscrete(threshold=0.5, threshold_values=True),
        ]
    )

    addchanel_tensor = torch.unsqueeze((torch.from_numpy(input_nparray)), 0)
    if interpolmode == "trilinear":
        post_resiz = Resize(
            spatial_size=newsize,
            mode=interpolmode,
            align_corners=True,
        )
    else:
        post_resiz = Resize(
            spatial_size=newsize,
            mode=interpolmode,
        )

    resized_tensor0 = post_resiz(addchanel_tensor)

    if binary:
        resized_tensor0 = binarizing(resized_tensor0)

    resized_nparray = np.asarray(resized_tensor0).squeeze(0)

    return resized_nparray


def resiz_nib_data(input_nib, newsize, interpolmode, binary):
    """
    Resize a nibabel object a specified new size.

    Args:
        ...
    Returns:
        ...
    """
    input_nparray = input_nib.get_fdata()
    affine = input_nib.affine

    resized_nparray = resiz_nparray(input_nparray, newsize, interpolmode, binary)

    resized_nib = nib.Nifti1Image(resized_nparray, affine)

    return resized_nib


def clean_nibmask(nib_mask, number_of_kidney):
    assert number_of_kidney in [1, 2], " number_of_kidney must be 1 or 2"

    affine = nib_mask.affine

    out = cc3d.connected_components(nib_mask.get_fdata())
    bins_origin = np.bincount(out.flatten())
    bins_copy = np.ndarray.tolist(np.bincount(out.flatten()))
    ind0 = 0
    bins_copy.remove(bins_origin[ind0])
    ind1 = np.where(bins_origin == max(bins_copy))[0][0]
    bins_copy.remove(bins_origin[ind1])

    if number_of_kidney == 2:
        ind2 = np.where(bins_origin == max(bins_copy))[0][0]
        bins_copy.remove(bins_origin[ind2])
        out1 = out.copy()
        out1[out1 != ind1] = 0
        out2 = out.copy()
        out2[out2 != ind2] = 0
        out1[out1 > 0] = 1
        out2[out2 > 0] = 1
        out_both = out1 + out2

    if number_of_kidney == 1:
        out1 = out.copy()
        out1[out1 != ind1] = 0
        out1[out1 > 0] = 1
        out_both = out1

    clean_mask_nib = nib.Nifti1Image(out_both, affine)

    print("mask cleaned and contained ", number_of_kidney, " kidney(s)")
    return clean_mask_nib
