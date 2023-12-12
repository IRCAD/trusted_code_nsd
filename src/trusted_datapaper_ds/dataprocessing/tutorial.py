"""
Here you can see examples of running common data processings you could have to do with the TRUSTED dataset
"""
import os
from os.path import join

import yaml

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.utils import parse_args


def main(
    resizing,
    CTmask_to_mesh_and_pcd,
    USmask_to_mesh_and_pcd,
    splitCTmask,
    shift_origin,
    fuse_mask,
):
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        data = yaml.safe_load(yaml_file)

    # Image or Mask resizing ###
    # Note: "resized_dirname" is the directory to save the resized data.
    #       You have to set it, if you want to save.
    if resizing:
        fpath = join(
            data["data_location"],
            data["usimgfol"],
            data["individual"] + data["k_side"] + data["usimg_end"],
        )
        newsize = [128, 128, 128]
        resized_dirname = None

        base = os.path.basename(fpath)
        if "img" in base:
            data = dt.Image(fpath)
        elif "mask" in base:
            data = dt.Mask(fpath)
        else:
            TypeError("Type no supported by our resize function")

        resized_nparray = data.resize(newsize=newsize, resized_dirname=resized_dirname)
        print(type(resized_nparray))

    # Convert CT ground truth Mask to Mesh and PCD ###
    # Note: "mesh_dirname" is the directory to save the mesh,
    #       "pcd_dirname" is the directory to save the point cloud
    #       You have to set it, if you want to save.
    if CTmask_to_mesh_and_pcd:
        maskpath = join(
            data["data_location"],
            data["ctimgfol"],
            data["individual"] + data["ctma_end"],
        )
        ctmask = dt.Mask(maskpath)
        mesh_dirname = None
        pcd_dirname = None
        o3d_meshCT_L, o3d_meshCT_R, o3d_pcdCT_L, o3d_pcdCT_R = ctmask.to_mesh_and_pcd(
            mesh_dirname=mesh_dirname,
            pcd_dirname=pcd_dirname,
            ok_with_suffix=True,
            mask_cleaning=False,
        )

    # Convert US ground truth Mask to Mesh and PCD ###
    # Note: "mesh_dirname" is the directory to save the mesh,
    #       "pcd_dirname" is the directory to save the point cloud
    #       You have to set it, if you want to save.
    if USmask_to_mesh_and_pcd:
        maskpath = join(
            data["data_location"],
            data["usimgfol"],
            data["individual"] + data["k_side"] + data["usma_end"],
        )
        usmask = dt.Mask(maskpath)
        mesh_dirname = None
        pcd_dirname = None
        o3d_meshUS, o3d_pcdUS = usmask.to_mesh_and_pcd(
            mesh_dirname=mesh_dirname,
            pcd_dirname=pcd_dirname,
            ok_with_suffix=True,
            mask_cleaning=False,
        )

    # Split CT mask from annotator 1 ###
    # Note: "split_dirname" is the directory to save the splitted data.
    #       You have to set it, if you want to save.
    if splitCTmask:
        maskpath = join(
            data["data_location"],
            data["ctimgfol"],
            data["individual"] + data["annotator1"] + data["ctma_end"],
        )
        ctmask = dt.Mask(maskpath)
        split_dirname = None
        nparrayL, nparrayR = ctmask.split(split_dirname=split_dirname)

    # Shift the origin of a CT image or mask ###
    # Note: "shifted_dirname" is the directory to save the shifted data.
    #       You have to set it, if you want to save.
    if shift_origin:
        imgpath = join(
            data["data_location"],
            data["ctimgfol"],
            data["individual"] + data["ctimg_end"],
        )
        ctimg = dt.Image(imgpath)
        shifted_dirname = None
        img_itk_shifted = ctimg.shift_origin(shifted_dirname=shifted_dirname)
        print(type(img_itk_shifted))

    # fuse CT masks from two annotator
    if fuse_mask:
        fused_dirname = None
        imgpath = join(
            data["data_location"],
            data["ctimgfol"],
            data["individual"] + data["ctimg_end"],
        )
        mask1path = join(
            data["data_location"],
            data["ctimgfol"],
            data["individual"] + data["annotator1"] + data["ctma_end"],
        )
        mask2path = join(
            data["data_location"],
            data["ctimgfol"],
            data["individual"] + data["annotator2"] + data["ctma_end"],
        )
        img = dt.Image(imgpath)
        mask1 = dt.Mask(mask1path)
        mask2 = dt.Mask(mask2path)
        list_of_trusted_masks = [mask1, mask2]

        fused_nib = dt.fuse_masks(
            list_of_trusted_masks,
            trusted_img=img,
            resizing=[
                512,
                384,
                384,
            ],  # I reduce the data shape to increase the speed of the process
            fused_dirname=fused_dirname,
        )
        print(type(fused_nib))

    return


if __name__ == "__main__":
    resizing = 0
    CTmask_to_mesh_and_pcd = 0
    USmask_to_mesh_and_pcd = 0
    splitCTmask = 0
    shift_origin = 0
    fuse_mask = 0

    main(
        resizing,
        CTmask_to_mesh_and_pcd,
        USmask_to_mesh_and_pcd,
        splitCTmask,
        shift_origin,
        fuse_mask,
    )


# USimgpath = join(data['data_location'], data["usimgfol"], data["individual"]+data["k_side"]+data["usimg_end"])
# USimgpath = join(data['data_location'], data["usimgfol"], data["individual"]+data["k_side"]+data["usimg_end"])


# USmaskpath1 = "~/US_DATA/US_masks/Annotator1/01R1_maskUS.nii.gz"
# trusted_USmask1 = dt.Mask(USmaskpath1)

# USmaskpath2 = "~/US_DATA/US_masks/Annotator2/01R2_maskUS.nii.gz"
# trusted_USmask2 = dt.Mask(USmaskpath2)

# USmaskpath = "~/US_DATA/US_masks/GT_estimated_masksUS/01R_maskUS.nii.gz"
# trusted_USmask = dt.Mask(USmaskpath)


# """ CT Image or Mask reading file and class initialization """
# CTimgpath = "~/CT_DATA/CT_images/01_imgCT.nii.gz"
# trusted_CTimg = dt.Image(CTimgpath)

# CTmaskpath = "~/CT_DATA/CT_masks/GT_estimated_masksCT/01_maskCT.nii.gz"
# trusted_CTmask = dt.Mask(CTmaskpath)

# CTmaskpath1 = "~/CT_DATA/CT_masks/Annotator1/01_1_maskCT.nii.gz"
# trusted_CTmask1 = dt.Mask(CTmaskpath1)

# CTmaskpath2 = "~/CT_DATA/CT_masks/Annotator2/01_2_maskCT.nii.gz"
# trusted_CTmask2 = dt.Mask(CTmaskpath2)


# """ Mesh reading file and class initialization """
# USmeshpath1 = "~/US_DATA/US_meshes/Annotator1/01R1meshfaceUS.obj"
# trusted_USmesh1 = dt.Mesh(USmeshpath1)
# USmeshpath2 = "~/US_DATA/US_meshes/Annotator2/01R2meshfaceUS.obj"
# trusted_USmesh2 = dt.Mesh(USmeshpath2)

# CTmeshpath1 = "~/CT_DATA/CT_meshes/Annotator1/01R1meshfaceCT.obj"
# trusted_CTmesh1 = dt.Mesh(CTmeshpath1)
# CTmeshpath2 = "~/CT_DATA/CT_meshes/Annotator2/01R2meshfaceCT.obj"
# trusted_CTmesh2 = dt.Mesh(CTmeshpath2)


# """ Landmarks set reading file and class initialization """
# USldkspath1 = "~/US_DATA/US_landmarks/Annotator1/01R1_ldkUS.txt"
# trusted_USldks1 = dt.Landmarks(USldkspath1)
# USldkspath2 = "~/US_DATA/US_landmarks/Annotator2/01R2_ldkUS.txt"
# trusted_USldks2 = dt.Landmarks(USldkspath2)

# CTldkspath1 = "~/CT_DATA/CT_landmarks/Annotator1/01R1_ldkCT.txt"
# trusted_CTldks1 = dt.Landmarks(CTldkspath1)
# CTldkspath2 = "~/CT_DATA/CT_landmarks/Annotator2/01R2_ldkCT.txt"
# trusted_CTldks2 = dt.Landmarks(CTldkspath2)


# """ Image or mask resizing (resized_dir must be created) """
# resizedUS_dirname = "~/US_DATA/US_masks/resized_GT_masksUS"
# resized_nparray = trusted_USimg.resize(newsize=[128,128,128], resized_dirname=resizedUS_dirname)
# resizedCT_dirname = "~/CT_DATA/CT_masks/resized_GT_masksCT"
# resized_nparray = trusted_CTimg.resize(newsize=[128,128,128], resized_dirname=resizedCT_dirname)


"""to_mesh_and_pcd (mesh_dir and pcd_dir must be created)"""
# mesh_dirname = "/home/wndzimbong/Bureau"
# pcd_dirname = "/home/wndzimbong/Bureau"
# o3d_meshUS, o3d_pcdUS = trusted_USmask.to_mesh_and_pcd(mesh_dirname=mesh_dirname,
#                                                     pcd_dirname=pcd_dirname,
#                                                     ok_with_suffix=True,
#                                                     mask_cleaning=False)

# o3d_meshCT_L, o3d_meshCT_R, o3d_pcdCT_L, o3d_pcdCT_R = trusted_CTmask.to_mesh_and_pcd(mesh_dirname=mesh_dirname,
#                                                                                    pcd_dirname=pcd_dirname,
#                                                                                    ok_with_suffix=True,
#                                                                                    mask_cleaning=False)

""" split CT """
# split_dirname = "/home/wndzimbong/Bureau"
# nparrayL, nparrayR = trusted_CTmask.split(split_dirname=split_dirname)

""" shift_origin """
# shifted_dirname = "/home/wndzimbong/Bureau"
# img_itk_shifed = trusted_CTimg.shift_origin(shifted_dirname=shifted_dirname)
# mask_itk_shifed = trusted_CTmask.shift_origin(shifted_dirname=shifted_dirname)

""" staple masks fusion """
# list_of_trusted_masks = [trusted_USmask1, trusted_USmask2]
# fused_dirname = "/home/wndzimbong/Bureau"
# fused_nib = dt.fuse_masks(
#     list_of_trusted_masks,
#     trusted_img=trusted_USimg,
#     resizing=[512, 384, 384],
#     # resizing=[128, 128, 128],
#     fused_dirname=fused_dirname,
# )
# print(type(fused_nib))
# print(dir(fused_nib))


""" Landmarks fusion """
# list_of_trusted_ldks = [trusted_USldks1, trusted_USldks2]
# fused_dirname = "/home/wndzimbong/Bureau"
# fused_nparray = dt.fuse_landmarks(
#     list_of_trusted_ldks,
#     fused_dirname=fused_dirname,
# )
# print(fused_nparray)


""" Mesh initialization """
# USmeshpath1 = (
#     "/home/wndzimbong/IRCAD_DOSSIER/2022_2023/DOUBLE_ANNOTATION_DATA/TRUSTED_MedImA_submission/US_DATA/" \
#     "US_meshes/Annotator1/01R1meshfaceUS.obj"
# )
# trusted_USmesh1 = dt.Mesh(USmeshpath1)
# USmeshpath2 = (
#     "/home/wndzimbong/IRCAD_DOSSIER/2022_2023/DOUBLE_ANNOTATION_DATA/TRUSTED_MedImA_submission/US_DATA/" \
#     "US_meshes/Annotator2/01R2meshfaceUS.obj"
# )
# trusted_USmesh2 = dt.Mesh(USmeshpath2)

# CTmeshpath1 = (
#     "/home/wndzimbong/IRCAD_DOSSIER/2022_2023/DOUBLE_ANNOTATION_DATA/TRUSTED_MedImA_submission/US_DATA/" \
#     "CT_meshes/Annotator1/01R1meshfaceCT.obj"
# )
# trusted_CTmesh1 = dt.Mesh(CTmeshpath1)
# CTmeshpath2 = (
#     "/home/wndzimbong/IRCAD_DOSSIER/2022_2023/DOUBLE_ANNOTATION_DATA/TRUSTED_MedImA_submission/US_DATA/" \
#     "CT_meshes/Annotator2/01R2meshfaceCT.obj"
# )
# trusted_CTmesh2 = dt.Mesh(CTmeshpath2)


# US2nparraypcd = trusted_USmesh2.to_nparraypcd()
# print(dir(trusted_USmesh2))

# US2o3dpcd = trusted_USmesh2.to_o3dpcd()
# print(dir(US2o3dpcd))


# python src/trusted_datapaper_ds/__draft__.py
