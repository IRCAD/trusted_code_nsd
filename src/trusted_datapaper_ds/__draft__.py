import sys

from trusted_datapaper_ds.dataprocessing import data as dt

#################################################################################
#################################################################################

""" US Image or Mask initialization """
USimgpath = (
    "/home/wndzimbong/IRCAD_DOSSIER/2022_2023/DOUBLE_ANNOTATION_DATA/TRUSTED_MedImA_submission/US_DATA/"
    "US_images/01R_imgUS.nii.gz"
)
trusted_USimg = dt.Image(USimgpath)

USmaskpath = (
    "/home/wndzimbong/IRCAD_DOSSIER/2022_2023/DOUBLE_ANNOTATION_DATA/TRUSTED_MedImA_submission/US_DATA/"
    "US_masks/GT_estimated_masksUS/01R_maskUS.nii.gz"
)
trusted_USmask = dt.Mask(USmaskpath)

""" CT Image or Mask initialization """
CTimgpath = (
    "/home/wndzimbong/IRCAD_DOSSIER/2022_2023/DOUBLE_ANNOTATION_DATA/TRUSTED_MedImA_submission/CT_DATA/"
    "CT_images/01_imgCT.nii.gz"
)
trusted_CTimg = dt.Image(CTimgpath)

CTmaskpath = (
    "/home/wndzimbong/IRCAD_DOSSIER/2022_2023/DOUBLE_ANNOTATION_DATA/TRUSTED_MedImA_submission/CT_DATA/"
    "CT_masks/GT_estimated_masksCT/01_maskCT.nii.gz"
)
trusted_CTmask = dt.Mask(CTmaskpath)


"""resize (resizee_dir must be created)"""
# resized_dirname = "/home/wndzimbong/Bureau"
# resized_nparray = trusted_USimg.resize(newsize=[128,128,128], resized_dirname=resized_dirname)
# resized_nparray = trusted_USmask.resize(newsize=[128,128,128], resized_dirname=resized_dirname)
# resized_nparray = trusted_CTimg.resize(newsize=[128,128,128], resized_dirname=resized_dirname)
# resized_nparray = trusted_CTmask.resize(newsize=[128,128,128], resized_dirname=resized_dirname)

"""get/set modality"""
# trusted_mask.setmodality("CT")
# print(trusted_mask.getmodality())

"""setsuffix"""
# trusted_mask.setsuffix()
# print(trusted_mask.suffix)

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
# USmaskpath1 = (
#     "/home/wndzimbong/IRCAD_DOSSIER/2022_2023/DOUBLE_ANNOTATION_DATA/TRUSTED_MedImA_submission/US_DATA/"
#     "US_masks/Annotator1/01R1_maskUS.nii.gz"
# )
# trusted_USmask1 = dt.Mask(USmaskpath1)

# USmaskpath2 = (
#     "/home/wndzimbong/IRCAD_DOSSIER/2022_2023/DOUBLE_ANNOTATION_DATA/TRUSTED_MedImA_submission/US_DATA/"
#     "US_masks/Annotator2/01R2_maskUS.nii.gz"
# )
# trusted_USmask2 = dt.Mask(USmaskpath2)

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

""" Landmarks initialization """
# USldkspath1 = (
#     "/home/wndzimbong/IRCAD_DOSSIER/2022_2023/DOUBLE_ANNOTATION_DATA/TRUSTED_MedImA_submission/US_DATA/"
#     "US_landmarks/Annotator1/01R1_ldkUS.txt"
# )
# trusted_USldks1 = dt.Landmarks(USldkspath1)
# USldkspath2 = (
#     "/home/wndzimbong/IRCAD_DOSSIER/2022_2023/DOUBLE_ANNOTATION_DATA/TRUSTED_MedImA_submission/US_DATA/"
#     "US_landmarks/Annotator2/01R2_ldkUS.txt"
# )
# trusted_USldks2 = dt.Landmarks(USldkspath2)

# CTldkspath1 = (
#     "/home/wndzimbong/IRCAD_DOSSIER/2022_2023/DOUBLE_ANNOTATION_DATA/TRUSTED_MedImA_submission/CT_DATA/"
#     "CT_landmarks/Annotator1/01R1_ldkCT.txt"
# )
# trusted_CTldks1 = dt.Landmarks(CTldkspath1)
# CTldkspath2 = (
#     "/home/wndzimbong/IRCAD_DOSSIER/2022_2023/DOUBLE_ANNOTATION_DATA/TRUSTED_MedImA_submission/CT_DATA/"
#     "CT_landmarks/Annotator2/01R2_ldkCT.txt"
# )
# trusted_CTldks2 = dt.Landmarks(CTldkspath2)


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


sys.exit(0)
