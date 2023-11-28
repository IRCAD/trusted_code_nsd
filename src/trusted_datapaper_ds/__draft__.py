# import monai
import sys

from trusted_datapaper_ds import data as dt

# print(sitk.__version__)
# print(type(itkimg))
# print(monai.__version__)
# print(np.__version__)
# print(skimage.__version__)

#################################################################################
#################################################################################

# # import skimage
# imgpath = (
#     "/home/wndzimbong/IRCAD_DOSSIER/2022_2023/DOUBLE_ANNOTATION_DATA/TRUSTED_MedImA_submission/US_DATA/"
#     "US_images/01L_imgUS.nii.gz"
# )

# imgpath = "/home/wndzimbong/IRCAD_DOSSIER/2022_2023/DOUBLE_ANNOTATION_DATA/TRUSTED_MedImA_submission/US_DATA/"\
#     "US_masks/GT_estimated_masksUS/01R_maskUS.nii.gz"

imgpath = (
    "/home/wndzimbong/IRCAD_DOSSIER/2022_2023/DOUBLE_ANNOTATION_DATA/TRUSTED_MedImA_submission/CT_DATA/"
    "CT_masks/GT_estimated_masksCT/02_maskCT.nii.gz"
)

resized_nibimg_path = "/home/wndzimbong/Bureau/01R_maskUS_resized.nii.gz"

trusted_mask = dt.Mask(imgpath)

"""resize"""
# resized_nparray = trusted_mask.resize(newsize=[128,128,128], resized_nibimg_path=resized_nibimg_path)

"""get/set modality"""
# trusted_mask.setmodality("CT")
# print(trusted_mask.getmodality())

"""setsuffix"""
# trusted_mask.setsuffix()
# print(trusted_mask.suffix)

"""tomesh"""
# mesh_dirname = "/home/wndzimbong/Bureau"
# pcd_dirname = "/home/wndzimbong/Bureau"
# o3d_meshCT_L, o3d_meshCT_R, o3d_pcdCT_L, o3d_pcdCT_R = trusted_mask.to_mesh_and_pcd(mesh_dirname=None,
#                                                                                    pcd_dirname=None,
#                                                                                    ok_with_suffix=True,
#                                                                                    mask_cleaning=False)


# path = imgpath
# basename = os.path.basename(path)
# itkimg = sitk.ReadImage(path)
# nibimg = nib.load(path)

# # Determine the modality of the image
# modality = None
# if "US" in basename:
#     modality = "US"
# if "CT" in basename:
#     modality = "CT"

# '''Extract image information'''
# size = np.array(itkimg.GetSize())
# origin = np.array(itkimg.GetOrigin())
# orientation = np.array(itkimg.GetDirection()).reshape((3, 3))
# spacing = np.array(itkimg.GetSpacing())
# nibaffine = nibimg.affine
# nparray = nibimg.get_fdata()

# '''Reisizing'''
# newsize = [128,128,128]
# interpolmode = "trilinear"
# resized_nparray = F.interpolate(
#     torch.unsqueeze(torch.unsqueeze(torch.from_numpy(nparray), 0), 0),
#     size=newsize,
#     mode=interpolmode,
#     align_corners=interpolmode=="trilinear",
# )
# transform = AsDiscrete(threshold_values=True, logit_thresh=0.5)
# resized_nparray = transform(resized_nparray)
# resized_nparray = (torch.squeeze(torch.squeeze(resized_nparray, 0), 0)).numpy()

# print("unique values in: ", np.unique(resized_nparray))

# # Save the resized NiBabel image if specified
# if resized_nibimg_path is not None:
#     resized_nibimg = nib.Nifti1Image(resized_nparray, nibaffine)
#     nib.save(resized_nibimg, resized_nibimg_path)


# python src/trusted_datapaper_ds/__draft__.py


sys.exit(0)
