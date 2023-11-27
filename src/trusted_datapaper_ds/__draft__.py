import nibabel as nib

# import monai
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

# import skimage
imgpath = (
    "/home/wndzimbong/IRCAD_DOSSIER/2022_2023/DOUBLE_ANNOTATION_DATA/TRUSTED_MedImA_submission/US_DATA/"
    "US_images/01L_imgUS.nii.gz"
)

# imgpath = "/home/wndzimbong/IRCAD_DOSSIER/2022_2023/DOUBLE_ANNOTATION_DATA/TRUSTED_MedImA_submission/US_DATA/"\
#     "US_masks/GT_estimated_masksUS/01R_maskUS.nii.gz"

itkimg = sitk.ReadImage(imgpath)
nibimg = nib.load(imgpath)

origin = np.array(itkimg.GetOrigin())
orientation = np.array(itkimg.GetDirection()).reshape((3, 3))
spacing = np.array(itkimg.GetSpacing())
nibaffine = nibimg.affine

print(origin, orientation, spacing)
print("#")
print(nibaffine)

# print(sitk.__version__)
# print(type(itkimg))
# print(monai.__version__)
# print(np.__version__)
# print(skimage.__version__)
# print(itkimg.GetMetaData('aux_file'))

size = np.array(itkimg.GetSize())
print(size)

nparray = nibimg.get_fdata()
# nparray = nparray.astype(np.float32)
print(nparray.shape)

new_nparray = F.interpolate(
    torch.unsqueeze(torch.unsqueeze(torch.from_numpy(nparray), 0), 0),
    size=[200, 128, 128],
    mode="trilinear",
    align_corners=True,
)
new_nparray = (torch.squeeze(torch.squeeze(new_nparray, 0), 0)).numpy()
print(new_nparray.shape)

new_nibimg = nib.Nifti1Image(new_nparray, nibaffine)
print(new_nibimg.affine)
