from glob import glob
from os.path import join

import cc3d
import nibabel as nib
import numpy as np
import yaml
from natsort import natsorted

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.utils import makedir, parse_args

# def upsampling_indiv(refpath, inpath, output_folder, clean=True):
#     makedir(output_folder)

#     for ref, ind in zip(imgpath_list, predpath_list):
#         imgpath = ref
#         maskpath = ind

#         img = dt.Image(imgpath)
#         mask = dt.Mask(maskpath, annotatorID=config["auto"])
#         modality = img.modality

#         assert (
#             img.individual_name == mask.individual_name
#         ), "reference and input seem to be different"

#         gt_affine = img.nibaffine
#         newsize = img.size

#         resized_mask_nparray = mask.resize(
#             newsize=newsize, interpolmode="trilinear", resized_dirname=None
#         )

#         if clean:
#             out = cc3d.connected_components(resized_mask_nparray)
#             bins_origin = np.bincount(out.flatten())
#             bins_copy = np.ndarray.tolist(np.bincount(out.flatten()))
#             ind0 = 0
#             bins_copy.remove(bins_origin[ind0])
#             ind1 = np.where(bins_origin == max(bins_copy))[0][0]
#             bins_copy.remove(bins_origin[ind1])

#             if modality == "CT":
#                 ind2 = np.where(bins_origin == max(bins_copy))[0][0]
#                 bins_copy.remove(bins_origin[ind2])
#                 out1 = out.copy()
#                 out1[out1 != ind1] = 0
#                 out2 = out.copy()
#                 out2[out2 != ind2] = 0
#                 out1[out1 > 0] = 1
#                 out2[out2 > 0] = 1
#                 out_both = out1 + out2
#                 del out
#                 reupsampled_array = out_both.copy()
#                 print("unique reupsampled_array: ", np.unique(reupsampled_array))
#                 del out_both

#             if modality == "US":
#                 out1 = out.copy()
#                 out1[out1 != ind1] = 0
#                 out1[out1 > 0] = 1
#                 out_both = out1
#                 del out
#                 reupsampled_array = out_both.copy()
#                 print("unique reupsampled_array: ", np.unique(reupsampled_array))
#                 del out_both

#         else:
#             reupsampled_array = resized_mask_nparray

#         reupsampled_nib = nib.Nifti1Image(reupsampled_array, gt_affine)
#         reupsampled_data_path = join(output_folder, mask.basename)
#         nib.save(reupsampled_nib, reupsampled_data_path)

#         print(
#             "resizing: ",
#             img.individual_name,
#             " to size ",
#             reupsampled_array.shape,
#             " DONE",
#         )

#         del resized_mask_nparray

#     return


def upsampling_list(
    annotatorID, imgpath_list, predpath_list, output_folder, clean=True
):
    makedir(output_folder)

    for ref, ind in zip(imgpath_list, predpath_list):
        imgpath = ref
        maskpath = ind

        img = dt.Image(imgpath)
        mask = dt.Mask(maskpath, annotatorID)
        modality = img.modality

        assert (
            img.individual_name == mask.individual_name
        ), "Image and Prediction seem to be different"

        gt_affine = img.nibaffine
        newsize = img.size

        masknparray = mask.nparray
        resized_mask_nparray = dt.resiz_nparray(
            masknparray, newsize, interpolmode="trilinear"
        )

        if clean:
            out = cc3d.connected_components(resized_mask_nparray)
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
                reupsampled_array = out_both.copy()
                print("unique reupsampled_array: ", np.unique(reupsampled_array))
                del out_both

            if modality == "US":
                out1 = out.copy()
                out1[out1 != ind1] = 0
                out1[out1 > 0] = 1
                out_both = out1
                del out
                reupsampled_array = out_both.copy()
                print("unique reupsampled_array: ", np.unique(reupsampled_array))
                del out_both

        else:
            reupsampled_array = resized_mask_nparray

        reupsampled_nib = nib.Nifti1Image(reupsampled_array, gt_affine)
        reupsampled_data_path = join(output_folder, mask.basename)
        nib.save(reupsampled_nib, reupsampled_data_path)

        print(
            "resizing: ",
            img.individual_name,
            " to size ",
            reupsampled_array.shape,
            " DONE",
        )

        del resized_mask_nparray

    return


def main(config, imgpath_list, predpath_list, output_folder):
    annotatorID = config["auto"]

    upsampling_list(annotatorID, imgpath_list, predpath_list, output_folder, clean=True)

    return


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    ref_folder = config["ref_location"]
    input_folder = join(
        config["seg128location"], config["segmodel"], config["training_target"]
    )

    imgpath_list = natsorted(glob(join(ref_folder, "*.nii.gz")))
    predpath_list = natsorted(glob(join(input_folder, "*.nii.gz")))

    output_folder = join(
        config["seglocation"], config["segmodel"], config["training_target"]
    )

    main(config, imgpath_list, predpath_list, output_folder)
