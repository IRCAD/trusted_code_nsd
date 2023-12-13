"""
Here you can see some examples (NOT EXHAUSTIVE) of
running common data processings you could have to do with the TRUSTED dataset.
Based on them, you could run those you want.

# Example of command to run the tutorial ####
# python src/trusted_datapaper_ds/dataprocessing/tutorial.py --config_path configs/config_file.yml

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
    splitCTmask1,
    splitCTmaskgt,
    shift_origin,
    fuse_USmask,
    fuse_CTmask,
    fuse_landmark,
    mesh_to_pcd,
):
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        data = yaml.safe_load(yaml_file)

    # Image or Mask resizing (here a US image) ###
    # Note: "resized_dirname" is the directory to save the resized data.
    #       You have to set it, if you want to save.
    if resizing:
        resized_dirname = "/home/wndzimbong/Bureau"
        fpath = join(
            data["data_location"],
            data["usimgfol"],
            data["individual"] + data["k_side"] + data["usimg_end"],
        )
        newsize = [128, 128, 128]
        base = os.path.basename(fpath)
        if "img" in base:
            data = dt.Image(fpath)
        elif "mask" in base:
            data = dt.Mask(fpath)
        else:
            TypeError("Type no supported by our resize function")
        resized_nparray = data.resize(newsize=newsize, resized_dirname=resized_dirname)
        print(type(resized_nparray))

    # Convert Mask to Mesh and PCD (here a CT ground truth) ###
    # Note: "mesh_dirname" is the directory to save the mesh,
    #       "pcd_dirname" is the directory to save the point cloud
    #       You have to set it, if you want to save.
    if CTmask_to_mesh_and_pcd:
        mesh_dirname = "/home/wndzimbong/Bureau"
        pcd_dirname = "/home/wndzimbong/Bureau"
        maskpath = join(
            data["data_location"],
            data["ctmagtfol"],
            data["individual"] + data["ctma_end"],
        )
        ctmask = dt.Mask(maskpath)
        o3d_meshCT_L, o3d_meshCT_R, o3d_pcdCT_L, o3d_pcdCT_R = ctmask.to_mesh_and_pcd(
            mesh_dirname=mesh_dirname,
            pcd_dirname=pcd_dirname,
            ok_with_suffix=True,
            mask_cleaning=False,
        )

    # Convert Mask to Mesh and PCD (here a US ground truth)###
    # Note: "mesh_dirname" is the directory to save the mesh,
    #       "pcd_dirname" is the directory to save the point cloud
    #       You have to set it, if you want to save.
    if USmask_to_mesh_and_pcd:
        mesh_dirname = "/home/wndzimbong/Bureau"
        pcd_dirname = "/home/wndzimbong/Bureau"
        maskpath = join(
            data["data_location"],
            data["usmagtfol"],
            data["individual"] + data["k_side"] + data["usma_end"],
        )
        usmask = dt.Mask(maskpath)
        o3d_meshUS, o3d_pcdUS = usmask.to_mesh_and_pcd(
            mesh_dirname=mesh_dirname,
            pcd_dirname=pcd_dirname,
            ok_with_suffix=True,
            mask_cleaning=False,
        )

    # Split CT mask (here from annotator 1) ###
    # Note: "split_dirname" is the directory to save the split data.
    #       You have to set it, if you want to save.
    # Important Note: add the "_" (like in the example ) in the annotator CT mask names
    if splitCTmask1:
        split_dirname = "/home/wndzimbong/Bureau"
        maskpath = join(
            data["data_location"],
            data["ctma1fol"],
            data["individual"] + "_" + data["annotator1"] + data["ctma_end"],
        )
        ctmask = dt.Mask(maskpath)
        nparrayL, nparrayR = ctmask.split(split_dirname=split_dirname)

    # Split CT mask (here from ground truth) ###
    # Note: "split_dirname" is the directory to save the split data.
    #       You have to set it, if you want to save.
    # Important Note: add the "_" (like in the example ) in the annotator CT mask names
    if splitCTmaskgt:
        split_dirname = "/home/wndzimbong/Bureau"
        maskpath = join(
            data["data_location"],
            data["ctmagtfol"],
            data["individual"] + data["ctma_end"],
        )
        ctmask = dt.Mask(maskpath)
        nparrayL, nparrayR = ctmask.split(split_dirname=split_dirname)

    # Shift the origin of an image or mask (here a CT image) ###
    # Note: "shifted_dirname" is the directory to save the shifted data.
    #       You have to set it, if you want to save.
    if shift_origin:
        shifted_dirname = "/home/wndzimbong/Bureau"
        imgpath = join(
            data["data_location"],
            data["ctimgfol"],
            data["individual"] + data["ctimg_end"],
        )
        ctimg = dt.Image(imgpath)
        img_itk_shifted = ctimg.shift_origin(shifted_dirname=shifted_dirname)
        print(type(img_itk_shifted))

    # Fuse masks from annotator1 and annotator2 (here a CT mask) ###
    # Note: "fused_dirname" is the directory to save the fused mask.
    #       You have to set it, if you want to save.
    # Important Note: add the "_" (like in the example ) in the annotator CT mask names
    if fuse_CTmask:
        fused_dirname = "/home/wndzimbong/Bureau"
        imgpath = join(
            data["data_location"],
            data["ctimgfol"],
            data["individual"] + data["ctimg_end"],
        )
        mask1path = join(
            data["data_location"],
            data["ctma1fol"],
            data["individual"] + "_" + data["annotator1"] + data["ctma_end"],
        )
        mask2path = join(
            data["data_location"],
            data["ctma2fol"],
            data["individual"] + "_" + data["annotator2"] + data["ctma_end"],
        )
        img = dt.Image(imgpath)
        mask1 = dt.Mask(mask1path)
        mask2 = dt.Mask(mask2path)
        list_of_masks = [mask1, mask2]

        fused_nib = dt.fuse_masks(
            list_of_trusted_masks=list_of_masks,
            trusted_img=img,
            img_intensity_scaling="normal",  # "normal" or "scale"
            resizing=None,  # I reduce the data shape to increase the speed of the process. Can be None
            fused_dirname=fused_dirname,
        )
        print(type(fused_nib))

    # Fuse masks from annotator1 and annotator2 (here a US mask) ###
    # Note: "fused_dirname" is the directory to save the fused mask.
    #       You have to set it, if you want to save.
    if fuse_USmask:
        fused_dirname = "/home/wndzimbong/Bureau"
        imgpath = join(
            data["data_location"],
            data["usimgfol"],
            data["individual"] + data["k_side"] + data["usimg_end"],
        )
        mask1path = join(
            data["data_location"],
            data["usma1fol"],
            data["individual"] + data["k_side"] + data["annotator1"] + data["usma_end"],
        )
        mask2path = join(
            data["data_location"],
            data["usma2fol"],
            data["individual"] + data["k_side"] + data["annotator2"] + data["usma_end"],
        )
        img = dt.Image(imgpath)
        mask1 = dt.Mask(mask1path)
        mask2 = dt.Mask(mask2path)
        list_of_masks = [mask1, mask2]

        fused_nib = dt.fuse_masks(
            list_of_trusted_masks=list_of_masks,
            trusted_img=img,
            img_intensity_scaling="normal",  # "normal" or "scale"
            resizing=[
                512,
                384,
                384,
            ],  # I reduce the data shape to increase the speed of the process. Can be None
            fused_dirname=fused_dirname,
        )
        print(type(fused_nib))

    # Fuse landmarks from annotator1 and annotator2 (here CT la,ndmarks) ###
    # Note: "fused_dirname" is the directory to save the fused mask.
    #       You have to set it, if you want to save.
    if fuse_landmark:
        fused_dirname = "/home/wndzimbong/Bureau"
        ldk1path = join(
            data["data_location"],
            data["ctld1fol"],
            data["individual"] + data["k_side"] + data["annotator1"] + data["ctld_end"],
        )
        ldk2path = join(
            data["data_location"],
            data["ctld2fol"],
            data["individual"] + data["k_side"] + data["annotator2"] + data["ctld_end"],
        )
        ldks1 = dt.Landmarks(ldk1path)
        ldks2 = dt.Landmarks(ldk2path)
        list_of_ldks = [ldks1, ldks2]
        fused_nparray = dt.fuse_landmarks(
            list_of_trusted_ldks=list_of_ldks,
            fused_dirname=fused_dirname,
        )
        print(type(fused_nparray))

    # Read a mesh and convert the vertices into pcd as numpy.array or like open3d pcd object (here a US mesh)
    if mesh_to_pcd:
        meshpath = join(
            data["data_location"],
            data["usmegtfol"],
            data["individual"] + data["k_side"] + data["usme_end"],
        )
        mesh = dt.Mesh(meshpath)
        nparraypcd = mesh.to_nparraypcd()
        o3dpcd = mesh.to_o3dpcd()
        print(type(nparraypcd), nparraypcd.shape)
        print(type(o3dpcd))

    return


if __name__ == "__main__":
    resizing = 0
    CTmask_to_mesh_and_pcd = 0
    USmask_to_mesh_and_pcd = 0
    splitCTmask1 = 0
    splitCTmaskgt = 0
    shift_origin = 0
    fuse_CTmask = 0
    fuse_USmask = 0
    fuse_landmark = 0
    mesh_to_pcd = 1

    main(
        resizing,
        CTmask_to_mesh_and_pcd,
        USmask_to_mesh_and_pcd,
        splitCTmask1,
        splitCTmaskgt,
        shift_origin,
        fuse_CTmask,
        fuse_USmask,
        fuse_landmark,
        mesh_to_pcd,
    )


# Example of command to run the tutorial ####
# python src/trusted_datapaper_ds/dataprocessing/tutorial.py --config_path configs/config_file.yml
