"""
Here you can see some examples (NOT EXHAUSTIVE) of
running common data processings you could have to do with an indivudual of the TRUSTED dataset.
Based on them, you could run those you want.

IMPORTANT: You could adapt the config_file.yml file

# Example of command to run the tutorial ####
# python src/trusted_datapaper_ds/dataprocessing/tutorial_for_individual.py --config_path configs/config_file.yml

"""
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
        config = yaml.safe_load(yaml_file)

    # Image and Mask resizing (here a US image, and mask from annotator 2) ###
    # Note: "resized_dirname" is the directory to save the resized data.
    if resizing:
        resized_dirname = config["out_location"]
        imgpath = join(
            config["data_location"],
            config["usimgfol"],
            config["individual"] + config["k_side"] + config["usimg_end"],
        )
        maskpath = join(
            config["data_location"],
            config["usma2fol"],
            config["individual"]
            + config["k_side"]
            + config["annotator2"]
            + config["usma_end"],
        )
        newsize = [128, 128, 128]

        img = dt.Image(imgpath)
        mask = dt.Mask(
            maskpath, annotatorID=config["annotator2"]
        )  # or simply annotatorID='2'

        resized_img_nparray = img.resize(
            newsize=newsize, resized_dirname=resized_dirname
        )
        resized_mask_nparray = mask.resize(
            newsize=newsize, resized_dirname=resized_dirname
        )
        print(type(resized_img_nparray))
        print(type(resized_mask_nparray))

    # Image and Mask resizing (here a CT image, and mask from annotator 1) ###
    # Note: "resized_dirname" is the directory to save the resized data.
    # Important Note: add the "_" (like in the example ) in the annotator CT mask names
    if resizing:
        resized_dirname = config["out_location"]
        imgpath = join(
            config["data_location"],
            config["ctimgfol"],
            config["individual"] + config["ctimg_end"],
        )
        maskpath = join(
            config["data_location"],
            config["ctma1fol"],
            config["individual"] + "_" + config["annotator1"] + config["ctma_end"],
        )
        newsize = [128, 128, 128]

        img = dt.Image(imgpath)
        mask = dt.Mask(
            maskpath, annotatorID=config["annotator1"]
        )  # or simply annotatorID='2'

        print(img.modality)
        print(mask.modality)
        resized_img_nparray = img.resize(
            newsize=newsize, resized_dirname=resized_dirname
        )
        resized_mask_nparray = mask.resize(
            newsize=newsize, resized_dirname=resized_dirname
        )
        print(type(resized_img_nparray))
        print(type(resized_mask_nparray))

    # Convert Mask to Mesh and PCD (here a CT ground truth) ###
    # Note: "mesh_dirname" is the directory to save the mesh,
    #       "pcd_dirname" is the directory to save the point cloud
    if CTmask_to_mesh_and_pcd:
        mesh_dirname = config["out_location"]
        pcd_dirname = config["out_location"]
        maskpath = join(
            config["data_location"],
            config["ctmagtfol"],
            config["individual"] + config["ctma_end"],
        )
        ctmask = dt.Mask(maskpath, annotatorID="gt")
        (
            o3d_meshCT_L,
            o3d_meshCT_R,
            o3d_pcdCT_L,
            o3d_pcdCT_R,
            mask_cleaned_nib,
        ) = ctmask.to_mesh_and_pcd(
            mesh_dirname=mesh_dirname,
            pcd_dirname=pcd_dirname,
            mask_cleaning=False,
        )

    # Convert Mask to Mesh and PCD (here a US ground truth)###
    # Note: "mesh_dirname" is the directory to save the mesh,
    #       "pcd_dirname" is the directory to save the point cloud
    if USmask_to_mesh_and_pcd:
        mesh_dirname = config["out_location"]
        pcd_dirname = config["out_location"]
        maskpath = join(
            config["data_location"],
            config["usmagtfol"],
            config["individual"] + config["k_side"] + config["usma_end"],
        )
        usmask = dt.Mask(maskpath, annotatorID="gt")
        o3d_meshUS, o3d_pcdUS, mask_cleaned_nib = usmask.to_mesh_and_pcd(
            mesh_dirname=mesh_dirname,
            pcd_dirname=pcd_dirname,
            mask_cleaning=False,
        )

    # Split CT mask (here from annotator 1) ###
    # Note: "split_dirname" is the directory to save the split data.
    # Important Note: add the "_" (like in the example ) in the annotator CT mask names
    if splitCTmask1:
        split_dirname = config["out_location"]
        maskpath = join(
            config["data_location"],
            config["ctma1fol"],
            config["individual"] + "_" + config["annotator1"] + config["ctma_end"],
        )
        ctmask = dt.Mask(maskpath, annotatorID=config["annotator1"])
        nibL, nibR = ctmask.split(split_dirname=split_dirname)

    # Split CT mask (here from ground truth) ###
    # Note: "split_dirname" is the directory to save the split data.
    if splitCTmaskgt:
        split_dirname = config["out_location"]
        maskpath = join(
            config["data_location"],
            config["ctmagtfol"],
            config["individual"] + config["ctma_end"],
        )
        ctmask = dt.Mask(maskpath, annotatorID="gt")
        nibL, nibR = ctmask.split(split_dirname=split_dirname)

    # Shift the origin of an image or mask (here a CT image) ###
    # Note: "shifted_dirname" is the directory to save the shifted data.
    if shift_origin:
        shifted_dirname = config["out_location"]
        imgpath = join(
            config["data_location"],
            config["ctimgfol"],
            config["individual"] + config["ctimg_end"],
        )
        ctimg = dt.Image(imgpath)
        img_itk_shifted = ctimg.shift_origin(shifted_dirname=shifted_dirname)
        print(type(img_itk_shifted))

    # Fuse masks from annotator1 and annotator2 (here a CT mask) ###
    # Note: "fused_dirname" is the directory to save the fused mask.
    # Important Note: add the "_" (like in the example ) in the annotator CT mask names
    if fuse_CTmask:
        fused_dirname = config["out_location"]
        imgpath = join(
            config["data_location"],
            config["ctimgfol"],
            config["individual"] + config["ctimg_end"],
        )
        mask1path = join(
            config["data_location"],
            config["ctma1fol"],
            config["individual"] + "_" + config["annotator1"] + config["ctma_end"],
        )
        mask2path = join(
            config["data_location"],
            config["ctma2fol"],
            config["individual"] + "_" + config["annotator2"] + config["ctma_end"],
        )
        img = dt.Image(imgpath)
        mask1 = dt.Mask(mask1path, annotatorID=config["annotator1"])
        mask2 = dt.Mask(mask2path, annotatorID=config["annotator2"])
        list_of_masks = [mask1, mask2]

        fused_nib = dt.fuse_masks(
            list_of_trusted_masks=list_of_masks,
            trusted_img=img,
            npmaxflow_lamda=2.5,
            img_intensity_scaling="normal",  # "normal" or "scale"
            resizing=None,  # I reduce the data shape to increase the speed of the process. Can be None
            fused_dirname=fused_dirname,
        )
        print(type(fused_nib))

    # Fuse masks from annotator1 and annotator2 (here a US mask) ###
    # Note: "fused_dirname" is the directory to save the fused mask.
    if fuse_USmask:
        fused_dirname = config["out_location"]
        imgpath = join(
            config["data_location"],
            config["usimgfol"],
            config["individual"] + config["k_side"] + config["usimg_end"],
        )
        mask1path = join(
            config["data_location"],
            config["usma1fol"],
            config["individual"]
            + config["k_side"]
            + config["annotator1"]
            + config["usma_end"],
        )
        mask2path = join(
            config["data_location"],
            config["usma2fol"],
            config["individual"]
            + config["k_side"]
            + config["annotator2"]
            + config["usma_end"],
        )
        img = dt.Image(imgpath)
        mask1 = dt.Mask(mask1path, annotatorID=config["annotator1"])
        mask2 = dt.Mask(mask2path, annotatorID=config["annotator2"])
        list_of_masks = [mask1, mask2]

        fused_nib = dt.fuse_masks(
            list_of_trusted_masks=list_of_masks,
            trusted_img=img,
            npmaxflow_lamda=2.5,
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
    if fuse_landmark:
        fused_dirname = config["out_location"]
        ldk1path = join(
            config["data_location"],
            config["ctld1fol"],
            config["individual"]
            + config["k_side"]
            + config["annotator1"]
            + config["ctld_end"],
        )
        ldk2path = join(
            config["data_location"],
            config["ctld2fol"],
            config["individual"]
            + config["k_side"]
            + config["annotator2"]
            + config["ctld_end"],
        )
        ldks1 = dt.Landmarks(ldk1path, annotatorID=config["annotator1"])
        ldks2 = dt.Landmarks(ldk2path, annotatorID=config["annotator2"])
        list_of_ldks = [ldks1, ldks2]
        fused_nparray = dt.fuse_landmarks(
            list_of_trusted_ldks=list_of_ldks,
            fused_dirname=fused_dirname,
        )
        print(type(fused_nparray))

    # Read a mesh and convert the vertices into pcd as numpy.array or like open3d pcd object (here a US mesh)
    if mesh_to_pcd:
        meshpath = join(
            config["data_location"],
            config["usmegtfol"],
            config["individual"] + config["k_side"] + config["usme_end"],
        )
        mesh = dt.Mesh(meshpath, annotatorID=config["gt"])
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
    fuse_USmask = 1
    fuse_landmark = 0
    mesh_to_pcd = 0

    main(
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
    )


# Example of command to run the tutorial ####
# python src/trusted_datapaper_ds/dataprocessing/tutorial_for_individual.py --config_path configs/config_file.yml
