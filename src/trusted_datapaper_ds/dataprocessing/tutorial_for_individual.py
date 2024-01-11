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
        data = yaml.safe_load(yaml_file)

    print(data)

    # Image and Mask resizing (here a US image, and mask from annotator 2) ###
    # Note: "resized_dirname" is the directory to save the resized data.
    if resizing:
        resized_dirname = data["out_location"]
        imgpath = join(
            data["data_location"],
            data["usimgfol"],
            data["individual"] + data["k_side"] + data["usimg_end"],
        )
        maskpath = join(
            data["data_location"],
            data["usma2fol"],
            data["individual"] + data["k_side"] + data["annotator2"] + data["usma_end"],
        )
        newsize = [128, 128, 128]

        img = dt.Image(imgpath)
        mask = dt.Mask(
            maskpath, annotatorID=data["annotator2"]
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
        resized_dirname = data["out_location"]
        imgpath = join(
            data["data_location"],
            data["ctimgfol"],
            data["individual"] + data["ctimg_end"],
        )
        maskpath = join(
            data["data_location"],
            data["ctma1fol"],
            data["individual"] + "_" + data["annotator1"] + data["ctma_end"],
        )
        newsize = [128, 128, 128]

        img = dt.Image(imgpath)
        mask = dt.Mask(
            maskpath, annotatorID=data["annotator1"]
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
        mesh_dirname = data["out_location"]
        pcd_dirname = data["out_location"]
        maskpath = join(
            data["data_location"],
            data["ctmagtfol"],
            data["individual"] + data["ctma_end"],
        )
        ctmask = dt.Mask(maskpath, annotatorID="gt")
        o3d_meshCT_L, o3d_meshCT_R, o3d_pcdCT_L, o3d_pcdCT_R = ctmask.to_mesh_and_pcd(
            mesh_dirname=mesh_dirname,
            pcd_dirname=pcd_dirname,
            mask_cleaning=False,
        )

    # Convert Mask to Mesh and PCD (here a US ground truth)###
    # Note: "mesh_dirname" is the directory to save the mesh,
    #       "pcd_dirname" is the directory to save the point cloud
    if USmask_to_mesh_and_pcd:
        mesh_dirname = data["out_location"]
        pcd_dirname = data["out_location"]
        maskpath = join(
            data["data_location"],
            data["usmagtfol"],
            data["individual"] + data["k_side"] + data["usma_end"],
        )
        usmask = dt.Mask(maskpath, annotatorID="gt")
        o3d_meshUS, o3d_pcdUS = usmask.to_mesh_and_pcd(
            mesh_dirname=mesh_dirname,
            pcd_dirname=pcd_dirname,
            mask_cleaning=False,
        )

    # Split CT mask (here from annotator 1) ###
    # Note: "split_dirname" is the directory to save the split data.
    # Important Note: add the "_" (like in the example ) in the annotator CT mask names
    if splitCTmask1:
        split_dirname = data["out_location"]
        maskpath = join(
            data["data_location"],
            data["ctma1fol"],
            data["individual"] + "_" + data["annotator1"] + data["ctma_end"],
        )
        ctmask = dt.Mask(maskpath, annotatorID=data["annotator1"])
        nparrayL, nparrayR = ctmask.split(split_dirname=split_dirname)

    # Split CT mask (here from ground truth) ###
    # Note: "split_dirname" is the directory to save the split data.
    if splitCTmaskgt:
        split_dirname = data["out_location"]
        maskpath = join(
            data["data_location"],
            data["ctmagtfol"],
            data["individual"] + data["ctma_end"],
        )
        ctmask = dt.Mask(maskpath, annotatorID="gt")
        nparrayL, nparrayR = ctmask.split(split_dirname=split_dirname)

    # Shift the origin of an image or mask (here a CT image) ###
    # Note: "shifted_dirname" is the directory to save the shifted data.
    if shift_origin:
        shifted_dirname = data["out_location"]
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
    # Important Note: add the "_" (like in the example ) in the annotator CT mask names
    if fuse_CTmask:
        fused_dirname = data["out_location"]
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
        mask1 = dt.Mask(mask1path, annotatorID=data["annotator1"])
        mask2 = dt.Mask(mask2path, annotatorID=data["annotator2"])
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
        fused_dirname = data["out_location"]
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
        mask1 = dt.Mask(mask1path, annotatorID=data["annotator1"])
        mask2 = dt.Mask(mask2path, annotatorID=data["annotator2"])
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
        fused_dirname = data["out_location"]
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
    mesh_to_pcd = 0

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
# python src/trusted_datapaper_ds/dataprocessing/tutorial_for_individual.py --config_path configs/config_file.yml
