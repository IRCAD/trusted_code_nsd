"""
Here you can see some examples (NOT EXHAUSTIVE) of
running common data processings you could have to do with a list of individuals the TRUSTED dataset.
Based on them, you could run those you want.

IMPORTANT: You could adapt the config_file.yml file

# Example of command to run the tutorial ####
# python src/trusted_datapaper_ds/dataprocessing/tutorial_for_list.py --config_path configs/config_file.yml

"""
from os.path import join

import yaml
from natsort import natsorted

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.dataprocessing.datanalysis_new import datanalysis
from trusted_datapaper_ds.utils import (
    build_analist,
    build_many_mask_analist,
    build_many_me_ld_analist,
    makedir,
    parse_args,
)


def main(
    ctlist_with_side,
    ctlist,
    uslist,
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
    build_a_list,
    usdata_analysis,
    ctdata_analysis,
):
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # List of Images and Masks resizing (here US images, and masks from annotator 2) ###
    # Note: "resized_dirname" is the directory to save the resized data.
    if resizing:
        resized_dirname = config["out_location"]
        for ind in uslist:
            k_side = ind[-1]
            individual = ind[:-1]
            imgpath = join(
                config["data_location"],
                config["usimgfol"],
                individual + k_side + config["usimg_end"],
            )
            maskpath = join(
                config["data_location"],
                config["usma2fol"],  # or config["usma" + config["annotator2"] + "fol"]
                individual + k_side + config["annotator2"] + config["usma_end"],
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

    # List of Images and Masks resizing (here CT images, and masks from annotator 1) ###
    # Note: "resized_dirname" is the directory to save the resized data.
    # Important Note: add the "_" (like in the example ) in the annotator CT mask names
    if resizing:
        resized_dirname = config["out_location"]
        for ind in ctlist:
            individual = ind
            imgpath = join(
                config["data_location"],
                config["ctimgfol"],
                individual + config["ctimg_end"],
            )
            maskpath = join(
                config["data_location"],
                config["ctma1fol"],  # or config["ctma" + config["annotator1"] + "fol"]
                individual + "_" + config["annotator1"] + config["ctma_end"],
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

    # Convert list of Masks to Meshes and PCDs (here CT ground truth) ###
    # Note: "mesh_dirname" is the directory to save the mesh,
    #       "pcd_dirname" is the directory to save the point cloud
    if CTmask_to_mesh_and_pcd:
        mesh_dirname = config["out_location"]
        pcd_dirname = config["out_location"]
        for ind in ctlist:
            individual = ind
            maskpath = join(
                config["data_location"],
                config["ctmagtfol"],  # or config["ctma" + config["gt"] + "fol"]
                individual + config["ctma_end"],
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

    # Convert list of Masks to Meshes and PCDs (here US ground truth)###
    # Note: "mesh_dirname" is the directory to save the mesh,
    #       "pcd_dirname" is the directory to save the point cloud
    if USmask_to_mesh_and_pcd:
        mesh_dirname = config["out_location"]
        pcd_dirname = config["out_location"]
        for ind in uslist:
            k_side = ind[-1]
            individual = ind[:-1]
            maskpath = join(
                config["data_location"],
                config["usmagtfol"],  # or config["usma" + config["gt"] + "fol"]
                individual + k_side + config["usma_end"],
            )
            usmask = dt.Mask(maskpath, annotatorID="gt")
            o3d_meshUS, o3d_pcdUS, mask_cleaned_nib = usmask.to_mesh_and_pcd(
                mesh_dirname=mesh_dirname,
                pcd_dirname=pcd_dirname,
                mask_cleaning=False,
            )

    # Split list of CT masks (here from annotator 1) ###
    # Note: "split_dirname" is the directory to save the split data.
    # Important Note: add the "_" (like in the example ) in the annotator CT mask names
    if splitCTmask1:
        split_dirname = join(config["out_location"], config["ctspma1fol"])
        makedir(split_dirname)
        for ind in ctlist:
            individual = ind
            maskpath = join(
                config["data_location"],
                config["ctma" + config["annotator1"] + "fol"],  # or config["ctma1fol"]
                individual + "_" + config["annotator1"] + config["ctma_end"],
            )
            ctmask = dt.Mask(maskpath, annotatorID=config["annotator1"])
            nibL, nibR = ctmask.split(split_dirname=split_dirname)

    # Split CT mask (here from ground truth) ###
    # Note: "split_dirname" is the directory to save the split data.
    if splitCTmaskgt:
        split_dirname = join(config["out_location"], config["ctspmagtfol"])
        makedir(split_dirname)
        for ind in ctlist:
            individual = ind
            maskpath = join(
                config["data_location"],
                config["ctma" + config["gt"] + "fol"],  # or config["ctmagtfol"]
                individual + config["ctma_end"],
            )
            ctmask = dt.Mask(maskpath, annotatorID="gt")
            nibL, nibR = ctmask.split(split_dirname=split_dirname)

    # Shift the origin of list of images or masks (here CT images) ###
    # Note: "shifted_dirname" is the directory to save the shifted data.
    if shift_origin:
        shifted_dirname = join(config["out_location"], config["ct0imgfol"])
        makedir(shifted_dirname)
        for ind in ctlist:
            individual = ind
            imgpath = join(
                config["data_location"],
                config["ctimgfol"],
                individual + config["ctimg_end"],
            )
            ctimg = dt.Image(imgpath)
            img_itk_shifted = ctimg.shift_origin(shifted_dirname=shifted_dirname)
            print(type(img_itk_shifted))

    # Fuse list of masks from annotator1 and annotator2 (here CT masks) ###
    # Note: "fused_dirname" is the directory to save the fused mask.
    # Important Note: add the "_" (like in the example ) in the annotator CT mask names
    if fuse_CTmask:
        fused_dirname = config["out_location"]
        for ind in ctlist:
            individual = ind
            imgpath = join(
                config["data_location"],
                config["ctimgfol"],
                individual + config["ctimg_end"],
            )
            mask1path = join(
                config["data_location"],
                config["ctma" + config["annotator1"] + "fol"],  # config["ctma1fol"],
                individual + "_" + config["annotator1"] + config["ctma_end"],
            )
            mask2path = join(
                config["data_location"],
                config["ctma2fol"],  # or config["ctma" + config["annotator2"] + "fol"],
                individual + "_" + config["annotator2"] + config["ctma_end"],
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

    # Fuse list of masks from annotator1 and annotator2 (here US masks) ###
    # Note: "fused_dirname" is the directory to save the fused mask.
    if fuse_USmask:
        fused_dirname = config["out_location"]
        for ind in uslist:
            k_side = ind[-1]
            individual = ind[:-1]
            imgpath = join(
                config["data_location"],
                config["usimgfol"],
                individual + k_side + config["usimg_end"],
            )
            mask1path = join(
                config["data_location"],
                config["usma1fol"],  # or config["usma" + config["annotator1"] + "fol"]
                individual + k_side + config["annotator1"] + config["usma_end"],
            )
            mask2path = join(
                config["data_location"],
                config["usma2fol"],  # or config["usma" + config["annotator2"] + "fol"]
                individual + k_side + config["annotator2"] + config["usma_end"],
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

    # Fuse list of landmark set from annotator1 and annotator2 (here CT landmarks) ###
    # Note: "fused_dirname" is the directory to save the fused mask.
    if fuse_landmark:
        fused_dirname = config["out_location"]
        for ind in ctlist:
            individual = ind
            for k_side in ["L", "R"]:
                ldk1path = join(
                    config["data_location"],
                    config[
                        "ctld1fol"
                    ],  # or config["ctld" + config["annotator1"] + "fol"]
                    individual + k_side + config["annotator1"] + config["ctld_end"],
                )
                ldk2path = join(
                    config["data_location"],
                    config[
                        "ctld2fol"
                    ],  # or config["ctld" + config["annotator2"] + "fol"]
                    individual + k_side + config["annotator2"] + config["ctld_end"],
                )
                ldks1 = dt.Landmarks(ldk1path, annotatorID=config["annotator1"])
                ldks2 = dt.Landmarks(ldk2path, annotatorID=config["annotator2"])
                list_of_ldks = [ldks1, ldks2]
                fused_nparray = dt.fuse_landmarks(
                    list_of_trusted_ldks=list_of_ldks,
                    fused_dirname=fused_dirname,
                )
                print(type(fused_nparray))

    # Fuse list of landmark set from annotator1 and annotator2 (here U landmarks) ###
    # Note: "fused_dirname" is the directory to save the fused mask.
    if fuse_landmark:
        fused_dirname = config["out_location"]
        for ind in uslist:
            k_side = ind[-1]
            individual = ind[:-1]
            ldk1path = join(
                config["data_location"],
                config["usld1fol"],  # or config["usld" + config["annotator1"] + "fol"]
                individual + k_side + config["annotator1"] + config["usld_end"],
            )
            ldk2path = join(
                config["data_location"],
                config["usld2fol"],  # or config["usld" + config["annotator2"] + "fol"]
                individual + k_side + config["annotator2"] + config["usld_end"],
            )
            ldks1 = dt.Landmarks(ldk1path, annotatorID=config["annotator1"])
            ldks2 = dt.Landmarks(ldk2path, annotatorID=config["annotator2"])
            list_of_ldks = [ldks1, ldks2]
            fused_nparray = dt.fuse_landmarks(
                list_of_trusted_ldks=list_of_ldks,
                fused_dirname=fused_dirname,
            )
            print(type(fused_nparray))

    # Read list of meshes and convert the vertices into pcd as numpy.array or like open3d pcd object (here CT meshes)
    if mesh_to_pcd:
        for ind in ctlist:
            individual = ind
            for k_side in ["L", "R"]:
                meshpath = join(
                    config["data_location"],
                    config["ctmegtfol"],  # or config["ctme" + config["gt"] + "fol"]
                    individual + k_side + config["ctme_end"],
                )
                mesh = dt.Mesh(meshpath, annotatorID="gt")
                nparraypcd = mesh.to_nparraypcd()
                o3dpcd = mesh.to_o3dpcd()
                print(type(nparraypcd), nparraypcd.shape)
                print(type(o3dpcd))

    # Read list of meshes and convert the vertices into pcd as numpy.array or like open3d pcd object (here US meshes)
    if mesh_to_pcd:
        for ind in uslist:
            k_side = ind[-1]
            individual = ind[:-1]
            meshpath = join(
                config["data_location"],
                config["usmegtfol"],  # or config["usme" + config["gt"] + "fol"]
                individual + k_side + config["usme_end"],
            )
            mesh = dt.Mesh(meshpath, annotatorID="gt")
            nparraypcd = mesh.to_nparraypcd()
            o3dpcd = mesh.to_o3dpcd()
            print(type(nparraypcd), nparraypcd.shape)
            print(type(o3dpcd))

    if build_a_list:
        USlike_IDlist = ["116L", "114R"]
        data_path_list = build_analist(
            data_config=config,
            modality="us",
            datatype="ld",
            # annotatorID='gt',
            annotatorID="annotator1",
            USlike_IDlist=USlike_IDlist,
            CTlike_IDlist=None,
        )
        print(data_path_list)

    if usdata_analysis:
        modality = "us"
        usdatanalysis_folder = join(
            config["out_location"], config["us_analysis_folder"]
        )
        makedir(usdatanalysis_folder)

        USlike_IDlist = uslist
        CTlike_IDlist = None

        usma1_files, usma2_files, usmagt_files = build_many_mask_analist(
            modality, config, USlike_IDlist, CTlike_IDlist
        )

        (
            usme1_files,
            usme2_files,
            usmegt_files,
            usld1_files,
            usld2_files,
            usldgt_files,
        ) = build_many_me_ld_analist(modality, config, USlike_IDlist, CTlike_IDlist)

        datanalysis(
            modality,
            usdatanalysis_folder,
            usma1_files,
            usma2_files,
            usmagt_files,
            usme1_files,
            usme2_files,
            usmegt_files,
            usld1_files,
            usld2_files,
            usldgt_files,
        )

    if ctdata_analysis:
        modality = "ct"
        ctdatanalysis_folder = join(
            config["out_location"], config["ct_analysis_folder"]
        )
        makedir(ctdatanalysis_folder)

        # For CT Masks
        USlike_IDlist = None
        CTlike_IDlist = ctlist
        ctma1_files, ctma2_files, ctmagt_files = build_many_mask_analist(
            modality, config, USlike_IDlist, CTlike_IDlist
        )

        # For CT Meshes and Landmarks which are name like US ones
        USlike_IDlist = ctlist_with_side
        CTlike_IDlist = None
        (
            ctme1_files,
            ctme2_files,
            ctmegt_files,
            ctld1_files,
            ctld2_files,
            ctldgt_files,
        ) = build_many_me_ld_analist(modality, config, USlike_IDlist, CTlike_IDlist)

        datanalysis(
            modality,
            ctdatanalysis_folder,
            ctma1_files,
            ctma2_files,
            ctmagt_files,
            ctme1_files,
            ctme2_files,
            ctmegt_files,
            ctld1_files,
            ctld2_files,
            ctldgt_files,
        )

    return


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    allct = natsorted(
        config["ctfold"]["cv1"]
        + config["ctfold"]["cv2"]
        + config["ctfold"]["cv3"]
        + config["ctfold"]["cv4"]
        + config["ctfold"]["cv5"]
    )
    allct_with_side = natsorted([j + "L" for j in allct] + [j + "R" for j in allct])
    allus = natsorted(
        config["usfold"]["cv1"]
        + config["usfold"]["cv2"]
        + config["usfold"]["cv3"]
        + config["usfold"]["cv4"]
        + config["usfold"]["cv5"]
    )

    ctcv1 = config["ctfold"]["cv1"]
    ctcv1_with_side = natsorted([j + "L" for j in ctcv1] + [j + "R" for j in ctcv1])
    uscv1 = config["usfold"]["cv1"]

    # ctlist_with_side = ctcv1_with_side
    # ctlist = ctcv1
    # uslist = uscv1

    ctlist_with_side = allct_with_side
    ctlist = allct
    uslist = allus

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
    build_a_list = 0
    usdata_analysis = 1
    ctdata_analysis = 1

    main(
        ctlist_with_side,
        ctlist,
        uslist,
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
        build_a_list,
        usdata_analysis,
        ctdata_analysis,
    )


# Example of command to run the tutorial ####
# python src/trusted_datapaper_ds/dataprocessing/tutorial_for_list.py --config_path configs/config_file.yml
