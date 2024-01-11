"""
Here you can see some examples (NOT EXHAUSTIVE) of
running common data processings you could have to do with a list of individuals the TRUSTED dataset.
Based on them, you could run those you want.

IMPORTANT: You could adapt the config_file.yml file

# Example of command to run the tutorial ####
# python src/trusted_datapaper_ds/dataprocessing/tutorial_for_list.py --config_path configs/config_file.yml

"""
from os.path import isfile, join

import yaml
from natsort import natsorted

from trusted_datapaper_ds.dataprocessing import data as dt
from trusted_datapaper_ds.utils import makedir, parse_args


def build_list(
    data_config, modality, datatype, annotatorID, USlike_IDlist=None, CTlike_IDlist=None
):
    """
    This function build a list of data paths. It is useful to run some operations in series.
    INPUTS:
        data_config: the yaml object containing the elements in config_file.yml
        modality(respect the word case):
                'us' for Ultrasound
                'ct' for CT
        datatype:
                'img' for image
                'ma' for mask
                'me' for mesh
                'ld' for landmarks
        annotatorID:
                'gt' for ground truth
                'annotator1' for annotator1
                'annotator2' for annotator2
        USlike_IDlist: list of kidneys ID in the format: individual+kidney_side (ex: ['01R', '02L'])
        CTlike_IDlist: list of kidneys ID in the format: individual+kidney_side (ex: ['01', '02'])
    OUTPUT:
        data_path_list: the list of data paths
    """
    assert modality in ["us", "ct"], " modality must be in ['us', 'ct'] "
    assert datatype in [
        "img",
        "ma",
        "me",
        "ld",
    ], " datatype must be 'us' or 'ct' in ['img', 'ma', 'me', 'ld'] "
    assert annotatorID in [
        "gt",
        "annotator1",
        "annotator2",
    ], " annotatorID must be in ['gt', 'annotator1', 'annotator2'] "
    assert (
        int(USlike_IDlist is None) + int(CTlike_IDlist is None)
    ) == 1, "one and only one IDlist must be empty"

    data = data_config

    if annotatorID == "gt":
        annotator = ""
    else:
        annotator = annotatorID

    if modality == "ct":
        if datatype == "img":
            IDlist = CTlike_IDlist
            data_path_list = [
                join(
                    data["data_location"],
                    data["ctimgfol"],
                    individual + data["ctimg_end"],
                )
                for individual in IDlist
                if isfile(
                    join(
                        data["data_location"],
                        data["ctimgfol"],
                        individual + data["ctimg_end"],
                    )
                )
            ]
        if datatype == "ma":
            if USlike_IDlist is not None:
                IDlist = USlike_IDlist
                data_path_list = [
                    join(
                        data["data_location"],
                        data["ctspma" + data[annotatorID] + "fol"],
                        individual + "_" + data[annotator] + data["ctma_end"],
                    )
                    for individual in IDlist
                    if isfile(
                        join(
                            data["data_location"],
                            data["ctspma" + data[annotatorID] + "fol"],
                            individual + "_" + data[annotator] + data["ctma_end"],
                        )
                    )
                ]

        if datatype == "me":
            if USlike_IDlist is not None:
                data_path_list = [
                    join(
                        data["data_location"],
                        data["ctme" + data[annotatorID] + "fol"],
                        individual + data[annotator] + data["ctme_end"],
                    )
                    for individual in IDlist
                ]

        if datatype == "ld":
            data_path_list = [
                join(
                    data["data_location"],
                    data["ctld" + data[annotatorID] + "fol"],
                    individual + data[annotator] + data["ctme_end"],
                )
                for individual in IDlist
            ]

    return data_path_list


def main(
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
    data_analysis,
):
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        data = yaml.safe_load(yaml_file)

    # List of Images and Masks resizing (here US images, and masks from annotator 2) ###
    # Note: "resized_dirname" is the directory to save the resized data.
    if resizing:
        resized_dirname = data["out_location"]
        for ind in uslist:
            k_side = ind[-1]
            individual = ind[:-1]
            imgpath = join(
                data["data_location"],
                data["usimgfol"],
                individual + k_side + data["usimg_end"],
            )
            maskpath = join(
                data["data_location"],
                data["usma2fol"],
                individual + k_side + data["annotator2"] + data["usma_end"],
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

    # List of Images and Masks resizing (here CT images, and masks from annotator 1) ###
    # Note: "resized_dirname" is the directory to save the resized data.
    # Important Note: add the "_" (like in the example ) in the annotator CT mask names
    if resizing:
        resized_dirname = data["out_location"]
        for ind in ctlist:
            individual = ind
            imgpath = join(
                data["data_location"],
                data["ctimgfol"],
                individual + data["ctimg_end"],
            )
            maskpath = join(
                data["data_location"],
                data["ctma1fol"],
                individual + "_" + data["annotator1"] + data["ctma_end"],
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

    # Convert list of Masks to Meshes and PCDs (here CT ground truth) ###
    # Note: "mesh_dirname" is the directory to save the mesh,
    #       "pcd_dirname" is the directory to save the point cloud
    if CTmask_to_mesh_and_pcd:
        mesh_dirname = data["out_location"]
        pcd_dirname = data["out_location"]
        for ind in ctlist:
            individual = ind
            maskpath = join(
                data["data_location"],
                data["ctmagtfol"],
                individual + data["ctma_end"],
            )
            ctmask = dt.Mask(maskpath, annotatorID="gt")
            (
                o3d_meshCT_L,
                o3d_meshCT_R,
                o3d_pcdCT_L,
                o3d_pcdCT_R,
            ) = ctmask.to_mesh_and_pcd(
                mesh_dirname=mesh_dirname,
                pcd_dirname=pcd_dirname,
                mask_cleaning=False,
            )

    # Convert list of Masks to Meshes and PCDs (here US ground truth)###
    # Note: "mesh_dirname" is the directory to save the mesh,
    #       "pcd_dirname" is the directory to save the point cloud
    if USmask_to_mesh_and_pcd:
        mesh_dirname = data["out_location"]
        pcd_dirname = data["out_location"]
        for ind in uslist:
            k_side = ind[-1]
            individual = ind[:-1]
            maskpath = join(
                data["data_location"],
                data["usmagtfol"],
                individual + k_side + data["usma_end"],
            )
            usmask = dt.Mask(maskpath, annotatorID="gt")
            o3d_meshUS, o3d_pcdUS = usmask.to_mesh_and_pcd(
                mesh_dirname=mesh_dirname,
                pcd_dirname=pcd_dirname,
                mask_cleaning=False,
            )

    # Split list of CT masks (here from annotator 1) ###
    # Note: "split_dirname" is the directory to save the split data.
    # Important Note: add the "_" (like in the example ) in the annotator CT mask names
    if splitCTmask1:
        split_dirname = join(data["out_location"], data["ctspma1fol"])
        makedir(split_dirname)
        for ind in ctlist:
            individual = ind
            maskpath = join(
                data["data_location"],
                data["ctma1fol"],
                individual + "_" + data["annotator1"] + data["ctma_end"],
            )
            ctmask = dt.Mask(maskpath, annotatorID=data["annotator1"])
            nparrayL, nparrayR = ctmask.split(split_dirname=split_dirname)

    # Split CT mask (here from ground truth) ###
    # Note: "split_dirname" is the directory to save the split data.
    if splitCTmaskgt:
        split_dirname = join(data["out_location"], data["ctspmagtfol"])
        makedir(split_dirname)
        for ind in ctlist:
            individual = ind
            maskpath = join(
                data["data_location"],
                data["ctmagtfol"],
                individual + data["ctma_end"],
            )
            ctmask = dt.Mask(maskpath, annotatorID="gt")
            nparrayL, nparrayR = ctmask.split(split_dirname=split_dirname)

    # Shift the origin of list of images or masks (here CT images) ###
    # Note: "shifted_dirname" is the directory to save the shifted data.
    if shift_origin:
        shifted_dirname = join(data["out_location"], data["ct0imgfol"])
        makedir(shifted_dirname)
        for ind in ctlist:
            individual = ind
            imgpath = join(
                data["data_location"],
                data["ctimgfol"],
                individual + data["ctimg_end"],
            )
            ctimg = dt.Image(imgpath)
            img_itk_shifted = ctimg.shift_origin(shifted_dirname=shifted_dirname)
            print(type(img_itk_shifted))

    # Fuse list of masks from annotator1 and annotator2 (here CT masks) ###
    # Note: "fused_dirname" is the directory to save the fused mask.
    # Important Note: add the "_" (like in the example ) in the annotator CT mask names
    if fuse_CTmask:
        fused_dirname = data["out_location"]
        for ind in ctlist:
            individual = ind
            imgpath = join(
                data["data_location"],
                data["ctimgfol"],
                individual + data["ctimg_end"],
            )
            mask1path = join(
                data["data_location"],
                data["ctma1fol"],
                individual + "_" + data["annotator1"] + data["ctma_end"],
            )
            mask2path = join(
                data["data_location"],
                data["ctma2fol"],
                individual + "_" + data["annotator2"] + data["ctma_end"],
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

    # Fuse list of masks from annotator1 and annotator2 (here US masks) ###
    # Note: "fused_dirname" is the directory to save the fused mask.
    if fuse_USmask:
        fused_dirname = data["out_location"]
        for ind in uslist:
            k_side = ind[-1]
            individual = ind[:-1]
            imgpath = join(
                data["data_location"],
                data["usimgfol"],
                individual + k_side + data["usimg_end"],
            )
            mask1path = join(
                data["data_location"],
                data["usma1fol"],
                individual + k_side + data["annotator1"] + data["usma_end"],
            )
            mask2path = join(
                data["data_location"],
                data["usma2fol"],
                individual + k_side + data["annotator2"] + data["usma_end"],
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

    # Fuse list of landmark set from annotator1 and annotator2 (here CT landmarks) ###
    # Note: "fused_dirname" is the directory to save the fused mask.
    if fuse_landmark:
        fused_dirname = data["out_location"]
        for ind in ctlist:
            individual = ind
            for k_side in ["L", "R"]:
                ldk1path = join(
                    data["data_location"],
                    data["ctld1fol"],
                    individual + k_side + data["annotator1"] + data["ctld_end"],
                )
                ldk2path = join(
                    data["data_location"],
                    data["ctld2fol"],
                    individual + k_side + data["annotator2"] + data["ctld_end"],
                )
                ldks1 = dt.Landmarks(ldk1path)
                ldks2 = dt.Landmarks(ldk2path)
                list_of_ldks = [ldks1, ldks2]
                fused_nparray = dt.fuse_landmarks(
                    list_of_trusted_ldks=list_of_ldks,
                    fused_dirname=fused_dirname,
                )
                print(type(fused_nparray))

    # Fuse list of landmark set from annotator1 and annotator2 (here U landmarks) ###
    # Note: "fused_dirname" is the directory to save the fused mask.
    if fuse_landmark:
        fused_dirname = data["out_location"]
        for ind in uslist:
            k_side = ind[-1]
            individual = ind[:-1]
            ldk1path = join(
                data["data_location"],
                data["usld1fol"],
                individual + k_side + data["annotator1"] + data["usld_end"],
            )
            ldk2path = join(
                data["data_location"],
                data["usld2fol"],
                individual + k_side + data["annotator2"] + data["usld_end"],
            )
            ldks1 = dt.Landmarks(ldk1path)
            ldks2 = dt.Landmarks(ldk2path)
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
                    data["data_location"],
                    data["ctmegtfol"],
                    individual + k_side + data["ctme_end"],
                )
                mesh = dt.Mesh(meshpath)
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
                data["data_location"],
                data["usmegtfol"],
                individual + k_side + data["usme_end"],
            )
            mesh = dt.Mesh(meshpath)
            nparraypcd = mesh.to_nparraypcd()
            o3dpcd = mesh.to_o3dpcd()
            print(type(nparraypcd), nparraypcd.shape)
            print(type(o3dpcd))

    if build_a_list:
        USlike_IDlist = ["116L", "114R"]
        data_path_list = build_list(
            data_config=data,
            modality="ct",
            datatype="ma",
            annotatorID="annotator1",
            USlike_IDlist=USlike_IDlist,
            CTlike_IDlist=None,
        )
        print(data_path_list)

    # if data_analysis:
    #     dana.usdatanalysis(
    #         usma1_files,
    #         usma2_files,
    #         usmagt_files,
    #         usme1_files,
    #         usme2_files,
    #         usmegt_files,
    #         usld1_files,
    #         usld2_files,
    #         usldgt_files,
    #     )

    return


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as yaml_file:
        data = yaml.safe_load(yaml_file)

    allct = natsorted(
        data["ctfold"]["cv1"]
        + data["ctfold"]["cv2"]
        + data["ctfold"]["cv3"]
        + data["ctfold"]["cv4"]
        + data["ctfold"]["cv5"]
    )
    allus = natsorted(
        data["usfold"]["cv1"]
        + data["usfold"]["cv2"]
        + data["usfold"]["cv3"]
        + data["usfold"]["cv4"]
        + data["usfold"]["cv5"]
    )

    ctcv1 = data["ctfold"]["cv1"]
    uscv1 = data["usfold"]["cv1"]

    ctlist = ctcv1
    uslist = uscv1
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
    build_a_list = 1
    data_analysis = 0

    main(
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
        data_analysis,
    )


# Example of command to run the tutorial ####
# python src/trusted_datapaper_ds/dataprocessing/tutorial_for_list.py --config_path configs/config_file.yml
