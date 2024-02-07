import numpy as np
import open3d as o3d
import SimpleITK as sitk
import vtk


def vtkmatrix_to_numpy(matrix):
    """
    Copies the elements of a vtkMatrix4x4 into a numpy array.

    :param matrix: The matrix to be copied into an array.
    :type matrix: vtk.vtkMatrix4x4
    :rtype: numpy.ndarray
    """
    m = np.ones((4, 4))
    for i in range(4):
        for j in range(4):
            m[i, j] = matrix.GetElement(i, j)
    return m


def create_pointcloud_polydata(points, colors=None):
    """https://github.com/lmb-freiburg/demon
    Creates a vtkPolyData object with the point cloud from numpy arrays

    points: numpy.ndarray
        pointcloud with shape (n,3)

    colors: numpy.ndarray
        uint8 array with colors for each point. shape is (n,3)

    Returns vtkPolyData object
    """
    vpoints = vtk.vtkPoints()
    vpoints.SetNumberOfPoints(points.shape[0])
    for i in range(points.shape[0]):
        vpoints.SetPoint(i, points[i])
    vpoly = vtk.vtkPolyData()
    vpoly.SetPoints(vpoints)

    if colors is not None:
        vcolors = vtk.vtkUnsignedCharArray()
        vcolors.SetNumberOfComponents(3)
        vcolors.SetName("Colors")
        vcolors.SetNumberOfTuples(points.shape[0])
        for i in range(points.shape[0]):
            vcolors.SetTuple3(i, colors[i, 0], colors[i, 1], colors[i, 2])
        vpoly.GetPointData().SetScalars(vcolors)

    vcells = vtk.vtkCellArray()

    for i in range(points.shape[0]):
        vcells.InsertNextCell(1)
        vcells.InsertCellPoint(i)

    vpoly.SetVerts(vcells)

    return vpoly, vpoints


def resample_itk(img_itk, transform_matrix):
    # Build the proper matrix for that
    trans = transform_matrix.copy()
    scale_matrix = np.diag(
        [
            np.linalg.norm(trans[:, 0]),
            np.linalg.norm(trans[:, 1]),
            np.linalg.norm(trans[:, 2]),
        ]
    )
    Inv_scale_matrix = np.diag(
        [
            1 / np.linalg.norm(trans[:, 0]),
            1 / np.linalg.norm(trans[:, 1]),
            1 / np.linalg.norm(trans[:, 2]),
        ]
    )
    ref_affine = trans[:3, :3] @ Inv_scale_matrix
    new_trans = trans.copy()
    new_trans[:3, :3] = ref_affine

    lps2ras = np.diag([-1, -1, 1, 1])
    ras2lps = np.diag([-1, -1, 1, 1])
    new_trans_itk = ras2lps @ new_trans @ lps2ras

    # Build the image reference
    img_array = sitk.GetArrayFromImage(img_itk)
    origin = np.array(img_itk.GetOrigin()).reshape((3, 1))
    ref_origin = new_trans_itk[:3, :3] @ origin + new_trans_itk[:3, 3].reshape((3, 1))
    direction = np.array(img_itk.GetDirection()).reshape((3, 3))
    ref_direction = new_trans_itk[:3, :3] @ direction
    ref_spacing = (np.diag(scale_matrix) * np.array(img_itk.GetSpacing())).tolist()

    img_ref_itk = sitk.GetImageFromArray(img_array)

    del img_array

    # SetOrigin is used for translation or to move the origin
    img_ref_itk.SetOrigin(ref_origin.flatten().tolist())
    # SetSpacing is applied because of the rescaling.
    img_ref_itk.SetSpacing(ref_spacing)
    # SetDirection is used for Tinit_itk
    img_ref_itk.SetDirection(ref_direction.flatten().tolist())
    # Set the transform for resampling
    tx = sitk.AffineTransform(3)
    tx.SetMatrix(np.eye(3).flatten().tolist())
    # Resample image
    img_resampled_itk = sitk.Resample(
        img_ref_itk,
        img_ref_itk,
        tx,
        interpolator=sitk.sitkLinear,
        defaultPixelValue=0.0,
    )

    return img_resampled_itk


def voxel2array(grid_index_array, array_shape):
    """
    convert a voxel_grid_index array to a fixed size array
    (N*3)->(voxel_size*voxel_size*voxel_size)

    :input grid_index_array: get from o3d.voxel_grid.get_voxels()
    :input array_shape: shape of the output. Here it is fixed to make sure that the output array could
                        content all the CT voxels
    :return array_voxel: array with shape(voxel_size*voxel_size*voxel_size),the grid_index in
    """
    array_voxel = np.zeros((array_shape[2], array_shape[1], array_shape[0]))
    array_voxel[
        grid_index_array[:, 2], grid_index_array[:, 1], grid_index_array[:, 0]
    ] = 1

    array_voxel = np.rot90(array_voxel, axes=(1, 2))
    array_voxel = np.rot90(array_voxel, axes=(1, 2))

    return array_voxel


def voxelization(o3dpcd, ref_itk):
    ref_voxelsize = np.max(np.array(ref_itk.GetSpacing()))
    ref_size = np.array(ref_itk.GetSize())

    pcdgrid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        o3dpcd,
        voxel_size=ref_voxelsize,  # to keep the mm
        min_bound=[0, 0, 0],
        max_bound=np.array([0, 0, 0])
        + np.array([ref_size[0], ref_size[1], ref_size[2]]),
    )

    # Get voxels
    pcdvoxel_list = pcdgrid.get_voxels()
    grid_index_list = list(map(lambda x: x.grid_index, pcdvoxel_list))
    grid_index_array = np.array(grid_index_list)
    pcdarray = voxel2array(grid_index_array=grid_index_array, array_shape=ref_size)
    pcdarray = pcdarray.astype(np.int32)

    return pcdarray


def array_to_itkmask(nparray: np.int32, ref_itk):
    ref_spacing = np.array(ref_itk.GetSpacing())
    voxel_size = np.max(ref_spacing)

    _itk = sitk.GetImageFromArray(nparray)
    _itk.SetSpacing(ref_spacing)  # IMPORTANT TO SET THE SPACING
    _itk.SetDirection(list(ref_itk.GetDirection()))
    _itk.SetOrigin(list(ref_itk.GetOrigin()))

    # Resampling (if necessary to position the mask in the initial direction. But here it is not necessary)
    tx = sitk.AffineTransform(3)
    tx.SetMatrix(np.diag(ref_spacing / voxel_size).flatten().tolist())

    out_itk = sitk.Resample(_itk, _itk, tx, sitk.sitkNearestNeighbor, 0.0)
    # Edge closing
    out_itk = sitk.BinaryMorphologicalClosing(out_itk, [10, 10, 10])
    # Hole filling to obtain the region
    out_itk = sitk.BinaryFillhole(out_itk, fullyConnected=False, foregroundValue=1.0)

    return out_itk
