import numpy as np
import open3d as o3d
from ripser import Rips
from persim import PersistenceImager
import torch
import collections
from contextlib import redirect_stdout
import os
collections.Iterable = collections.abc.Iterable

# TODO: Create k_nn function. Use FAIS by facebook for speed..?
def k_nn(all_points, centroids, k):
    '''
    :param all_points:
    :param centroids:
    :param k: amount of neighbors.
    :return:
    function receives all points in point cloud and centroids and amount of neighbors and returns
    #centroid patches (centroid and k nearest neighbors)
    '''
    pass

#TODO: Check later on if should use gridGCN algorithm
# def sample(pcd, num_of_centroids, num_of_point_in_each_patch):
#     '''
#     :param pcd: original point cloud (type: open3d.cpu.pybind.geometry.PointCloud)
#     :param num_of_centroids: amount of subsampled points
#     :param num_of_point_in_each_patch: how many neighbors of the chosen centroid should we return
#     :return: centroids, NumPy arrays - distances to each neighbor, list of point clouds (without centroids)
#     '''
#     centroids = pcd.farthest_point_down_sample(num_of_centroids)
#     points1 = np.asarray(pcd.points)
#     points2 = np.asarray(centroids.points)
#     dimension = 3
#
#     index = faiss.IndexFlatL2(dimension)
#     index.add(points1)
#
#     num_neighbors = num_of_point_in_each_patch
#     distances_arr = []
#     neighbor_point_clouds = []
#     for centroid in points2:
#         distances, indices = index.search(centroid.reshape(1, -1), num_neighbors + 1)  # +1 to exclude the closest point
#         nearest_neighbors = points1[indices[0][1:]]  # Exclude the closest point
#
#         distances_arr.append(distances[0][1:])
#
#         neighbor_pcd = o3d.geometry.PointCloud()
#         neighbor_pcd.points = o3d.utility.Vector3dVector(nearest_neighbors)
#         neighbor_point_clouds.append(neighbor_pcd)
#
#     distances_arr = np.array(distances_arr)
#     return centroids, distances_arr, neighbor_point_clouds


#TODO: Create calculateSurfaces function.
def calculateSurfaces(patch_points, centroid):
    '''

    :param patch_points:
    :param centroid:
    :return:
    '''

    # ELIAHU HOROWITZ STUFF: check them out!

    # def get_plane_eq(unorganized_pc, ransac_n_pts=50):
    #     o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc))
    #     plane_model, inliers = o3d_pc.segment_plane(distance_threshold=0.004, ransac_n=ransac_n_pts,
    #                                                 num_iterations=1000)
    #     return plane_model
    #
    # def connected_components_cleaning(organized_pc, organized_rgb, image_path):
    #     unorganized_pc = mvt_util.organized_pc_to_unorganized_pc(organized_pc)
    #     unorganized_rgb = mvt_util.organized_pc_to_unorganized_pc(organized_rgb)
    #
    #     nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
    #     unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
    #     o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
    #     labels = np.array(o3d_pc.cluster_dbscan(eps=0.006, min_points=30, print_progress=False))
    pass

#TODO: Create calculateCurvature function.
def calculateCurvature(patch_points, centroid):
    '''

    :param patch_points:
    :param centroid:
    :return:
    Function receives patch and calculates curvature.
    '''
    pass

#TODO: Create calculateEigenvectorsOfPatch function.
def calculateEigenvectorsOfPatch(centroid, patch_points):
    '''

    :param patch_points:
    :param centroid:
    :return:
    Function receives patch of points and return the patch's three eigenvectors.
    Note: smallest eigenvector is the normal of the surface.
    '''
    pass

def density_ratio(point_cloud, centroid, radius=0.05):
    '''
    Calculate the density ratio of points within a specified radius around a centroid.

    :param point_cloud: PyTorch tensor of shape (num_points, 3)
    :param centroid: PyTorch tensor of shape (3,)
    :param radius: Radius to consider for density calculation
    :return: Density ratio
    '''
    # Calculate the Euclidean distance between each point and the centroid
    distances = torch.norm(point_cloud - centroid, dim=0)

    # Count the number of points within the specified radius
    num_points_within_radius = torch.sum(distances < radius).item()

    # Calculate the total number of points in the point cloud
    total_points = point_cloud.shape[1]

    # Calculate the density ratio
    density_ratio = num_points_within_radius / total_points

    return density_ratio

def calculateSurfaceVarianceAndEigens(centroid, patch_points):
    '''

    :param centroid:
    :param patch_points:
    :return:
    '''
    full_patch = patchWithCentroid(centroid, patch_points)
    covariance = np.cov(full_patch, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    arr = [(val, vec) for val,vec in zip(eigenvalues, eigenvectors)]
    eigens = sorted(arr, key=lambda x: x[0])
    eigens = [np.append(x[0],x[1]) for x in eigens]
    eigens = [item for sublist in eigens for item in sublist]
    surfaceVariance = np.max(eigenvalues) / (np.sum(eigenvalues))
    eigens.append(surfaceVariance)
    return eigens
def patchWithCentroid(centroid, patch_points):
    '''
    :param centroid: PyTorch tensor of shape (3,)
    :param patch_points: PyTorch tensor of shape (num_of_nn, 3)
    :return: patch of shape (num_of_nn + 1, 3) where the first point is the centroid (PyTorch tensor)
    '''
    if patch_points.shape[1] != 3:
        patch_points = patch_points.reshape(-1, 3)
    centroid = centroid.reshape(1, 3)
    full_patch = torch.cat((centroid, patch_points), dim=0)
    return full_patch.cpu().numpy()
#TODO: Maybe use non linear function in persistence image such that H1 "strong holes" will have high values.
def calculatePersistentHomology( centroid, patch_points, pixel_size=0.1, birth_range=(0.0, 1.0), pers_range=(0.0, 1.0) ):
    '''

    :param centroid: Centroid of patch (type: numpy array).
    :param patch_points: Neighbors of centroid in patch (type: Vector3dVector).
    :param pixel_size: Size of pixel in persistence image.
    :param birth_range: X-axis of peristent image (type: tuple).
    :param pers_range: Y-axis of peristent image (type: tuple).
    :return: A list of two persistent images of H0, H1 respectively.
    '''

    with open(os.devnull, 'w') as f:
        with redirect_stdout(f):
            rips = Rips()
    full_patch = patchWithCentroid(centroid, patch_points)
    pdgms = rips.fit_transform(full_patch)

    pdgms[0] = pdgms[0][0:-1,:]
    pimgr = PersistenceImager(pixel_size=pixel_size, birth_range=birth_range, pers_range=pers_range)
    pimgs = pimgr.transform(pdgms, True)
    return pimgs

#TODO: Create calculateFPFH function.
def calculateFPFH(patch_points, centroid, max_nn = 20):
    '''

    :param patch_points:
    :param centroid:
    :return: FPFH
    '''
    full_patch = patchWithCentroid(centroid, patch_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(full_patch)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamKNN(knn=max_nn))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamKNN(knn = max_nn))

    return pcd_fpfh.data