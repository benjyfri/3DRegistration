from utils.util1 import *
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from pyvista import examples
import open3d as o3d
import faiss
from ripser import Rips
from persim import PersistenceImager
from sklearn import datasets


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
def sample(pcd, num_of_centroids, num_of_point_in_each_patch):
    '''
    :param pcd: original point cloud (type: open3d.cpu.pybind.geometry.PointCloud)
    :param num_of_centroids: amount of subsampled points
    :param num_of_point_in_each_patch: how many neighbors of the chosen centroid should we return
    :return: centroids, NumPy arrays - distances to each neighbor, list of point clouds (without centroids)
    '''
    centroids = pcd.farthest_point_down_sample(num_of_centroids)
    points1 = np.asarray(pcd.points)
    points2 = np.asarray(centroids.points)
    dimension = 3

    index = faiss.IndexFlatL2(dimension)
    index.add(points1)

    num_neighbors = num_of_point_in_each_patch
    distances_arr = []
    neighbor_point_clouds = []
    for centroid in points2:
        distances, indices = index.search(centroid.reshape(1, -1), num_neighbors + 1)  # +1 to exclude the closest point
        nearest_neighbors = points1[indices[0][1:]]  # Exclude the closest point

        distances_arr.append(distances[0][1:])

        neighbor_pcd = o3d.geometry.PointCloud()
        neighbor_pcd.points = o3d.utility.Vector3dVector(nearest_neighbors)
        neighbor_point_clouds.append(neighbor_pcd)

    distances_arr = np.array(distances_arr)
    return centroids, distances_arr, neighbor_point_clouds


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
def calculateEigenvectorsOfPatch(patch_points, centroid):
    '''

    :param patch_points:
    :param centroid:
    :return:
    Function receives patch of points and return the patch's three eigenvectors.
    Note: smallest eigenvector is the normal of the surface.
    '''
    pass


def calculatePersistentHomology( centroid, patch_points, pixel_size=0.2, birth_range=(0.0, 1.0), pers_range=(0.0, 1.0) ):
    '''

    :param centroid: Centroid of patch (type: numpy array)
    :param patch_points: Neighbors of centroid in patch (type: Vector3dVector)
    :param pixel_size: Size of pixel in persistence image
    :param birth_range: X-axis of peristent image (type: tuple)
    :param pers_range: Y-axis of peristent image (type: tuple)
    :return: A list of two persistent images of H0, H1 respectively.
    '''
    np_patch_points = np.asarray(patch_points)

    rips = Rips()
    full_patch = np.append(np_patch_points , centroid.reshape((1,3)) , axis = 0)
    pdgms = rips.fit_transform(full_patch)

    pdgms[0] = pdgms[0][0:-1,:]
    pimgr = PersistenceImager(pixel_size=pixel_size, birth_range=birth_range, pers_range=pers_range)
    pimgs = pimgr.transform(pdgms, True)
    return pimgs

#TODO: Create calculateFPFH function.
def calculateFPFH(patch_points, centroid):
    '''

    :param patch_points:
    :param centroid:
    :return:
    '''
    pass