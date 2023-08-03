import open3d as o3d
import numpy as np

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

#TODO: Create sample function. Use FPS or later on gridGCN algorithm
def sample(all_points, num_of_centroids):
    '''
    :param all_points:
    :param num_of_centroids:
    :return:
    function receives all points and amount of centroids to create and returns centroids
    '''
    pass

#TODO: Create calculateDensityOfPatch function.
def calculateDensityOfPatch(patch_points, centroid, radius):
    '''
    :param patch_points:
    :param centroid:
    :param radius:
    :return:
    function receives patch points and centroid and radius and returns the inverse of the amount of points that
    are within the radius from the centroid.
    Maybe do this only with centroids and other points and not all points with all points like in original paper.
    MAYBE this should be calculated already as part of k_NN??
    '''
    pass

#TODO: Create calculateCurvature function.
def calculateCurvature(patch_points, centroid):
    '''

    :param patch_points:
    :param centroid:
    :return:
    Function receives patch and calculates curvature.
    '''
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(patch_points)

    # Define the neighborhood search radius for curvature estimation
    radius = 0.1  # Adjust this value based on your point cloud density

    # Calculate curvature using Open3D
    curvature = point_cloud.compute_point_cloud_curvature(radius)

    # Access the curvature values and their indices in the point cloud
    curvature_values = np.asarray(curvature[0])
    curvature_indices = np.asarray(curvature[1])
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
#TODO: Create calculatePersistentHomology function.
def calculatePersistentHomology(patch_points, centroid):
    '''

    :param patch_points:
    :param centroid:
    :return:
    Function receives patch of points and return the patch's persistent homology.
    '''
    pass