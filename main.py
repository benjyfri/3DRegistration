from DeepBBS.geo_util import *
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from pyvista import examples
import open3d as o3d
import faiss
import random
from persim import PersistenceImager
from utils.geo_util import sample
def visualize_patches():
    ply_point_cloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
    # Call the sample function to obtain centroids, distances_arr, and neighbor_point_clouds
    centroids, distances_arr, neighbor_point_clouds = sample(pcd, num_of_centroids=1000, num_of_point_in_each_patch=55)

    # Create a list of geometries for visualization (centroid and its neighbors)
    geometries = []

    for centroid, neighbor_pcd in zip(centroids.points, neighbor_point_clouds):
        # Set the color of the point cloud to red
        color = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
        neighbor_pcd.paint_uniform_color(color)

        # Append the centroid and its colored neighbor point cloud to the geometries list
        geometries.append(neighbor_pcd)

    centroids.paint_uniform_color([1, 0, 0])  # Strong red color for centroids
    geometries.append(centroids)
    # Visualize the geometries
    o3d.visualization.draw_geometries(geometries,
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
def tryPersistentHomology():
    ply_point_cloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
    # Call the sample function to obtain centroids, distances_arr, and neighbor_point_clouds
    centroids, distances_arr, neighbor_point_clouds = sample(pcd, num_of_centroids=10, num_of_point_in_each_patch=55)
    persistent_images =[]
    for centroid, patch in zip(centroids.points, neighbor_point_clouds):
        persistent_images.append(calculatePersistentHomology(centroid, patch.points))
    plot_persistent_images(persistent_images)
def plot_persistent_images(persistent_images):
    num_images = len(persistent_images)

    fig, axs = plt.subplots(num_images, 2, figsize=(8, 2 * num_images))

    pimgr = PersistenceImager(pixel_size=0.5)
    for i in range(num_images):
        H_0, H_1 = persistent_images[i]

        ax1 = axs[i, 0]
        ax1.set_title("H_0")
        pimgr.plot_image(H_0, ax=ax1)

        ax2 = axs[i, 1]
        ax2.set_title("H_1")
        pimgr.plot_image(H_1, ax=ax2)

    plt.tight_layout()
    plt.show()
def showMeWhatYouGot():

    # Sample data
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    # Create a new figure and axis
    plt.figure()

    # Create a line plot
    plt.plot(x, y, marker='o', linestyle='-', color='b', label='Data Points')

    # Adding labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Simple Line Plot')

    # Adding a legend
    plt.legend()

    # Display the plot
    plt.show()
def testDensity():
    ply_point_cloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
    # Call the sample function to obtain centroids, distances_arr, and neighbor_point_clouds
    centroids, distances_arr, neighbor_point_clouds = sample(pcd, num_of_centroids=10, num_of_point_in_each_patch=55)
    eigens = []
    for centroid, patch in zip(centroids.points, neighbor_point_clouds):
        eigens.append(calculateFPFH(torch.tensor(pcd.points), torch.tensor(centroid),  0.05))
    print(eigens)
def readData():
    # ply_path = "G://My Drive//3DRegistration//3dmatch//train//7-scenes-chess//fragments//cloud_bin_0.ply"
    # ply = o3d.io.read_point_cloud(ply_path)
    # print(ply)
    pass
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    testDensity()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
