from utils.util1 import *
import numpy as np
import trimesh
from trimesh.curvature import discrete_gaussian_curvature_measure, discrete_mean_curvature_measure, sphere_ball_intersection
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from pyvista import examples


mesh = trimesh.creation.icosphere()

def print_hi(name):
    print(name)
    arr = np.array([[0,0,0],[2,2,3],[3,3,3]])

    centroid = np.array([0,0])
    arr = np.random.rand(100, 3)

    a = calculateCurvature(arr, centroid)
    print(a)

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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    arr = np.random.rand(100, 3)
    showMeWhatYouGot()
    # a =discrete_gaussian_curvature_measure(mesh,mesh.vertices, 0)
    # print(type(a))
    # print(a.shape)
    print("Load a ply point cloud, print it, and render it")
    ply_point_cloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(ply_point_cloud.path)
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_plotly([pcd],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])


    print("hey")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
