import numpy as np
import matplotlib.pyplot as plt


def getObstacleMapFromPointcloud(pointcloud_filename, grid_size):
    """
    :param pointcloud_filename: relative filename of the pointcloud
    :param grid_size: size of grid in meters
    :return: obstacle map with given grid size
    """
    # Load the point cloud from file
    points = np.loadtxt(pointcloud_filename)

    # Calculate the maximum and minimum x and y values
    x_max = np.max(points[:, 0])
    x_min = np.min(points[:, 0])
    y_max = np.max(points[:, 1])
    y_min = np.min(points[:, 1])

    # Calculate the number of rows and columns in the grid
    num_rows = int(np.ceil((y_max - y_min) / grid_size))
    num_cols = int(np.ceil((x_max - x_min) / grid_size))

    # Create the grid
    obstacle_map = np.zeros((num_rows, num_cols))

    # Iterate over the points and fill in the grid
    for point in points:
        x_index = int(np.floor((point[0] - x_min) / grid_size))
        y_index = int(np.floor((point[1] - y_min) / grid_size))
        obstacle_map[y_index, x_index] = 1

    return obstacle_map


def displayMap(grid_map):
    """
    :param grid_map:
    """
    # Plot the grid as an image
    plt.imshow(grid_map, cmap='gray')
    plt.show()
