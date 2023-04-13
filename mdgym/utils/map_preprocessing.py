import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from enum import Enum


class Color(Enum):
    NO_COLOR = 0
    GREEN = 1
    WHITE = 2
    ORANGE = 3
    YELLOW = 4


def get_obstacle_map_from_pcd(pcd_filename, grid_size):
    """
    :param pcd_filename: relative filename of the pointcloud
    :param grid_size: size of grid in meters
    :return: obstacle map with given grid size
    """
    # Load the point cloud from file
    points = np.loadtxt(pcd_filename)

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

    # TODO: Create coordinates map that stores the relative coordinates of points
    return obstacle_map


def display_map(grid_map):
    """
    :param grid_map:
    """
    # Plot the grid as an image
    plt.imshow(grid_map, cmap='gray')
    plt.show()


def save_png_from_numpy(numpy_array, output_img_filename):
    # convert the numpy array to PIL Image
    img = Image.fromarray((numpy_array * 255).astype(np.uint8), mode='L')

    # save the image as PNG file
    img.save(output_img_filename)


def save_numpy_as_npy(numpy_array, output_filename):
    # save the numpy array in the output_filename
    np.save(output_filename, numpy_array)
