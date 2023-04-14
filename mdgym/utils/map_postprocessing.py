import numpy as np
import networkx as nx
from mdgym.utils.types import *


def invert_array(arr):
    return np.where(arr == 0, 1, 0)


def get_reachable_locations(obstacle_map: np.ndarray, start_loc: TwoIntTuple):
    """
    Returns a list of all the reachable locations from a given start location that are not obstacles.

    Parameters:
    obstacle_map (numpy.ndarray): A 2D numpy array where 1 represents an obstacle and 0 represents no obstacle.
    start_loc (tuple): A tuple representing the starting location (x, y).

    Returns:
    reachable_locs (list): A list of tuples representing all the reachable locations from the start location that are not obstacles.
    """
    x, y = start_loc
    visited = {(x, y)}
    queue = [(x, y)]
    reachable_locs = []

    while queue:
        x, y = queue.pop(0)
        for i, j in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            if 0 <= i < obstacle_map.shape[0] and 0 <= j < obstacle_map.shape[1]:
                if obstacle_map[i][j] == 0 and (i, j) not in visited:
                    visited.add((i, j))
                    queue.append((i, j))
                    reachable_locs.append((i, j))

    return reachable_locs


def get_adj_matrix_from_numpy(obstacle_map: np.ndarray):
    # create the adjacency matrix
    adj_matrix = np.zeros((obstacle_map.size, obstacle_map.size), dtype=int)

    # set the weights of the edges
    for i in range(obstacle_map.shape[0]):
        for j in range(obstacle_map.shape[1]):
            if obstacle_map[i][j] == 0:
                if j < obstacle_map.shape[1] - 1 and obstacle_map[i][j + 1] == 0:
                    adj_matrix[i * obstacle_map.shape[1] + j][i * obstacle_map.shape[1] + j + 1] = 1
                    adj_matrix[i * obstacle_map.shape[1] + j + 1][i * obstacle_map.shape[1] + j] = 1
                if i < obstacle_map.shape[0] - 1 and obstacle_map[i + 1][j] == 0:
                    adj_matrix[i * obstacle_map.shape[1] + j][(i + 1) * obstacle_map.shape[1] + j] = 1
                    adj_matrix[(i + 1) * obstacle_map.shape[1] + j][i * obstacle_map.shape[1] + j] = 1

    # create the grid graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix)

    # set the node attributes for obstacle nodes
    for i in range(obstacle_map.size):
        if obstacle_map.flatten()[i] == 1:
            G.nodes[i]['obstacle'] = True
        else:
            G.nodes[i]['obstacle'] = False

    # set the edge weights based on the Euclidean distance between the nodes
    for u, v in G.edges():
        x1, y1 = divmod(u, obstacle_map.shape[1])
        x2, y2 = divmod(v, obstacle_map.shape[1])
        dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        G[u][v]['weight'] = dist

    print(G.nodes.data())
    print(G.edges.data())

    return G
