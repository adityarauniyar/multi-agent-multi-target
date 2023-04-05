from utils import getObstacleMapFromPointcloud, savePNGFromNumpy, displayMap, saveNumpyAsNPY
from ENV import POINTCLOUD_FILENAME, GRID_SIZE, OBSTACLE_MAP_FILENAME, OBSTACLE_MAP_IMG_FILENAME

if __name__ == "__main__":

    obstacle_map = getObstacleMapFromPointcloud(POINTCLOUD_FILENAME, GRID_SIZE)
    #
    saveNumpyAsNPY(obstacle_map, OBSTACLE_MAP_FILENAME)
    #
    displayMap(obstacle_map)
    savePNGFromNumpy(obstacle_map, OBSTACLE_MAP_IMG_FILENAME)

    # selectAreasOnImage(OBSTACLE_MAP_IMG_FILENAME)

