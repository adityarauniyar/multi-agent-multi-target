from utils import getObstacleMapFromPointcloud, savePNGFromNumpy, displayMap, saveNumpyAsNPY
from ENV import POINTCLOUD_FILENAME, \
    GRID_SIZE, OBSTACLE_MAP_FILENAME, \
    OBSTACLE_MAP_IMG_FILENAME, SEMANTIC_SEGMENTED_IMG_FILENAME, \
    BUILDING_RGB_TUPLE, BUILDING_LOOKS_SCORE
from gym.envs.objects import semantic_object as sem


def generateMapAndDisplay():
    obstacle_map = getObstacleMapFromPointcloud(POINTCLOUD_FILENAME, GRID_SIZE)

    saveNumpyAsNPY(obstacle_map, OBSTACLE_MAP_FILENAME)

    displayMap(obstacle_map)
    savePNGFromNumpy(obstacle_map, OBSTACLE_MAP_IMG_FILENAME)


if __name__ == "__main__":
    # Building object created using defined variables
    Building = sem.SemanticObject(sem.ObjectType.BUILDING,
                                  SEMANTIC_SEGMENTED_IMG_FILENAME,
                                  BUILDING_RGB_TUPLE,
                                  BUILDING_LOOKS_SCORE)
