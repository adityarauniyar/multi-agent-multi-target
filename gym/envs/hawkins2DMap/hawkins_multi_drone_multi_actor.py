from gym.utils.map_preprocessing import getObstacleMapFromPointcloud, savePNGFromNumpy, displayMap, saveNumpyAsNPY
from ENV import *
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

    Vehicle = sem.SemanticObject(sem.ObjectType.VEHICLE,
                                 SEMANTIC_SEGMENTED_IMG_FILENAME,
                                 VEHICLE_RGB_TUPLE,
                                 VEHICLE_LOOKS_SCORE)

    Tree = sem.SemanticObject(sem.ObjectType.TREE,
                              SEMANTIC_SEGMENTED_IMG_FILENAME,
                              TREE_RGB_TUPLE,
                              TREE_LOOKS_SCORE)

    Streetlight = sem.SemanticObject(sem.ObjectType.STREETLIGHT,
                                     SEMANTIC_SEGMENTED_IMG_FILENAME,
                                     STREETLIGHT_RGB_TUPLE,
                                     STREETLIGHT_LOOKS_SCORE)

