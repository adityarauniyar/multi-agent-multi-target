from mdgym.utils.map_preprocessing import get_obstacle_map_from_pcd, save_png_from_numpy, display_map, save_numpy_as_npy
from ENV import *
from mdgym.envs.objects import semantic_object as sem


def generateMapAndDisplay():
    obstacle_map = get_obstacle_map_from_pcd(POINTCLOUD_FILENAME, GRID_SIZE)

    save_numpy_as_npy(obstacle_map, OBSTACLE_MAP_FILENAME)

    display_map(obstacle_map)
    save_png_from_numpy(obstacle_map, OBSTACLE_MAP_IMG_FILENAME)


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

