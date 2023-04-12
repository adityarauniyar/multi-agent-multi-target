# This python file contains all the environment variables to be used by this environment.


# Filename and locations
POINTCLOUD_FILENAME = "pointclouds/(FLOORLESS) UNIFIED_HAWKINS_OUTDOOR_ADJUSTED_CROPPED.txt"
OBSTACLE_MAP_FILENAME = "grid-maps/hawkins_obstacle_map.npy"
OBSTACLE_MAP_IMG_FILENAME = "grid-maps/obstacle-map.png"
SEMANTIC_SEGMENTED_IMG_FILENAME = "grid-maps/obstacle-map-with-segmentation.png"

# 2D Grid Parameters
GRID_SIZE = 1.0

# Object RGP Tuple values
BUILDING_RGB_TUPLE = (127, 127, 127)
VEHICLE_RGB_TUPLE = (237, 28, 36)
TREE_RGB_TUPLE = (34, 177, 76)
STREETLIGHT_RGB_TUPLE = (255, 242, 0)
OPERATIONAL_RGB_VALUE = (255, 255, 255)

# Looks score for objects in the environment
BUILDING_LOOKS_SCORE = 5
VEHICLE_LOOKS_SCORE = 5
TREE_LOOKS_SCORE = 5
STREETLIGHT_LOOKS_SCORE = 5
