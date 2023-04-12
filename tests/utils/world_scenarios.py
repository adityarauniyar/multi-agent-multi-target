import numpy as np

OPERATIONAL_MAP1 = np.array(
    [[0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
     [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
     [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  #
     [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  #
     [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
     [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
     [1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
     [1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
     [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
     [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])

OPERATIONAL_MAP2 = np.array(
    [[1, 1, 1, 0, 1, 0, 0, 1, 1, 0],
     [1, 1, 1, 0, 1, 0, 0, 1, 1, 0],
     [1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
     [1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
     [1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
     [1, 1, 1, 0, 1, 0, 0, 1, 1, 0],
     [1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
     [1, 1, 1, 0, 1, 0, 0, 1, 1, 0],
     [1, 1, 1, 0, 1, 0, 0, 1, 1, 0],
     [1, 1, 1, 0, 1, 0, 0, 1, 1, 1]])

DRONE1_LOCATION1 = (3, 3, 0)
DRONE2_LOCATION1 = (4, 0, 0)
DRONE3_LOCATION1 = (np.shape(OPERATIONAL_MAP1)[0], np.shape(OPERATIONAL_MAP1)[1], 0)

DRONE_POSITIONS = [(DRONE1_LOCATION1[0], DRONE1_LOCATION1[1]),
                   (DRONE2_LOCATION1[0], DRONE2_LOCATION1[1]),
                   (DRONE3_LOCATION1[0], DRONE3_LOCATION1[1])]

ACTOR1_LOCATION1 = (5, 7, 0)
ACTOR2_LOCATION1 = (3, 7, 0)
ACTOR3_LOCATION1 = (9, 7, 0)

ACTOR_POSITIONS = [(ACTOR1_LOCATION1[0], ACTOR1_LOCATION1[1]),
                   (ACTOR2_LOCATION1[0], ACTOR2_LOCATION1[1]),
                   (ACTOR3_LOCATION1[0], ACTOR3_LOCATION1[1])]

# noinspection DuplicatedCode
OBS_MAP1_DRONE1_LOC1 = np.array([
    [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Obstacle Map
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
     [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
     [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
     [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
     [1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
     [1, 1, 0, 0, 0, 1, 0, 0, 0, 0]],

    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Agents (DRONES + ACTORS)Position Map
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Actors Position Projected Map (Drone goals positions)
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # This is without projection
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Contains all the actor location even the one that current drone is following
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],

    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Actors Position Projected Map (Drone goals positions)
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # This is without projection
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
])


def invert_array(arr):
    return np.where(arr == 0, 1, 0)


if __name__ == "__main__":
    print(invert_array(OPERATIONAL_MAP1))