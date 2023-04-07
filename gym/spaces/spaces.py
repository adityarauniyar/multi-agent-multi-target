import numpy as np


class Space:
    """
    SpaceClass contains Observation and Action spaces.

    In gym, spaces are crucial to defining the format of legal actions and observations. They have several functions:
    * They allow us to work with highly structured data and transform it into flat arrays that can be used
        in learning code.
    * They provide a method for sampling random elements.
    * They clearly define how to interact with environments, i.e., they specify what actions need to look like
        and what observations will look like. This is extremely helpful for investigation and troubleshooting.
    """

    def __init__(
            self,
            grid_size=1.0,
            operational_map=None,
            start_position=(0, 0, 0),
            viewing_angle=90,
            viewing_range=15
    ):
        """
        Initializes the attributes of the Space class.

        Parameters:
        * grid_size: float, the size of the grid.
        * operational_map: array, the size of the 2D grid environment.
        * start_position: tuple (x,y, orientation), the starting position of the drone .
        * viewing_angle: int, the angle of the drone's camera view.
        * viewing_range: int, the range of the drone's camera view.
        """
        self.grid_size = grid_size
        self.operational_map = operational_map
        self._map_height = np.shape(operational_map)[0]  # Obstacle map height
        self._map_width = np.shape(operational_map)[1]  # Obstacle map width
        self.start_position = start_position
        self.current_position = start_position  # current position
        self.viewing_angle = viewing_angle  # angle of the drone's camera view
        self.viewing_range = viewing_range  # range of the drone's camera view
        self.translation_dirs = [(0, 0),  # stay in place
                                 (1, 0), (-1, 0),  # move north/south/east/west
                                 (0, 1), (0, -1),
                                 (1, 1), (-1, 1),  # move diagonally
                                 (1, -1), (-1, -1)]

        self.total_translation_dirs = len(self.translation_dirs)

        # rotational directions in degrees
        self.rotation_dirs = [45, 90, 135, 180,  # Rotations in right, clockwise as positive
                              -45, -90, -135, -180]  # Rotations in left, counter-clockwise as negative
        self.total_rotation_dirs = len(self.rotation_dirs)

    def translation_move(self, current_location, translation_sequence):
        """
        Returns the new position of the drone after a translation move.

        Parameters:
        * current_location: tuple, the current position of the drone.
        * translation_sequence: int, the sequence of the drone's translation move.

        Returns:
        * A tuple of the new position of the drone.
        """
        return current_location[0] + self.translation_dirs[translation_sequence][0], \
            current_location[1] + self.translation_dirs[translation_sequence][1]

    def rotation_move(self, current_orientation, rotation_sequence):
        """
        Returns the new orientation of the drone after a rotation move.

        Parameters:
        * current_orientation: int, the current orientation of the drone.
        * rotation_sequence: int, the sequence of the drone's rotation move.

        Returns:
        * The new orientation of the drone.
        """
        return current_orientation + self.rotation_dirs[rotation_sequence]

    def is_valid_action(self, new_position):
        """
        Checks if the new position is a valid action or not.

        Parameters:
        * new_position: tuple, the new position to be checked.

        Returns:
        * True if the new position is a inside the operational map dimension and is on the operational location

        """
        return 0 <= new_position[0] < self._map_height and \
            0 <= new_position[1] < self._map_width and \
            self.operational_map[(new_position[0], new_position[1])]

    def update_state(self, new_position, orientation):
        """
        Update the current state of the drone with a new position and orientation.

        Args:
        - new_position: tuple, the new position of the drone (x, y)
        - orientation: float, the new orientation of the drone in degrees

        Returns:
        - None
        """
        # Update the current position with the new position and orientation
        self.current_position = (new_position[0], new_position[1], orientation)


