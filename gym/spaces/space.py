import numpy as np
from gym.utils.types import ThreeIntTuple, AgentType
import logging


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
            grid_size: float = 1.0,
            operational_map: np.ndarray = np.ones((3, 3)),
            start_position: ThreeIntTuple = (0, 0, 0),
            goal_position: ThreeIntTuple = (0, 0, 0),
            agent_type: AgentType = AgentType.DRONE,
            agent_id: int = 0,
    ):
        """
        Initializes the attributes of the Space class.

        Parameters:
        * grid_size: float, the size of the grid.
        * operational_map: array, the size of the 2D grid environment.
        * start_position: tuple (x,y, orientation), the starting position of the drone .
        """
        self.agent_type = agent_type
        self.agent_id = agent_id

        self.grid_size = grid_size
        self.operational_map = operational_map
        self.map_height = np.shape(operational_map)[0]  # Obstacle map height
        self.map_width = np.shape(operational_map)[1]  # Obstacle map width

        self.start_position = start_position
        self.previous_position = start_position
        self.current_position = start_position  # current position
        self.goal_position = goal_position

        self.translation_dirs = [(0, 0),  # 0: stay in place
                                 (1, 0),  # 1: east
                                 (-1, 0),  # 2: west
                                 (0, 1),  # 3: north
                                 (0, -1),  # 4: south
                                 (1, 1),  # 5: north-east
                                 (-1, 1),  # 6: north-west
                                 (1, -1),  # 7: south-east
                                 (-1, -1)]  # 8: south-west

        self.total_translation_dirs = len(self.translation_dirs)

        # rotational directions in degrees
        self.rotation_dirs = [45, 90, 135, 180,  # Rotations in left, counter-clockwise as positive
                              -45, -90, -135, -180]  # Rotations in right, clockwise as negative
        self.total_rotation_dirs = len(self.rotation_dirs)

        self.logger = logging.getLogger(__name__)

    def get_new_translated_position_by_seq(self, translation_sequence: int) -> ThreeIntTuple:
        """
        Returns the new position of the drone after a translation move.

        Parameters:
        * current_location: tuple, the current position of the drone.
        * translation_sequence: int, the sequence of the drone's translation move.

        Returns:
        * A tuple of the new position of the drone.
        """

        new_x = self.current_position[0] + self.translation_dirs[translation_sequence][0]
        new_y = self.current_position[1] + self.translation_dirs[translation_sequence][1]
        new_position = (new_x, new_y, self.current_position[2])

        return new_position

    def get_new_rotated_position_by_seq(self, rotation_sequence: int) -> ThreeIntTuple:
        """
        Returns the new orientation of the drone after a rotation move.

        Parameters:
        * current_orientation: int, the current orientation of the drone.
        * rotation_sequence: int, the sequence of the drone's rotation move.

        Returns:
        * The new orientation of the drone.
        """
        new_orientation = self.current_position[2] + self.rotation_dirs[rotation_sequence]
        new_position = (self.current_position[0], self.current_position[1], new_orientation)

        return new_position

    def is_valid_action(self, new_position: ThreeIntTuple) -> bool:
        """
        Checks if the new position is a valid action or not.

        Parameters:
        * new_position: tuple, the new position to be checked.

        Returns:
        * True if the new position is an inside the operational map dimension and is on the operational location

        """
        is_valid = True

        if not 0 <= new_position[0] < self.map_height or not 0 <= new_position[1] < self.map_width:
            self.logger.debug(f"(Agent: {self.agent_type}, ID: {self.agent_id}) New position({new_position}), outside "
                              f"the map.")
            is_valid = False
        elif new_position[2] % 45 != 0:
            self.logger.debug(f"(Agent: {self.agent_type}, ID: {self.agent_id}) New position({new_position}) "
                              f"orientation is not valid.")
            is_valid = False
        elif not self.operational_map[new_position[0]][new_position[1]]:
            self.logger.debug(f"(Agent: {self.agent_type}, ID: {self.agent_id}) New position({new_position}) has "
                              f"obstacle on it.")
            is_valid = False

        return is_valid

    def move_to(self, new_position: ThreeIntTuple) -> bool:
        """
        Update the current state of the drone with a new position and orientation.

        Args:
        - new_position: tuple, the new position of the drone (x, y)
        - orientation: float, the new orientation of the drone in degrees

        Returns:
        - None
        """
        # Update the current position with the new position
        success = False
        if self.is_valid_action(new_position):
            self.previous_position = self.current_position
            self.current_position = new_position
            success = True
            self.logger.info(
                f"(Agent: {self.agent_type}, ID: {self.agent_id})  moved to new position: {self.current_position}")

        else:
            self.logger.warning("Cannot move to new position({})".format(new_position))
        return success

    def update_goal_position(self, new_goal_position: ThreeIntTuple) -> bool:
        success = False
        if self.is_valid_action(new_goal_position):
            self.goal_position = new_goal_position
            success = True
            self.logger.info(
                f"(Agent: {self.agent_type}, ID: {self.agent_id})  GOAL moved to new position: {self.current_position}")
        else:
            self.logger.warning("Cannot move GOAL to new position({})".format(new_goal_position))
        return success
