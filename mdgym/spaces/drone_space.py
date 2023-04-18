from mdgym.spaces.state import State
from math import floor
from mdgym.utils.types import ThreeIntTuple, TwoIntTupleList, Tuple, List, AgentType
import numpy as np
import logging


class DroneAgentState(State):
    """
    Drone Space Class to capture the behaviour of the drones moving given 2D grid.

    It inherits most of the action spaces from the Space Parent class.
    """

    def __init__(
            self,
            grid_size: float = 1.0,
            operational_map: np.ndarray = np.ones((3, 3)),
            start_position: ThreeIntTuple = (0, 0, 0),
            goal_position: ThreeIntTuple = (0, 0, 0),
            viewing_angle: float = 90.0,
            viewing_range: float = 15.0,
            observation_space_size: int = 10,
            agent_id: int = 0
    ):
        """
        :param grid_size: float, the size of the grid.
        :param operational_map: array, the size of the 2D grid environment.
        :param start_position: tuple (x,y, orientation), the starting position of the drone.
        :param viewing_angle: int, the angle of the drone's camera view.
        :param viewing_range: int, the range of the drone's camera view.
        """

        super().__init__(
            grid_size=grid_size,
            operational_map=operational_map,
            start_position=start_position,
            goal_position=goal_position,
            agent_id=agent_id,
            agent_type=AgentType.DRONE
        )

        self.viewing_angle = viewing_angle  # angle of the drone's camera view
        # TODO: Use the optimal height and angle for filming actors paper to come up with the viewing range.
        #  Viewing range instead can be replaced with the Lens description like MP or Focal Length.
        self.viewing_range = viewing_range  # range of the drone's camera view

        self.observation_space_size = observation_space_size

        # self.logger.(vars(self))


    @property
    def current_cam_coverage_locations(self) -> TwoIntTupleList:
        # Return the camera coverage locations based on the camera location, angle and camera range
        circle_center_x, circle_center_y, circle_radius, circle_arc_in_deg, arc_centerline_angle_deg = self.get_camera_coverage_definition()

        arc_centerline_angle_rad = np.deg2rad(arc_centerline_angle_deg)

        # Initialize the camera coverage map to False values
        # TODO: Create a new grid based on the camera range and orientation of the Drone
        camera_coverage_grid = np.zeros((self.map_height, self.map_width), dtype=bool)

        # Iterate over each grid element.
        for i in range(self.map_height):
            for j in range(self.map_width):
                # Calculate the distance between the center and the grid element.
                dx = abs(j - circle_center_x)
                dy = abs(i - circle_center_y)
                dist = np.sqrt(dx ** 2 + dy ** 2)

                # Check if the grid element is within the radius and angle.
                if dist <= circle_radius:
                    camera_coverage_grid[i, j] = True
                    # TODO: Add logic to only mark those pixels as True that falls under the camera viewing angle.
                    # Assuming zero deg of the centerline is facing towards x-axis
                    # grid_angle_in_rad = np.pi / 2 - np.arc-tan2(dy, dx)
                    #
                    # grid_angle_with_centerline_in_rad = grid_angle_in_rad - arc_centerline_angle_rad
                    #
                    # if arc_centerline_angle_rad >= 0:
                    #     if abs(grid_angle_with_centerline_in_rad) <= np.deg2rad(circle_arc_in_deg / 2):
                    #
                    # else:
                    #     if (arc_centerline_angle_rad / 2) + np.pi >= grid_angle_in_rad >= (np.pi / 2) - (
                    #             arc_centerline_angle_rad / 2):
                    #         camera_coverage_grid[i, j] = True

        # Convert the grid to a list of grid locations.
        locations = []
        for i in range(self.map_height):
            for j in range(self.map_width):
                if camera_coverage_grid[i, j]:
                    locations.append((i, j))

        return locations

    def get_camera_coverage_definition(self) -> Tuple[int, int, float, float, int]:
        """

        :return: x, y, radius, arc_angle_in_deg
        """
        circle_center_x = self.current_position[0]
        circle_center_y = self.current_position[1]

        circle_radius = self.viewing_range

        circle_arc_in_deg = self.viewing_angle

        arc_centerline_angle_deg = self.current_position[2]

        return circle_center_x, circle_center_y, circle_radius, circle_arc_in_deg, arc_centerline_angle_deg

    def get_current_observation_channels(
            self,
            current_agents_position_arr: TwoIntTupleList,
            current_actor_position_arr: TwoIntTupleList
    ) -> np.ndarray | None:
        current_actor_x, current_actor_y, current_actor_orientation = self.current_position
        self.logger.debug(f"Current actor position: ({current_actor_x}, {current_actor_y})")

        # Get the observation channel origin
        relative_observation_window_origin_x = floor(current_actor_x - (self.observation_space_size / 2.0))
        relative_observation_window_origin_y = floor(current_actor_y - (self.observation_space_size / 2.0))
        self.logger.debug(f"Relative Observation window origin: ({relative_observation_window_origin_x},"
                          f" {relative_observation_window_origin_y})")

        # Get the observation channel window top coordinates
        relative_observation_window_top_x = min(self.observation_space_size,
                                                int(relative_observation_window_origin_x + self.observation_space_size))
        relative_observation_window_top_y = min(self.observation_space_size,
                                                (relative_observation_window_origin_y + self.observation_space_size))
        self.logger.debug(f"Relative Observation Top Coordinate: ({relative_observation_window_top_x},"
                          f" {relative_observation_window_top_y})")

        # Stores location as True for those that has obstacles
        obstacle_map = np.ones((self.observation_space_size, self.observation_space_size))

        # Stores all the locations as False for those that doesn't have agents on it.
        agents_position_map = np.zeros((self.observation_space_size, self.observation_space_size))

        # Stores all the locations as False by default that doesn't have neighbours goal position projected.
        neighbours_goal_map = np.zeros((self.observation_space_size, self.observation_space_size))

        # Stores all the locations as False by default that doesn't have agents goals, no projection is used to
        # update this
        agent_goal_map = np.zeros((self.observation_space_size, self.observation_space_size))
        self.logger.debug(f"Current actor position: ({current_actor_x}, {current_actor_y})")

        try:
            # TODO: Vectorize this double for-loop
            for y in range(relative_observation_window_origin_y, relative_observation_window_top_y):
                for x in range(relative_observation_window_origin_x, relative_observation_window_top_x):
                    self.logger.debug(f"Traversing over ({x}, {y}) to update observation channel.")
                    # Check if the window is inside the operational map
                    if x < 0 or y < 0:
                        # Continue if this location is outside the main map
                        self.logger.debug(f"Location out of operation map, hence using default values.")
                        continue
                    else:

                        current_map_x = x
                        current_map_y = y
                        self.logger.debug(f"Current map coordinates: ({current_map_x}, {current_map_y})")

                        current_window_x = x - relative_observation_window_origin_x
                        current_window_y = y - relative_observation_window_origin_y
                        self.logger.debug(f"Current Window coordinates: ({current_window_x}, {current_window_y})")

                        if self.operational_map[current_map_x, current_map_y]:
                            obstacle_map[current_window_x, current_window_y] = 0
                            self.logger.debug(f"obstacle_map[{current_window_x}, {current_window_y}] = 0")

                        if (current_map_x, current_map_y) in current_agents_position_arr:
                            agents_position_map[current_window_x, current_window_y] = 1
                            self.logger.debug(f"agents_position_map[{current_window_x}, {current_window_y}] = 1")

                        # QUES: Should we keep track of which drone is going to which actor or just come up with the
                        # actor locations as the drone goal locations

                        if (current_map_x, current_map_y) in current_actor_position_arr:
                            neighbours_goal_map[current_window_x, current_window_y] = 1
                            self.logger.debug(f"neighbours_goal_map[{current_window_x}, {current_window_y}] = 1")

                        # Right now taking all the actor location as .self drone goal location
                        if (current_map_x, current_map_y) in current_actor_position_arr:
                            agent_goal_map[current_window_x, current_window_y] = 1
                            self.logger.debug(f"agent_goal_map[{current_window_x}, {current_window_y}] = 1")

            observation_channel = np.array([obstacle_map, agents_position_map, neighbours_goal_map, agent_goal_map])
        except Exception as err:
            self.logger.error(str(err) + "Something went grond at getting observation channel.")
            observation_channel = None

        return observation_channel


class DronesSpace:
    def __init__(
            self,
            start_positions: List[ThreeIntTuple],
            grid_size: float = 1.0,
            operational_map: np.ndarray = np.ones((3, 3)),
            goal_positions: List[ThreeIntTuple] | None = None,
            viewing_angle: float = 90.0,
            viewing_range: float = 15.0,
            observation_space_size: int = 10,
            num_agents: int = 1
    ):
        self.logger = logging.getLogger(__name__)

        self.drones = []

        self.logger.info(f"Starting the Drone(s)Space with following parameters: num_agents = {num_agents},\n "
                         f"observation Size = {observation_space_size}, world size = {operational_map.size}, "
                         f"grid size ={grid_size},"
                         f" Actors position length = {len(goal_positions) if goal_positions is not None else 0}, "
                         f" Drones position length = {len(start_positions) if start_positions is not None else 0}, ")

        for agentID in range(num_agents):
            self.drones.append(DroneAgentState(
                grid_size=grid_size,
                operational_map=operational_map,
                start_position=start_positions[agentID],
                goal_position=goal_positions[agentID] if goal_positions is not None else None,
                viewing_angle=viewing_angle,
                viewing_range=viewing_range,
                observation_space_size=observation_space_size,
                agent_id=agentID))
