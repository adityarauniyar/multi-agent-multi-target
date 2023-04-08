from spaces import Space
import numpy as np


class DroneSpace(Space):
    """
    Drone Space Class to capture the behaviour of the drones moving given 2D grid.

    It inherits most of the action spaces from the Space Parent class.
    """

    def __int__(
            self,
            grid_size=1.0,
            operational_map=None,
            start_position=(0, 0, 0),
            viewing_angle=90.0,
            viewing_range=15.0
    ):
        """
        :param grid_size: float, the size of the grid.
        :param operational_map: array, the size of the 2D grid environment.
        :param start_position: tuple (x,y, orientation), the starting position of the drone.
        :param viewing_angle: int, the angle of the drone's camera view.
        :param viewing_range: int, the range of the drone's camera view.
        """
        super(DroneSpace, self).__init__(grid_size, operational_map, start_position)

        self.viewing_angle = viewing_angle  # angle of the drone's camera view
        self.viewing_range = viewing_range  # range of the drone's camera view

    @property
    def current_cam_coverage_locations(self):
        # Return the camera coverage locations based on the camera location, angle and camera range
        pass

    def get_camera_coverage_definition(self):
        """

        :return: x, y, radius, arc_angle_in_deg
        """
        circle_center_x = self.current_position[0]
        circle_center_y = self.current_position[1]

        circle_radius = self.viewing_range

        circle_arc_in_deg = self.viewing_angle

        arc_centerline_angle_deg = self.current_position[2]

        return circle_center_x, circle_center_y, circle_radius, circle_arc_in_deg, arc_centerline_angle_deg
