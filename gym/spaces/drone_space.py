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
        # TODO: Use the optimal height and angle for filming actors paper to come up with the viewing range.
        #  Viewing range instead can be replace with the Lens description like MP or Focal Length.
        self.viewing_range = viewing_range  # range of the drone's camera view

    @property
    def current_cam_coverage_locations(self):
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
                    self.camera_coverage_grid[i, j] = True
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
