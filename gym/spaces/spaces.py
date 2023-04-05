import numpy as np


# Define the action space
class ActionSpace:
    def __init__(self, grid_size, viewing_angle, viewing_range):
        self.grid_size = grid_size  # size of the 2D grid environment
        self.viewing_angle = viewing_angle  # angle of the drone's camera view
        self.viewing_range = viewing_range  # range of the drone's camera view
        self.actions = [(0, 0),  # stay in place
                        (1, 0), (-1, 0),  # move north/south/east/west
                        (0, 1), (0, -1),
                        (1, 1), (-1, 1),  # move diagonally
                        (1, -1), (-1, -1),
                        ('turn_right',), ('turn_left',)]  # yaw rotation

    def sample(self):
        return np.random.choice(self.actions)

    def get_possible_actions(self, position, orientation):
        """
        Returns a list of possible actions for the drone given its current position and orientation.
        """
        possible_actions = []
        for action in self.actions:
            if isinstance(action, tuple):
                # Move action
                new_pos = tuple(map(sum, zip(position, action)))
                if 0 <= new_pos[0] < self.grid_size[0] and 0 <= new_pos[1] < self.grid_size[1]:
                    # Check if new position is within the grid
                    if self.is_visible(position, new_pos, orientation):
                        # Check if the new position is visible from the current position
                        possible_actions.append(action)
            else:
                # Yaw rotation
                new_orientation = (orientation + (1 if action == 'turn_right' else -1) * 45) % 360
                possible_actions.append((action, new_orientation))
        return possible_actions

    def is_visible(self, position, new_pos, orientation):
        """
        Returns True if the new position is visible from the current position, given the drone's orientation.
        """
        dx, dy = new_pos[0] - position[0], new_pos[1] - position[1]
        angle = np.degrees(np.arctan2(dy, dx)) % 360
        angle_diff = (angle - orientation) % 360
        distance = np.sqrt(dx ** 2 + dy ** 2)
        return angle_diff <= self.viewing_angle / 2 and distance <= self.viewing_range
