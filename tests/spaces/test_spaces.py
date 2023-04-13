import unittest
import logging
import numpy as np
from gym.spaces.space import Space
from gym.spaces.drone_space import DroneSpace, DronesSpace
from gym.utils.types import List, AgentType
from tests.utils.world_scenarios import *

MAP_HEIGHT = 3
MAP_WIDTH = 3

logging.basicConfig(level=logging.INFO)


class TestSpace(unittest.TestCase):

    def setUp(self):
        self.space = Space(grid_size=1.0,
                           operational_map=np.ones((MAP_HEIGHT, MAP_WIDTH)),
                           start_position=(0, 0, 0),
                           agent_type=AgentType.DRONE)

    def test_init(self):
        self.assertEqual(self.space.grid_size, 1.0)
        self.assertTrue(np.array_equal(self.space.operational_map, np.ones((MAP_HEIGHT, MAP_WIDTH))))
        self.assertEqual(self.space.map_height, MAP_HEIGHT)
        self.assertEqual(self.space.map_width, MAP_WIDTH)

        self.assertEqual(self.space.start_position, (0, 0, 0))
        self.assertEqual(self.space.current_position, (0, 0, 0))
        self.assertEqual(self.space.previous_position, (0, 0, 0))
        self.assertEqual(self.space.goal_position, (0, 0, 0))

        self.assertEqual(self.space.total_translation_dirs, 9)
        self.assertEqual(self.space.total_rotation_dirs, 8)

        self.assertEqual(self.space.agent_id, 0)
        self.assertEqual(self.space.agent_type, AgentType.DRONE)

    def test_get_new_translated_position_by_seq(self):
        new_pos = self.space.get_new_translated_position_by_seq(0)
        self.assertEqual(new_pos, (0, 0, 0))

        new_pos = self.space.get_new_translated_position_by_seq(1)
        self.assertEqual(new_pos, (1, 0, 0))

        new_pos = self.space.get_new_translated_position_by_seq(4)
        self.assertEqual(new_pos, (0, -1, 0))

    def test_get_new_rotated_position_by_seq(self):
        new_pos = self.space.get_new_rotated_position_by_seq(0)
        self.assertEqual(new_pos, (0, 0, 45))

        new_pos = self.space.get_new_rotated_position_by_seq(1)
        self.assertEqual(new_pos, (0, 0, 90))

        new_pos = self.space.get_new_rotated_position_by_seq(4)
        self.assertEqual(new_pos, (0, 0, -45))

    def test_is_valid_action(self):
        self.assertTrue(self.space.is_valid_action((0, 0, 0)))
        self.assertTrue(self.space.is_valid_action((1, 1, 0)))
        self.assertFalse(self.space.is_valid_action((-1, -1, 0)))
        self.assertFalse(self.space.is_valid_action((-1, 1, 0)))
        self.assertFalse(self.space.is_valid_action((1, -1, 0)))
        self.assertFalse(self.space.is_valid_action((1, MAP_WIDTH, 0)))
        self.assertFalse(self.space.is_valid_action((MAP_HEIGHT, 1, 0)))
        self.assertFalse(self.space.is_valid_action((0, 0, 1)))
        self.assertFalse(self.space.is_valid_action((0, 0, -1)))

    def test_move_to(self):
        self.assertTrue(self.space.move_to((1, 0, 0)))
        # print(f"Current position: {self.space.current_position}; Past position: {self.space.previous_position}")
        self.assertEqual(self.space.current_position, (1, 0, 0))
        self.assertEqual(self.space.previous_position, (0, 0, 0))

        self.assertFalse(self.space.move_to((-1, 0, 0)))
        self.assertEqual(self.space.current_position, (1, 0, 0))
        self.assertEqual(self.space.previous_position, (0, 0, 0))

        self.assertFalse(self.space.move_to((0, 0, 1)))
        self.assertEqual(self.space.current_position, (1, 0, 0))
        self.assertEqual(self.space.previous_position, (0, 0, 0))

        self.assertTrue(self.space.move_to((1, 0, 0)))
        self.assertEqual(self.space.current_position, (1, 0, 0))
        self.assertEqual(self.space.previous_position, (1, 0, 0))

    def test_update_goal_position(self):
        self.assertTrue(self.space.update_goal_position((1, 0, 0)))
        self.assertEqual(self.space.goal_position, (1, 0, 0))

        self.assertFalse(self.space.update_goal_position((-1, 0, 0)))
        self.assertEqual(self.space.goal_position, (1, 0, 0))

        self.assertFalse(self.space.update_goal_position((0, 0, 1)))
        self.assertEqual(self.space.goal_position, (1, 0, 0))


class TestDroneSpace(unittest.TestCase):

    def setUp(self) -> None:
        self.grid_size = 1.0
        self.operational_map = OPERATIONAL_MAP1
        self.start_position = (0, 0, 0)
        self.goal_position = (0, 0, 0)
        self.viewing_angle = 90.0
        self.viewing_range = 15.0
        self.observation_space_size = 10
        self.agent_id = 0
        self.env = DroneSpace(self.grid_size, self.operational_map, self.start_position, self.goal_position,
                              self.viewing_angle, self.viewing_range, self.observation_space_size, self.agent_id)

    def test_initialization(self):
        self.assertIsInstance(self.env, Space)
        self.assertEqual(self.env.grid_size, self.grid_size)
        np.testing.assert_array_equal(self.env.operational_map, self.operational_map)
        self.assertEqual(self.env.start_position, self.start_position)
        self.assertEqual(self.env.viewing_angle, self.viewing_angle)
        self.assertEqual(self.env.viewing_range, self.viewing_range)
        self.assertEqual(self.env.observation_space_size, self.observation_space_size)
        self.assertEqual(self.env.agent_id, self.agent_id)

    def test_get_camera_coverage_definition(self):
        x, y, radius, arc_angle_in_deg, centerline_angle_deg = self.env.get_camera_coverage_definition()
        self.assertIsInstance(x, int)
        self.assertIsInstance(y, int)
        self.assertIsInstance(radius, float)
        self.assertIsInstance(arc_angle_in_deg, float)
        self.assertIsInstance(centerline_angle_deg, int)

    @unittest.skip("test_current_cam_coverage_locations")
    def test_current_cam_coverage_locations(self):
        # self.seed = seeding.create_seed(0)
        # self.np_random = np.random.RandomState(self.seed)
        coverage = self.env.current_cam_coverage_locations
        self.assertIsInstance(coverage, List)

    def test_get_current_observation_channels(self):
        for i in range(len(DRONE_STATES)):
            print("Testing for drone position at {}".format(DRONE_STATES[i]))
            self.assertTrue(self.env.move_to(DRONE_STATES[i]))
            actual_observation_channels = self.env.get_current_observation_channels(DRONE_POSITIONS,
                                                                                    ACTOR_POSITIONS)
            print("Actual Observation Channel: \n{}".format(actual_observation_channels))
            self.assertIsInstance(actual_observation_channels, np.ndarray)
            self.assertEqual(actual_observation_channels.shape,
                             (4, self.observation_space_size, self.observation_space_size))
            self.assertTrue(np.alltrue(actual_observation_channels == OBS_MAP1_DRONE1_LOCS[i]))


class TestDronesSpace(unittest.TestCase):

    def setUp(self):
        self.grid_size = 1.0
        self.operational_map = OPERATIONAL_MAP1
        self.start_positions = DRONE_STATES
        self.goal_positions = ACTOR_POSITIONS
        self.viewing_angle = 90.0
        self.viewing_range = 15.0
        self.observation_space_size = 10
        self.num_agents = 3
        self.drones_space = DronesSpace(
            self.grid_size, self.operational_map, self.start_positions, self.goal_positions,
            self.viewing_angle, self.viewing_range, self.observation_space_size, self.num_agents)

    def test_initialization(self):
        # test if initialization is successful
        self.assertEqual(len(self.drones_space.drones), self.num_agents)
        for i in range(self.num_agents):
            drone = self.drones_space.drones[i]
            self.assertEqual(drone.grid_size, self.grid_size)
            self.assertTrue(np.alltrue(drone.operational_map == self.operational_map))
            self.assertEqual(drone.start_position, self.start_positions[i])
            self.assertEqual(drone.current_position, self.start_positions[i])
            self.assertEqual(drone.viewing_angle, self.viewing_angle)
            self.assertEqual(drone.viewing_range, self.viewing_range)
            self.assertEqual(drone.observation_space_size, self.observation_space_size)
            self.assertEqual(drone.agent_id, i)

    def test_movement(self):
        # test if drone moves correctly
        agent_id = 0
        new_position_with_obstacle = (1, 0, 0)
        drone = self.drones_space.drones[agent_id]
        self.assertFalse(drone.move_to(new_position_with_obstacle))

        new_position_without_obstacle = (7, 4, 0)
        self.assertTrue(drone.move_to(new_position_without_obstacle))
        self.assertEqual(drone.current_position, new_position_without_obstacle)
        self.assertEqual(drone.start_position, DRONE_STATES[agent_id])


if __name__ == '__main__':
    unittest.main()
