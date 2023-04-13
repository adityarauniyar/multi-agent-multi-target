import unittest
import logging
import numpy as np
from gym.spaces.space import Space
from gym.spaces.drone_space import DroneSpace
from gym.utils.types import List
from tests.utils.world_scenarios import *

MAP_HEIGHT = 3
MAP_WIDTH = 3

logging.basicConfig(level=logging.DEBUG)


class TestSpace(unittest.TestCase):

    def setUp(self):
        self.space = Space(grid_size=1.0,
                           operational_map=np.ones((MAP_HEIGHT, MAP_WIDTH)),
                           start_position=(0, 0, 0))

    def test_init(self):
        self.assertEqual(self.space.grid_size, 1.0)
        self.assertTrue(np.array_equal(self.space.operational_map, np.ones((MAP_HEIGHT, MAP_WIDTH))))
        self.assertEqual(self.space.map_height, MAP_HEIGHT)
        self.assertEqual(self.space.map_width, MAP_WIDTH)
        self.assertEqual(self.space.start_position, (0, 0, 0))
        self.assertEqual(self.space.current_position, (0, 0, 0))
        self.assertEqual(self.space.total_translation_dirs, 9)
        self.assertEqual(self.space.total_rotation_dirs, 8)

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
        self.assertEqual(self.space.current_position, (1, 0, 0))

        self.assertFalse(self.space.move_to((-1, 0, 0)))
        self.assertEqual(self.space.current_position, (1, 0, 0))

        self.assertFalse(self.space.move_to((0, 0, 1)))
        self.assertEqual(self.space.current_position, (1, 0, 0))


class TestDroneSpace(unittest.TestCase):

    def setUp(self) -> None:
        self.grid_size = 1.0
        self.operational_map = OPERATIONAL_MAP1
        self.start_position = (0, 0, 0)
        self.viewing_angle = 90.0
        self.viewing_range = 15.0
        self.observation_space_size = 10
        self.agent_id = 0
        self.env = DroneSpace(self.grid_size, self.operational_map, self.start_position, self.viewing_angle,
                              self.viewing_range, self.observation_space_size, self.agent_id)

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
        self.seed = seeding.create_seed(0)
        self.np_random = np.random.RandomState(self.seed)
        coverage = self.env.current_cam_coverage_locations
        self.assertIsInstance(coverage, List)

    def test_get_current_observation_channels(self):
        self.assertTrue(self.env.move_to(DRONE1_LOCATION1))
        actual_observation_channels = self.env.get_current_observation_channels(DRONE_POSITIONS,
                                                                                ACTOR_POSITIONS)
        print("Actual Observation Channel: \n{}".format(actual_observation_channels))
        self.assertIsInstance(actual_observation_channels, np.ndarray)
        self.assertEqual(actual_observation_channels.shape,
                         (4, self.observation_space_size, self.observation_space_size))
        self.assertTrue(np.alltrue(actual_observation_channels == OBS_MAP1_DRONE1_LOC1))


if __name__ == '__main__':
    unittest.main()
