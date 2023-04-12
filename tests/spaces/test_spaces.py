import unittest
import numpy as np
from typing import Tuple, List
from gym.spaces.space import Space

MAP_HEIGHT = 3
MAP_WIDTH = 3


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


if __name__ == '__main__':
    unittest.main()
