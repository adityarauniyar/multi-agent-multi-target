import unittest
import numpy as np
from typing import Tuple, List
from gym.spaces.space import Space


class TestSpace(unittest.TestCase):

    def setUp(self):
        self.space = Space(grid_size=1.0, operational_map=np.ones((3, 3)), start_position=(0, 0, 0))

    def test_init(self):
        self.assertEqual(self.space.grid_size, 1.0)
        self.assertEqual(self.space.operational_map.shape, (3, 3))
        self.assertEqual(self.space.start_position, (0, 0, 0))
        self.assertEqual(self.space.current_position, (0, 0, 0))
        self.assertEqual(self.space.total_translation_dirs, 9)
        self.assertEqual(self.space.total_rotation_dirs, 8)

    def test_translation_move(self):
        self.assertEqual(self.space.translation_move((0, 0, 0), 0), (0, 0))
        self.assertEqual(self.space.translation_move((0, 0, 0), 1), (1, 0))
        self.assertEqual(self.space.translation_move((0, 0, 0), 2), (-1, 0))
        self.assertEqual(self.space.translation_move((0, 0, 0), 3), (0, 1))
        self.assertEqual(self.space.translation_move((0, 0, 0), 4), (0, -1))
        self.assertEqual(self.space.translation_move((0, 0, 0), 5), (1, 1))
        self.assertEqual(self.space.translation_move((0, 0, 0), 6), (-1, 1))
        self.assertEqual(self.space.translation_move((0, 0, 0), 7), (1, -1))
        self.assertEqual(self.space.translation_move((0, 0, 0), 8), (-1, -1))

    def test_rotation_move(self):
        self.assertEqual(self.space.rotation_move(0, 0), 45)
        self.assertEqual(self.space.rotation_move(0, 1), 90)
        self.assertEqual(self.space.rotation_move(0, 2), 135)
        self.assertEqual(self.space.rotation_move(0, 3), 180)
        self.assertEqual(self.space.rotation_move(0, 4), -45)
        self.assertEqual(self.space.rotation_move(0, 5), -90)
        self.assertEqual(self.space.rotation_move(0, 6), -135)
        self.assertEqual(self.space.rotation_move(0, 7), -180)

    def test_is_valid_action(self):
        self.assertTrue(self.space.is_valid_action((0, 0, 0)))
        self.assertTrue(self.space.is_valid_action((1, 1, 0)))
        self.assertFalse(self.space.is_valid_action((-1, -1, 0)))
        self.assertFalse(self.space.is_valid_action((3, 3, 0)))

    def test_update_state(self):
        self.space.update_state((1, 1), 45)
        self.assertEqual(self.space.current_position, (1, 1, 45))
        self.space.update_state((2, 2), -90)
        self.assertEqual(self.space.current_position, (2, 2, -90))
