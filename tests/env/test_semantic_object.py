import unittest
from PIL import Image
from gym.envs.objects.semantic_object import SemanticObject, ObjectType
import numpy as np


class TestSemanticObject(unittest.TestCase):

    def setUp(self):
        self.object_name = ObjectType.BUILDING
        self.image_filename = "obstacle-map-with-segmentation.png"
        self.semantic_rgb_val = (127, 127, 127)
        self.look_score = 5
        self.building = SemanticObject(self.object_name, self.image_filename, self.semantic_rgb_val,
                                       self.look_score)

    def test_object_attributes(self):
        self.assertEqual(self.building.object_name, self.object_name)
        self.assertEqual(self.building.image_filename, self.image_filename)
        self.assertEqual(self.building.semantic_rgb_tuple, self.semantic_rgb_val)
        self.assertEqual(self.building.reward, self.look_score)

    def test_image(self):
        self.assertIsInstance(self.building.image, Image.Image)

    def test_dimensions(self):
        self.assertEqual(self.building.image_height, 266)
        self.assertEqual(self.building.image_width, 324)

    def test_RGB_tuple_table(self):
        self.assertIsInstance(self.building.rgb_image_tuple_table, np.ndarray)
        self.assertEqual(self.building.rgb_image_tuple_table.shape, (266, 324, 3))
        self.assertEqual(self.building.rgb_image_tuple_table[0][0][0], 127)

    def test_object_presence_by_location(self):
        self.assertIsInstance(self.building.presence_grid, np.ndarray)
        self.assertEqual(len(self.building.presence_grid), len(self.building.image_numpy))
        self.assertEqual(self.building.presence_grid[(0, 0)], True)
        self.assertEqual(self.building.presence_grid[(100, 100)], True)
        self.assertEqual(self.building.presence_grid[(200, 200)], False)


if __name__ == '__main__':
    unittest.main()
