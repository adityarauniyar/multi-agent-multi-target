from PIL import Image
import numpy as np
from enum import Enum


class ObjectType(Enum):
    BUILDING = 0
    TREE = 1
    STREETLIGHT = 2
    VEHICLE = 3


class SemanticObject:
    """
    SemanticObject class is meant to represent an object in a semantic segmentation image.
    The class stores information such as the object name, the image file name,
    the RGB value of the object in the image, and a look score.
    The class also has methods for reading in the image file,
    extracting the RGB tuples from the image and storing them in a table,
    and creating a hash table indicating whether the object is present at each pixel location.
    """

    def __init__(self,
                 object_name: ObjectType,
                 image_filename: str,
                 semantic_rgb_val: tuple,
                 look_score: int):
        """
        Initializes a SemanticObject instance.

        :param object_name: Name of the semantic object to be assigned.
        :param image_filename: Relative location of the semantic segmented image.
        :param semantic_rgb_val: RGB value of the object in the given image.
        :param look_score: The look score to be given when the drone sees this object's pixels.
        """
        self.object_name = object_name
        self.image_filename = image_filename
        self.semantic_rgb_tuple = semantic_rgb_val
        self.reward = look_score

        # Read the image and store its dimensions as instance attributes.
        self.image = self.__read_image()
        self.image_height = self.image.size[1]
        self.image_width = self.image.size[0]

        # Convert the image to a numpy array and store it as an instance attribute.
        self.image_numpy = np.asarray(self.image)

        # Create a table to store the RGB tuples of each pixel in the image.
        # Each entry of the table is a tuple of three unsigned 8-bit integers.
        self.rgb_image_tuple_table = self.__get_rgb_tuple_table_from_image()

        # Create a hash table to store whether each pixel contains the object of interest.
        # The hash table maps (i,j) coordinates to True if the corresponding pixel contains
        # the object of interest, and False otherwise.
        self.presence_grid = self.__get_object_presence_table()

    def __read_image(self):
        """
        Private method that reads the image file and returns an Image object.
        """
        return Image.open(self.image_filename)

    def __get_rgb_tuple_table_from_image(self):
        """
        Private method that creates a table of size image height x width to store the RGB tuples
        of each pixel in the image.
        """
        # Create a table of size image height x width to store the RGB tuples.
        rgb_table = np.zeros((self.image_height, self.image_width), dtype=(np.uint8, 3))

        # Copy the RGB tuples from the image to the table.
        for i in range(self.image_height):
            for j in range(self.image_width):
                # Below code can be used to know what RGB values correspond to objects
                # if self.image_numpy[i, j][2] not in [127, 36, 76, 255]:
                #     print(f"({i}, {j}) gives {self.image_numpy[i, j]}")
                rgb_table[i, j] = tuple(self.image_numpy[i, j][:3])

        return rgb_table

    def __get_object_presence_table(self):
        """
        Private method that creates a hash table to store whether each pixel contains the
        object of interest.
        """
        object_location_arr = np.zeros((self.image_height, self.image_width))

        # Iterate over each pixel in the image.
        for i in range(self.image_height):
            for j in range(self.image_width):
                # If the RGB tuple of the pixel matches the semantic RGB value of the object,
                # store True in the hash table. Otherwise, store False.
                if tuple(self.rgb_image_tuple_table[i, j]) == self.semantic_rgb_tuple:
                    object_location_arr[i, j] = 1

        return object_location_arr
