class StreetLight:
    def __init__(self, pixel_loc, look_score):
        """
        :param pixel_loc:
        :param look_score: use the Enum class to assign score for looks
        """
        self.pixel_loc = pixel_loc
        self.look_score = look_score
        self.aesthetic_score = self.calculate_aesthetic_score()

    def calculate_aesthetic_score(self):
        # Calculate the aesthetic score of the building using its pixel location goes here
        return self.look_score
