from enum import Enum


class VehicleType(Enum):
    CAR = 1
    VAN = 2
    TRUCK = 3


class LookScore(Enum):
    NOT_GOOD = 0
    GOOD = 1
    WOW = 3


class Vehicle:
    def __init__(self, look_score, location, orientation, vehicle_type):
        self.look_score = look_score
        self.location = location
        self.orientation = orientation
        self.vehicle_type = vehicle_type

    def get_aesthetic_score(self):
        # Determine if the vehicle is aesthetically pleasing based on its features

        return self.look_score

