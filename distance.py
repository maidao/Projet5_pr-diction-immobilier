import numpy as np
import math

def compute_distance(lon,lat):
    RK = 6371 # Radius of the earth in km
    RM = 3959 # Radius of the earth in mile
    lon_centre = 4.835701
    lat_centre = 45.767733

    dlon = deg2rad(lon - lon_centre)
    dlat = deg2rad(lat - lat_centre)

    a = np.sin(dlat / 2) * np.sin(dlat / 2) + \
        np.cos(deg2rad(lat_centre)) * np.cos(deg2rad(lat)) * \
        np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = c * RK

    return distance


def deg2rad(deg):
    return deg * (math.pi/180)

