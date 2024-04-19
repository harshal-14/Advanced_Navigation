import numpy as np
from typing import List, NamedTuple, Union

class PixelCoordinate(NamedTuple):
    x: int
    y: int

class Coordinate(NamedTuple):
    x: float
    y: float
    z: float


class Marker(NamedTuple):
    """
    Markers are named tuples with the an id field,
    followed by fields p1 through p4, wherein they
    represent the coordinate locations of each corner
    of the april tag
    """

    id: str
    tl: Union[Coordinate, PixelCoordinate] # top left
    tr: Union[Coordinate, PixelCoordinate] # top right
    bl: Union[Coordinate, PixelCoordinate] # bottom left
    br: Union[Coordinate, PixelCoordinate] # bottom right

class Data(NamedTuple):

    img: np.ndarray
    tags: List[Marker]
    timestamp: float
    rpy: np.ndarray
    acc: np.ndarray
    omg: np.ndarray


class ActualData(NamedTuple):
    """
    ActualData is data read from our motion capture system
    with the following format:
    timestamp - the time in seconds of the measurement
    x, y, z - the position of the drone in meters
    roll, pitch, yaw - the orientation of the drone in radians
    vx, vy, vz - the velocity of the drone in m/s per axis
    wx, wy, wz - the angular velocity of the drone in rad/s per
        axis
    """

    timestamp: float
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    vx: float
    vy: float
    vz: float
    wx: float
    wy: float
    wz: float


def rmse(array1, array2):
    return np.sqrt(np.mean((array1 - array2) ** 2))

def interpolate_ground_truth(ground_truth_data: List[ActualData], estimation: Data):

    # Get the current timestep from the estimation
    timestamp = estimation.timestamp

    # Find the first ground truth time that is past that timestamp
    data_point1 = None
    data_point2 = None
    for idx in range(1, len(ground_truth_data)):  # Start at index 1 to safely access index - 1
        if ground_truth_data[idx].timestamp > timestamp:
            data_point1 = ground_truth_data[idx - 1]
            data_point2 = ground_truth_data[idx]
            break

    if data_point1 is None or data_point2 is None:
        raise ValueError("Timestamp doesn't map to any ground truth data.")

    total_change = data_point2.timestamp - data_point1.timestamp

    weight1 = (timestamp - data_point1.timestamp) / total_change
    weight2 = (data_point2.timestamp - timestamp) / total_change

    # Finally taking weighted average between data_point1 and data_point2 given our percentages as weights
    data_point1_vec = np.array([data_point1.x, data_point1.y, data_point1.z, data_point1.roll, data_point1.pitch, data_point1.yaw]).reshape(6, 1)
    data_point2_vec = np.array([data_point2.x, data_point2.y, data_point2.z, data_point2.roll, data_point2.pitch, data_point2.yaw]).reshape(6, 1)

    interp_state = (1 - weight1) * data_point1_vec
    interp_state += (1 - weight2) * data_point2_vec

    # Create a new ground truth object with the interpolated state vector
    return interp_state