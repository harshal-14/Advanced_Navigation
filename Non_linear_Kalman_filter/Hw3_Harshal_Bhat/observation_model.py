from __future__ import annotations

from math import pi
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from cv2 import Rodrigues, solvePnP
from scipy.io import loadmat
from utils import PixelCoordinate, Coordinate, Marker, Data, ActualData

class ObservationModel:
    """
    DataMap is a class that maintains a mapping of the
    AprilTag IDs to their real world coordinates. It also
    contains the camera matrix and distortion coefficients
    for the camera used to take the images.
    """

    def __init__(
        self,
        tags: Optional[Dict[int, Coordinate]] = None,
        camera_matrix: Optional[np.ndarray] = None,
        distortion_coefficients: Optional[np.ndarray] = None,
    ):
        # If tags is None, we will set up the tags in a grid pattern
        if tags is None:
            self.setup_tags()
        else:
            self.tags = tags
        self.camera_matrix = camera_matrix if camera_matrix is not None else np.array(
                [[314.1779, 0, 199.4848], [0, 314.2218, 113.7838], [0, 0, 1]]
            )
        self.distortion_coefficients = distortion_coefficients if distortion_coefficients is not None else np.array(
                [-0.438607, 0.248625, 0.00072, -0.000476, -0.0911]
            )

    def setup_tags(self):
        # We will define the tags in a grid pattern
        tags: Dict[int, Marker] = {}
        layout = [
            [0, 12, 24, 36, 48, 60, 72, 84, 96],
            [1, 13, 25, 37, 49, 61, 73, 85, 97],
            [2, 14, 26, 38, 50, 62, 74, 86, 98],
            [3, 15, 27, 39, 51, 63, 75, 87, 99],
            [4, 16, 28, 40, 52, 64, 76, 88, 100],
            [5, 17, 29, 41, 53, 65, 77, 89, 101],
            [6, 18, 30, 42, 54, 66, 78, 90, 102],
            [7, 19, 31, 43, 55, 67, 79, 91, 103],
            [8, 20, 32, 44, 56, 68, 80, 92, 104],
            [9, 21, 33, 45, 57, 69, 81, 93, 105],
            [10, 22, 34, 46, 58, 70, 82, 94, 106],
            [11, 23, 35, 47, 59, 71, 83, 95, 107],
        ]

        # We now need to calculate the p1 through p4 values of each tag
        special_offset = 0.152
        extra_offset = 0.178 - special_offset
        # We will iterate through the layout and calculate the p1 through p4 values
        for idx, row_num in enumerate(layout):
            x_offset = special_offset * idx * 2
            for tag_idx, tag in enumerate(row_num):
                y_offset = special_offset * tag_idx * 2
                if tag_idx >= 3:
                    y_offset += extra_offset
                if tag_idx >= 6:
                    y_offset += extra_offset

                tl = Coordinate(x_offset, y_offset, 0)
                tr = Coordinate(x_offset, y_offset + special_offset, 0)
                br = Coordinate(x_offset + special_offset, y_offset + special_offset, 0)
                bl = Coordinate(x_offset + special_offset, y_offset, 0)
                tags[tag] = Marker(
                    id=tag, # id is the tag number
                    tl=tl,  # tl is the top left corner
                    tr=tr,  # tr is the top right corner
                    br=br,  # br is the bottom right corner
                    bl=bl,  # bl is the bottom left corner
                )

        self.tags = tags #Tags is a dictionary with key as tag number and value as Marker object

    def estimate_pose(self, tags: List[Marker]) -> Tuple[np.ndarray, np.ndarray]:
        """
        estimate_pose will, given a list of observed Markers,
        pair them with their real world coordinates in order to
        estimate the orientation and position of the camera at
        that moment in time.
        """
        world_points = []
        image_points = []
        for tag in tags:
            # import ipdb; ipdb.set_trace()
            # print(self.tags[tag.id])
            # print(tag.bl)
            world_points.append(self.tags[tag.id].bl)
            world_points.append(self.tags[tag.id].br)
            world_points.append(self.tags[tag.id].tr)
            world_points.append(self.tags[tag.id].tl)

            image_points.append(tag.bl)
            image_points.append(tag.br)
            image_points.append(tag.tr)
            image_points.append(tag.tl)
        world_points = np.array(world_points)
        image_points = np.array(image_points)

        _ , orientation, position = solvePnP(
            world_points,
            image_points,
            self.camera_matrix,
            self.distortion_coefficients,
            flags=0,
        )

        z_rotation = np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4), 0], 
                               [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
                               [0, 0, 1],])
        # We need to rotate the camera frame to match the drone frame, since the drone frame is rotated 180 degrees about the x-axis.
        x_rotation = np.array([[1, 0, 0],
                               [0, -1, 0],
                               [0, 0, -1],])
        rotation = x_rotation@z_rotation

        # We need to translate the camera frame to match the drone frame, since the drone frame is translated 0.04m in the x-axis and 0.03m in the z-axis.
        cam_to_drone_transform = np.array([[rotation[0, 0], rotation[0, 1], rotation[0, 2], -0.04],
                                           [rotation[1, 0], rotation[1, 1], rotation[1, 2], 0],
                                           [rotation[2, 0], rotation[2, 1], rotation[2, 2], -0.03],
                                           [0, 0, 0, 1],])
    #     import ipdb; ipdb.set_trace()
        
        orientation = Rodrigues(orientation)[0]

        cam_to_world_transform = np.array([np.concatenate((orientation[0], position[0])),
                                           np.concatenate((orientation[1], position[1])),
                                           np.concatenate((orientation[2], position[2])),
                                           [0, 0, 0, 1],])

        # We need to convert the camera frame to the world frame, for this we need to invert the transformation matrix
        # This will give us the drone frame in the world frame
        drone_to_world_transform = np.dot(np.linalg.inv(cam_to_world_transform), cam_to_drone_transform)

        estimated_position = drone_to_world_transform[0:3, 3]
        # Convert the rotation matrix back to a vector
        estimated_orientation = rotation_matrix_to_euler_angles(drone_to_world_transform[0:3, 0:3])

        return estimated_orientation, estimated_position


def rotation_matrix_to_euler_angles(
    rotation_matrix: np.ndarray,
) -> Tuple[float, float, float]:
    """
    rotation_matrix_to_euler_angles converts a 3x3 rotation matrix to
    a tuple of Euler angles in XZY rotation order.
    """
    r11, r12, r13 = rotation_matrix[0]
    r21, r22, r23 = rotation_matrix[1]
    r31, r32, r33 = rotation_matrix[2]

    yaw = np.arctan(-r12 / r22)
    roll = np.arctan(r32 * np.cos(yaw) / r22)
    pitch = np.arctan(-r31 / r33)

    return yaw, pitch, roll


def pose_to_ypr(
    orientation: np.ndarray,
) -> Tuple[float, float, float]:
    """
    pose_to_ypr will take a rotation matrix and
    convert it to a tuple of yaw, pitch, and roll.
    """
    # Convert the 3x1 matrix to a rotation matrix
    rotation = Rodrigues(orientation)[0]
    # Convert the rotation matrix to Euler angles, by taking the arctan of the elements of the rotation matrix
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    pitch = np.arctan2(-rotation[2, 0], np.sqrt(rotation[2, 1] ** 2 + rotation[2, 2] ** 2))
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])

    return yaw, pitch, roll


def load_mat_data(filepath: str) -> Tuple[List[Data], List[ActualData]]:
    """
    Read the .mat file from the given filepath and return the
    data and ground truth as a tuple of lists of Data and
    ActualData objects, respectively.
    """

    mat = loadmat(filepath, simplify_cells=True)

    data_mat = mat["data"]
    time_mat = mat["time"]
    vicon_mat = mat["vicon"]

    # Build our data list
    data: List[Data] = []
    for index, datum in enumerate(data_mat):
        tags: List[Marker] = []

        if isinstance(datum["id"], int):
            datum["id"] = [datum["id"]]
            for point in ["p1", "p2", "p3", "p4"]:
                datum[point] = [[datum[point][0]], [datum[point][1]]]

        for index, id in enumerate(datum["id"]):
            # import ipdb; ipdb.set_trace()
            tags.append(
                Marker(
                    id=id,
                    bl=PixelCoordinate(
                        datum["p1"][0][index], datum["p1"][1][index]
                    ),
                    br=PixelCoordinate(
                        datum["p2"][0][index], datum["p2"][1][index]
                    ),
                    tr=PixelCoordinate(
                        datum["p3"][0][index], datum["p3"][1][index]
                    ),
                    tl=PixelCoordinate(
                        datum["p4"][0][index], datum["p4"][1][index]
                    ),
                )
            )

        omg = datum["omg"] if "omg" in datum else datum["drpy"]

        data.append(
            Data(
                img=datum["img"],
                tags=tags,
                timestamp=datum["t"],
                rpy=datum["rpy"],
                acc=datum["acc"],
                omg=omg,
            )
        )

    # Build our ground truth list
    gt: List[ActualData] = []
    for index, moment in enumerate(time_mat):
        gt.append(
            ActualData(
                timestamp=moment,
                x=vicon_mat[0][index],
                y=vicon_mat[1][index],
                z=vicon_mat[2][index],
                roll=vicon_mat[3][index],
                pitch=vicon_mat[4][index],
                yaw=vicon_mat[5][index],
                vx=vicon_mat[6][index],
                vy=vicon_mat[7][index],
                vz=vicon_mat[8][index],
                wx=vicon_mat[9][index],
                wy=vicon_mat[10][index],
                wz=vicon_mat[11][index],
            )
        )

    return data, gt
