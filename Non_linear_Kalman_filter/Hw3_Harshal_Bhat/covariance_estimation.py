from typing import List
from utils import interpolate_ground_truth
import numpy as np
from observation_model import (Data, ActualData, ObservationModel, pose_to_ypr,
                   load_mat_data)

def estimate_total_cov(gt, positions, orientations, data):
    total_cov = []
    cntr = 0

    # Find the index of the first data point that is after the first ground truth
    start_idx = next((idx for idx, d in enumerate(data) if d.timestamp >= gt[0].timestamp), len(data))

    for idx in range(start_idx, len(positions)):
        cntr += 1
        try:
            interpolated = interpolate_ground_truth(gt, data[idx])
        except ValueError:
            continue

        position_vector = np.array([positions[idx][0], positions[idx][1], positions[idx][2],
                                    orientations[idx][0], orientations[idx][1], orientations[idx][2]]).reshape(6, 1)
        ERROR = interpolated - position_vector

        total_covariance = ERROR @ ERROR.T
        total_cov.append(total_covariance)  

    # Compute the average total covariance matrix
    if total_cov:

        average_total_covariance = sum(total_cov) / (cntr - 1)
    else:

        average_total_covariance = np.zeros((6, 6))

    return average_total_covariance


data_dir = "/home/lucifer/WPI/Spring_courses/Advanced_Navigation/Non_linear_Kalman_filter/data"
data_num = 1
for data_num in range(1, 8):
    dataset = f"{data_dir}/studentdata{data_num}.mat"

    total_cov: List[np.ndarray] = []

    base_data, gt = load_mat_data(dataset)

    obs_model = ObservationModel()

    positions: List[np.ndarray] = []
    orientations: List[np.ndarray] = []
    times: List[float] = []
    data: List[Data] = []
    for datum in base_data:
        # Estimate the pose of the camera
        if len(datum.tags) == 0:
            continue
        data.append(datum)
        orientation, position = obs_model.estimate_pose(datum.tags)
        positions.append(position)
        orientations.append(pose_to_ypr(orientation))
        times.append(datum.timestamp)

    average_total_covariance = estimate_total_cov(gt, positions, orientations, data)

    total_cov.append(average_total_covariance)

    print(f"-" * 50)
    print(f"Dataset: {dataset}")
    print(average_total_covariance)

    average_total_covariance = (1 / len(total_cov)) * np.sum(total_cov, axis=0)

    print(f"-" * 50)
    print("Average total_covariance")
    print(average_total_covariance)
    print(f"-" * 50)
