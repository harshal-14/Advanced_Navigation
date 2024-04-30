from math import pi
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ins_gnss import Data, load_data


# def rmse(y_true, y_pred):
#     return np.sqrt(np.mean((y_true - y_pred) ** 2))

def rmse(y_true, y_pred):
    """Calculate the root mean squared error between two sets of arrays along each axis."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute RMSE for each dimension (column) if multidimensional, otherwise just compute directly
    if y_true.ndim > 1:
        return np.sqrt(((y_true - y_pred) ** 2).mean(axis=0))
    else:
        return np.sqrt(np.mean((y_true - y_pred) ** 2))


def create_overlay_plots(ground_truth: List[np.ndarray], estimates: List[np.ndarray], haversines: List[float], times: List[float]):
    """
    Returns three figures: one for positions (latitude, longitude, and altitude), one for haversine distances,
    and one for orientation (roll, pitch, yaw).
    """
    # Extract position data
    gt_lat = [gt[0] for gt in ground_truth]
    gt_lon = [gt[1] for gt in ground_truth]
    gt_alt = [gt[2] for gt in ground_truth]
    est_lat = [est[0] for est in estimates]
    est_lon = [est[1] for est in estimates]
    est_alt = [est[2] for est in estimates]

    # Position Figure - Latitude and Longitude
    position_fig, pos_axs = plt.subplots(3, 1, figsize=(10, 15))
    position_fig.suptitle("Overlay of Ground Truth and Estimated Positions")

    pos_axs[0].set_xlabel("Time")
    pos_axs[0].set_ylabel("Latitude")
    pos_axs[0].set_title("Latitude Over Time")
    pos_axs[0].plot(times, gt_lat[1:], label="Ground Truth", marker='o', linestyle='-')
    pos_axs[0].plot(times, est_lat, label="Estimated", marker='x', linestyle='--')
    pos_axs[0].legend()

    pos_axs[1].set_xlabel("Time")
    pos_axs[1].set_ylabel("Longitude")
    pos_axs[1].set_title("Longitude Over Time")
    pos_axs[1].plot(times, gt_lon[1:], label="Ground Truth", marker='o', linestyle='-')
    pos_axs[1].plot(times, est_lon, label="Estimated", marker='x', linestyle='--')
    pos_axs[1].legend()

    pos_axs[2].set_xlabel("Time")
    pos_axs[2].set_ylabel("Altitude")
    pos_axs[2].set_title("Altitude Over Time")
    pos_axs[2].plot(times, gt_alt[1:], label="Ground Truth", marker='o', linestyle='-')
    pos_axs[2].plot(times, est_alt, label="Estimated", marker='x', linestyle='--')
    pos_axs[2].legend()

    # Haversine Distance Plot
    haversine_fig, hav_ax = plt.subplots(figsize=(10, 5))
    haversine_fig.suptitle("Haversine Distances Over Time")
    hav_ax.set_xlabel("Time")
    hav_ax.set_ylabel("Haversine Distance")
    hav_ax.plot(times, haversines, label="Haversine Distances", marker='o', linestyle='-')
    hav_ax.legend()

    # Lat vs Lon Plot
    lat_lon_fig, lat_lon_ax = plt.subplots(figsize=(10, 5))
    lat_lon_fig.suptitle("Latitude vs Longitude")
    lat_lon_ax.set_xlabel("Latitude")
    lat_lon_ax.set_ylabel("Longitude")
    lat_lon_ax.plot(gt_lat, gt_lon, label="Ground Truth", marker='o', linestyle='-')
    lat_lon_ax.plot(est_lat, est_lon, label="Estimated", marker='x', linestyle='--')
    lat_lon_ax.legend()
    
    return position_fig, haversine_fig, lat_lon_fig

def rmse_plots(
    ground_truth: List[np.ndarray],
    estimates: List[np.ndarray],
    data: List[Data],
) -> plt.Figure:
    gt_pos = [gt[0:3] for gt in ground_truth]
    est_pos = [est[0:3] for est in estimates]
    measured_pos = [np.array([d.z_lat, d.z_lon, d.z_alt]) for d in data]
    times = [d.time for d in data]

    error_est_positions: List[np.ndarray] = [
        rmse(gt, est) for gt, est in zip(gt_pos[1:], est_pos)
    ]
    error_measured_positions: List[np.ndarray] = [
        rmse(gt, measured) for gt, measured in zip(gt_pos, measured_pos)
    ]

    figure = plt.figure(figsize=(10, 6), layout="tight")
    axes = plt.axes()
    axes.set_xlabel("Time")
    axes.set_ylabel("RMSE Error")
    axes.dist = 11
    axes.set_title(
        "RMSE Error of Estimated and Measured Positions against Ground Truth"
    )
    axes.plot(times[2:], error_est_positions, label="Estimated Positions")

    return figure

