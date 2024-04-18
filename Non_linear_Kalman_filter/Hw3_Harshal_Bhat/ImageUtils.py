import matplotlib.pyplot as plt

from utils import interpolate_ground_truth, rmse, ActualData, Data, Coordinate

from typing import List, Optional, Tuple

import numpy as np

from math import pi


def traj_plot(title: str, *args) -> plt.Figure:
    """
    Creates isometric 3D scatter plots with multiple datasets.
    Each pair in the args must contain a label for the dataset and its datapoints,
    with the 'jet' colormap applied by default to color data points based on their z-coordinate.

    Args:
        title (str): The title of the plot.
        *args: Alternating labels and corresponding data points, where each
               data point must be an iterable of coordinates [x, y, z].

    Returns:
        plt.Figure: The matplotlib figure object with the 3D scatter plot.

    Raises:
        ValueError: If no arguments are provided or the number of arguments
                    is not even, indicating missing labels or data.
    """
    if len(args) == 0:
        raise ValueError("No plotting arguments provided")
    elif len(args) % 2 != 0:
        raise ValueError("Arguments must be in pairs of label and data")

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.dist = 11
    ax.set_title(title)

    cmap = plt.get_cmap('plasma')  

    # Process each label-data pair
    for label, data in zip(args[::2], args[1::2]):
        data = np.array(data)  # Convert data to numpy array for efficient slicing
        colors = cmap((data[:, 2] - data[:, 2].min()) / (data[:, 2].max() - data[:, 2].min()))
        ax.scatter3D(
            data[:, 0], data[:, 1], data[:, 2],
            color=colors, 
            linewidths=0.3,
            label=label
        )

    ax.legend()
    return fig

def pos_ypr_plots(
    ground_truth: List[np.ndarray],
    estimates: List[np.ndarray],
    timestamps: List[float],
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Return separate comparison plots for positions (X, Y, Z individually) and orientations (yaw, pitch, roll individually) 
    versus incoming estimated est_state. Adjusts data lengths to match the smallest array size among inputs.

    Returns positions figure, orientations figure respectively.
    """
    # Ensure data alignment by truncating to the shortest array length among estimates or timestamps
    min_length = min(len(timestamps), len(ground_truth), len(estimates))
    if len(timestamps) > min_length:
        timestamps = timestamps[:min_length]
    if len(ground_truth) > min_length:
        ground_truth = ground_truth[:min_length]
    if len(estimates) > min_length:
        estimates = estimates[:min_length]

    # Extract positions and orientations directly as specified
    est_pos = [estimate[:3] for estimate in estimates]
    est_ypr = [estimate[3:6] for estimate in estimates]

    gt_coords = [Coordinate(x=gt[0], y=gt[1], z=gt[2]) for gt in ground_truth]
    est_coords = [Coordinate(x=pos[0], y=pos[1], z=pos[2]) for pos in est_pos]

    # Setup figures for position and orientation comparisons
    positions_figure, pos_axs = plt.subplots(3, 1, figsize=(10, 18))
    positions_figure.suptitle("Position Comparison of Ground Truth and Estimated")
    orientations_figure, ori_axs = plt.subplots(3, 1, figsize=(10, 18))
    orientations_figure.suptitle("Orientation Comparison of Ground Truth and Estimated")

    # Labels for axes
    position_labels = ['X', 'Y', 'Z']
    orientation_labels = ['Yaw', 'Pitch', 'Roll']

    # Extract and plot position data
    for ax, label, idx in zip(pos_axs, position_labels, range(3)):
        gt_data = np.array([getattr(coord, label.lower()) for coord in gt_coords])
        est_data = np.array([getattr(coord, label.lower()) for coord in est_coords])
        
        
        ax.set_title(f"{label} Position")
        ax.set_xlabel("Idx")
        ax.set_ylabel(f"{label} Coordinate Value")

        # Markers at the min and max points
        min_idx_gt = np.argmin(gt_data)
        max_idx_gt = np.argmax(gt_data)
        min_idx_est = np.argmin(est_data)
        max_idx_est = np.argmax(est_data)

        ax.plot(timestamps, [getattr(coord, label.lower()) for coord in gt_coords], label=f"GT {label}", marker='o', markevery=[min_idx_gt, max_idx_gt])
        ax.plot(timestamps, [getattr(coord, label.lower()) for coord in est_coords], label=f"Est {label}", linestyle='--', marker='x', markevery=[min_idx_est, max_idx_est])
        ax.legend()

    # Extract and plot orientation data
    for ax, label, idx in zip(ori_axs, orientation_labels, range(3)):
        ax.set_title(f"{label} Orientation")
        ax.set_xlabel("Time")
        ax.set_ylabel(f"{label} (radians)")
        gt_orientation = [gt[idx+3] for gt in ground_truth]
        est_orientation = [est[idx] for est in est_ypr]

        # Markers at the min and max points
        min_idx_gt = np.argmin(gt_orientation)
        max_idx_gt = np.argmax(gt_orientation)
        min_idx_est = np.argmin(est_orientation)
        max_idx_est = np.argmax(est_orientation)

        ax.plot(timestamps, gt_orientation, label=f"GT {label}", marker='o', markevery=[min_idx_gt, max_idx_gt])
        ax.plot(timestamps, est_orientation, label=f"Est {label}", linestyle='--', marker='x', markevery=[min_idx_est, max_idx_est])
        ax.legend()

    return positions_figure, orientations_figure

def pos_ypr_ukf_plots(
    ground_truth: List[np.ndarray],
    camera_estimations: List[np.ndarray],
    ukf_estimations: List[np.ndarray],
    timestamps: List[float],
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Plot comparison of positions and orientations from ground truth, camera estimations,
    and UKF estimations over time.

    Args:
        ground_truth (List[np.ndarray]): Ground truth data.
        camera_estimations (List[np.ndarray]): Camera estimations data.
        ukf_estimations (List[np.ndarray]): UKF estimations data.
        timestamps (List[float]): Timestamps for the data points.

    Returns:
        Tuple[plt.Figure, plt.Figure]: Tuple containing the matplotlib figures for position and orientation plots.
    """
    min_length = min(len(timestamps), len(ground_truth), len(camera_estimations), len(ukf_estimations))
    if len(timestamps) > min_length:
        timestamps = timestamps[:min_length]
    if len(ground_truth) > min_length:
        ground_truth = ground_truth[:min_length]
    if len(camera_estimations) > min_length:
        camera_estimations = camera_estimations[:min_length]
    if len(ukf_estimations) > min_length:
        ukf_estimations = ukf_estimations[:min_length]

    # Prepare data for plotting
    gt_pos = [gt[:3] for gt in ground_truth]
    gt_orient = [gt[3:6] for gt in ground_truth]

    cam_pos = [est[:3] for est in camera_estimations]
    cam_orient = [est[3:6] for est in camera_estimations]

    ukf_pos = [est[:3] for est in ukf_estimations]
    ukf_orient = [est[3:6] for est in ukf_estimations]

    # Create figures
    pos_fig, pos_axes = plt.subplots(3, 1, figsize=(10, 15))
    pos_fig.suptitle('Position Comparisons')
    orient_fig, orient_axes = plt.subplots(3, 1, figsize=(10, 15))
    orient_fig.suptitle('Orientation Comparisons')

    # Labels for plots
    units = ['m', 'm', 'm', 'rad', 'rad', 'rad']
    position_labels = ['X', 'Y', 'Z']
    orientation_labels = ['Yaw', 'Pitch', 'Roll']

    # Plotting positions
    for i, ax in enumerate(pos_axes):
        ax.plot(timestamps, [pos[i] for pos in gt_pos], label='Ground Truth', marker='o')
        ax.plot(timestamps, [pos[i] for pos in cam_pos], label='Camera Estimate', linestyle='--')
        ax.plot(timestamps, [pos[i] for pos in ukf_pos], label='UKF Estimate', linestyle='-.')
        ax.set_ylabel(f'{position_labels[i]} ({units[i]})')
        ax.legend()

    # Plotting orientations
    for i, ax in enumerate(orient_axes):
        ax.plot(timestamps, [orient[i] for orient in gt_orient], label='Ground Truth', marker='o')
        ax.plot(timestamps, [orient[i] for orient in cam_orient], label='Camera Estimate', linestyle='--')
        ax.plot(timestamps, [orient[i] for orient in ukf_orient], label='UKF Estimate', linestyle='-.')
        ax.set_ylabel(f'{orientation_labels[i]} Orientation ({units[i+3]})')
        ax.legend()

    # Setting x-labels
    for ax in [*pos_axes, *orient_axes]:
        ax.set_xlabel('Time (s)')

    return pos_fig, orient_fig

def plot_rmse_loss(
    real_state: List[np.ndarray],
    cam_estimations: List[np.ndarray],
    est_state: np.ndarray,
    timestamps: List[float],
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Given the ground truths, data, and est_state calculated, return
    plots for position and orientation, respectively.
    """

    gt_pos = [np.array([gt[0], gt[1], gt[2]]) for gt in real_state]
    gt_orientations = [np.array([gt[3], gt[4], gt[5]]) for gt in real_state]

    cam_pos = [np.array([estimate[0], estimate[1], estimate[2]]) for estimate in cam_estimations ]
    cam_orientations = [ np.array([estimate[3], estimate[4], estimate[5]]) for estimate in cam_estimations ]

    x_pos = [state[0:3] for state in est_state]
    x_orientations = [state[3:6] for state in est_state]

    # Calculate all RMSEs
    cam_pos_rmse = [ rmse(gt, cam_pos[idx]) for idx, gt in enumerate(gt_pos) ]
    cam_orientation_rmse = [ rmse(gt, cam_orientations[idx]) for idx, gt in enumerate(gt_orientations) ]

    x_pos_rmse = [ rmse(gt, x_pos[idx]) for idx, gt in enumerate(gt_pos[1:])]
    x_orientation_rmse = [ rmse(gt, x_orientations[idx]) for idx, gt in enumerate(gt_orientations[1:])]

    pos_fig = plt.figure(figsize=(10, 6), layout="tight")
    axes = plt.axes()
    axes.set_title("Position RMSE Loss")
    axes.set_xlabel("Time")
    axes.set_ylabel("RMSE Loss (m)")

    # Plot the RMSEs
    axes.plot(timestamps, cam_pos_rmse, label="Accumulated Error for Camera Estimates")
    axes.plot(timestamps[1:], x_pos_rmse, label="Error for UKF")
    axes.legend()

    orientation_fig = plt.figure(figsize=(10, 6), layout="tight")
    axes = plt.axes()
    axes.set_title("Orientation RMSE Loss")
    axes.set_xlabel("Time")
    axes.set_ylabel("RMSE Loss (rad)")

    # Plot the RMSEs
    axes.plot(timestamps, cam_orientation_rmse, label="Accumulated Error for Camera Estimates")
    axes.plot(timestamps[1:], x_orientation_rmse, label="Error for UKF")
    axes.legend()

    return pos_fig, orientation_fig
