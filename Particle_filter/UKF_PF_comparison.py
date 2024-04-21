import os
import numpy as np
from typing import List
from ImageUtils import traj_plot, plot_rmse_loss
from utils import interpolate_ground_truth, rmse, Data, Coordinate
from observation_model import ObservationModel, load_mat_data, pose_to_ypr
from Particle_filter import ParticleFilter
from UnscentedKalmanFilter import UnscentedKalmanFilter
from time import time
from matplotlib import pyplot as plt


def pos_ypr_ukf_pf(timestamps, gt_data, pf_data, ukf_data):
    """
    Creates and returns figures comparing the position and orientation data from ground truth, camera estimates, and particle filter (PF) estimates.
    
    Args:
        timestamps (List[float]): Timestamps corresponding to each data point.
        gt_data (List[np.ndarray]): Ground truth data (positions and orientations).
        camera_data (List[np.ndarray]): Camera estimation data.
        pf_data (List[np.ndarray]): Particle Filter estimation data.

    Returns:
        Tuple[plt.Figure, plt.Figure]: A tuple containing two figures; one for position comparison and one for orientation comparison.
    """
    fig_pos, ax_pos = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig_ori, ax_ori = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

    # Labels and titles
    pos_labels = ['X Position', 'Y Position', 'Z Position']
    ori_labels = ['Yaw', 'Pitch', 'Roll']
    sources = ['Ground Truth', 'PF Estimate', 'UKF Estimate']
    data_sources = [gt_data, pf_data, ukf_data]
    colors = ['blue', 'orange', 'green']

    # Plotting positions
    for i in range(3):
        for source, color, data in zip(sources, colors, data_sources):
            ax_pos[i].plot(timestamps, [point[i] for point in data], label=source, color=color)
        ax_pos[i].set_title(pos_labels[i])
        ax_pos[i].set_xlabel('Time (s)')
        ax_pos[i].set_ylabel('Meters')
        ax_pos[i].legend()

    # Plotting orientations
    for i in range(3):
        for source, color, data in zip(sources, colors, data_sources):
            ax_ori[i].plot(timestamps, [point[i + 3] for point in data], label=source, color=color)
        ax_ori[i].set_title(ori_labels[i])
        ax_ori[i].set_xlabel('Time (s)')
        ax_ori[i].set_ylabel('Radians')
        ax_ori[i].legend()

    fig_pos.tight_layout()
    fig_ori.tight_layout()

    return fig_pos, fig_ori

def main():
    # Configure paths and directories
    base_dir = "/home/lucifer/WPI/Spring_courses/Advanced_Navigation/Particle_filter"
    data_dir = os.path.join(base_dir, "data")
    img_dir = os.path.join(base_dir, "hw4", "imgs", "task3")
    os.makedirs(img_dir, exist_ok=True)
    
    # Load dataset
    dataset = os.path.join(data_dir, "studentdata2.mat")
    drone_data, gt = load_mat_data(dataset)
    # print(f"1st row of data for debugging: {drone_data[0]}")
    dataset_name = os.path.basename(dataset).split('.')[0]

    # Initialize observation model
    atagmap = ObservationModel()
    data = []
    positions = []
    orientations = []
    times = []
    interpolated_gt = []

    # Data processing
    for datum in drone_data:
        if not datum.tags:
            continue
        try:
            interpolated_gt.append(interpolate_ground_truth(gt, datum))
        except Exception:
            continue
        data.append(datum)
        orientation, position = atagmap.estimate_pose(datum.tags)
        positions.append(position)
        orientations.append(pose_to_ypr(orientation))
        times.append(datum.timestamp)
    
    # Unscented Kalman Filter
    print("Running UKF")
    start_ukf_time = time()
    ukf = UnscentedKalmanFilter()
    ukf_estimates = ukf.run_filter(data)
    end_ukf_time = time()
    print(f"UKF took {end_ukf_time - start_ukf_time} seconds")
    
    # Particle Filter
    print("Running PF for 5000 particles")
    start_pf_time = time()
    pf = ParticleFilter(num_particles=5000)
    pf_estimates = pf.run_filter(data)
    end_pf_time = time()
    print(f"PF took {end_pf_time - start_pf_time} seconds")
    ground_truth_positions = np.array([[gti[0], gti[1], gti[2]] for gti in interpolated_gt[1:]]).reshape(-1, 3)
    ukf_estimates_positions = np.squeeze(ukf_estimates)[:, :3]
    pf_estimates_positions = np.squeeze(pf_estimates)[:, :3]

    min_length = min(len(interpolated_gt), len(pf_estimates), len(ukf_estimates),len(positions), len(orientations), len(times))
    interpolated_gt = interpolated_gt[:min_length]
    pf_estimates = [pf_estimates[i] for i in range(min_length)]
    ukf_estimates = [ukf_estimates[i] for i in range(min_length)]
    positions = positions[:min_length]
    orientations = orientations[:min_length]
    timestamps = times[:min_length]  # Adjusting timestamps to match the data length

    ground_truth_data = np.array([[gt[0], gt[1], gt[2], gt[3], gt[4], gt[5]] for gt in interpolated_gt])
    pf_estimates_data = np.array([[est[0], est[1], est[2], est[3], est[4], est[5]] for est in pf_estimates])
    ukf_estimates_data = np.array([[est[0], est[1], est[2], est[3], est[4], est[5]] for est in ukf_estimates])

    pos_fig, orient_fig = pos_ypr_ukf_pf(
        timestamps, 
        ground_truth_data,
        pf_estimates_data,
        ukf_estimates_data
    )

    #Plot RMSE loss comparison for PF and UKF
    pos_rmse, orientation_rmse = plot_rmse_loss(
        interpolated_gt,
        pf_estimates_data,
        ukf_estimates_data,
        timestamps
    )

    pos_fig.savefig(os.path.join(img_dir, f"{dataset_name}_ukf_pf_comparison.png"))
    orient_fig.savefig(os.path.join(img_dir, f"{dataset_name}_ukf_pf_comparison_orientations.png"))
    pos_rmse.savefig(os.path.join(img_dir, f"{dataset_name}_ukf_pf_pos_rmse_comparison.png"))
    orientation_rmse.savefig(os.path.join(img_dir, f"{dataset_name}_ukf_pf_orient_rmse_comparison.png"))
    
    #Print RMSE loss comparison for PF and UKF
    print(f"RMSE Loss for PF: {rmse(ground_truth_positions, pf_estimates_positions)}")
    print(f"RMSE Loss for UKF: {rmse(ground_truth_positions, ukf_estimates_positions)}")

if __name__ == "__main__":
    main()



