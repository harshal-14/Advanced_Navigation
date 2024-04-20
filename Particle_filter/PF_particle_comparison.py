import os
import numpy as np
from typing import List
from ImageUtils import pos_ypr_pf_plots
from utils import interpolate_ground_truth, rmse, Data
from observation_model import ObservationModel, pose_to_ypr, load_mat_data
from Particle_filter import ParticleFilter
from time import time
from matplotlib import pyplot as plt

def main():
    base_dir = "/home/lucifer/WPI/Spring_courses/Advanced_Navigation/Particle_filter"
    data_dir = os.path.join(base_dir, "data")
    img_dir = os.path.join(base_dir, "performance_comparison")
    os.makedirs(img_dir, exist_ok=True)
    
    dataset = os.path.join(data_dir, "studentdata1.mat")
    drone_data, gt = load_mat_data(dataset)
    dataset_name = os.path.basename(dataset).split('.')[0]
    
    atagmap = ObservationModel()
    data: List[Data] = []
    positions: List[np.ndarray] = []
    orientations: List[np.ndarray] = []
    times: List[float] = []
    interpolated_gt: List[np.ndarray] = []

    for datum in drone_data:
        if not datum.tags:
            continue
        interpolated_gt.append(interpolate_ground_truth(gt, datum))
        data.append(datum)
        orientation, position = atagmap.estimate_pose(datum.tags)
        positions.append(position)
        orientations.append(pose_to_ypr(orientation))
        times.append(datum.timestamp)

    num_particles_options = [250, 500, 750, 1000, 2000, 3000, 4000, 5000]
    rmse_positions = []
    rmse_orientations = []
    execution_times = []

    for num_particles in num_particles_options:
        print(f"Running Particle Filter with {num_particles} particles...")
        start_time = time()
        particle_filter = ParticleFilter(num_particles=num_particles)
        pf_estimates = particle_filter.run_filter(data)
        elapsed_time = time() - start_time
        execution_times.append(elapsed_time)

        # Synchronize lengths of all data arrays
        min_length = min(len(interpolated_gt), len(pf_estimates), len(positions), len(orientations), len(times))
        interpolated_gt = interpolated_gt[:min_length]
        pf_estimates = [pf_estimates[i] for i in range(min_length)]
        positions = positions[:min_length]
        orientations = orientations[:min_length]
        times = times[:min_length]

        # Compute RMSE after ensuring all arrays have the same dimensions
        gt_pos = np.array([gt[:3] for gt in interpolated_gt]).reshape(min_length, 3)
        est_pos = np.squeeze(np.array([est[:3] for est in pf_estimates]), axis=-1)
        pos_error = rmse(gt_pos, est_pos)
        rmse_positions.append(pos_error)

        gt_ori = np.array([gt[3:6] for gt in interpolated_gt]).reshape(min_length, 3)
        est_ori = np.squeeze(np.array([est[3:6] for est in pf_estimates]), axis=-1)
        orient_error = rmse(gt_ori, est_ori)
        rmse_orientations.append(orient_error)

        print(f"Time taken: {elapsed_time:.2f}s, RMSE Position: {pos_error:.2f}, RMSE Orientation: {orient_error:.2f}")

    # Plotting results
    plt.figure(figsize=(10, 5))
    plt.plot(num_particles_options, execution_times, marker='o', linestyle='-', color='b', label='Execution Time (s)')
    plt.xlabel('Number of Particles')
    plt.ylabel('Execution Time (s)')
    plt.title('Particle Filter Performance Analysis')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(img_dir, f"{dataset_name}_execution_times.png"))

    plt.figure(figsize=(10, 5))
    plt.plot(num_particles_options, rmse_positions, marker='o', linestyle='-', color='r', label='Position RMSE')
    plt.plot(num_particles_options, rmse_orientations, marker='o', linestyle='-', color='g', label='Orientation RMSE')
    plt.xlabel('Number of Particles')
    plt.ylabel('RMSE')
    plt.title('Particle Filter RMSE Analysis')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(img_dir, f"{dataset_name}_rmse_analysis.png"))

if __name__ == "__main__":
    main()
