import os
import numpy as np
import matplotlib.pyplot as plt
from ImageUtils import traj_plot, pos_ypr_pf_plots, plot_rmse_loss
from utils import interpolate_ground_truth, rmse, Data, Coordinate
from observation_model import ObservationModel, pose_to_ypr, load_mat_data
from Particle_filter import ParticleFilter
from time import time

def main():
    base_dir = "/home/lucifer/WPI/Spring_courses/Advanced_Navigation/Particle_filter"
    data_dir = os.path.join(base_dir, "data")
    img_dir_template = os.path.join(base_dir, "hw4", "imgs", "task1")
    os.makedirs(img_dir_template, exist_ok=True)

    filter_types = ['weighted_average', 'max', 'average']
    rmse_results = {ftype: {'position': [], 'orientation': []} for ftype in filter_types}
    dataset_labels = []

    obsmodel = ObservationModel()

    for i in range(1, 8):
        dataset = os.path.join(data_dir, f"studentdata{i}.mat")
        img_dir = os.path.join(img_dir_template, f"task1_{i}")
        os.makedirs(img_dir, exist_ok=True)

        print(f"Processing dataset: {dataset}")
        drone_data, gt = load_mat_data(dataset)
        dataset_name = os.path.basename(dataset).split('.')[0]
        dataset_labels.append(dataset_name)

        # Filter out data without tags
        data = [datum for datum in drone_data if len(datum.tags) > 0]
        particle_filter = ParticleFilter(num_particles=5000)
        interpolated_gt = []

        for datum in drone_data:
            if len(datum.tags) == 0:
                continue
            try:
                interpolated_gt.append(interpolate_ground_truth(gt, datum))
            except Exception as e:
                print(f"Error interpolating ground truth for datum: {e}")
                continue

        for ftype in filter_types:
            pf_estimates = particle_filter.run_filter(data, type=ftype)
            pf_estimates = np.squeeze(pf_estimates)
            
            ground_truth_positions = np.array([[gti[0], gti[1], gti[2]] for gti in interpolated_gt]).reshape(-1, 3)
            ground_truth_orientations = np.array([gti[3:] for gti in interpolated_gt]).reshape(-1, 3)

            # Ensuring the same length for comparison
            min_length = min(len(ground_truth_positions), len(pf_estimates))
            ground_truth_positions = ground_truth_positions[:min_length]
            pf_estimates_positions = pf_estimates[:min_length, :3]
            ground_truth_orientations = ground_truth_orientations[:min_length]
            pf_estimates_orientations = pf_estimates[:min_length, 3:6]  

            pos_rmse = rmse(ground_truth_positions, pf_estimates_positions)
            ori_rmse = rmse(ground_truth_orientations, pf_estimates_orientations)

            rmse_results[ftype]['position'].append(pos_rmse)
            rmse_results[ftype]['orientation'].append(ori_rmse)

    # Plotting RMSE results
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(dataset_labels))
    for ftype in filter_types:
        ax.plot(x, rmse_results[ftype]['position'], marker='o', label=f'{ftype} Position')
        ax.plot(x, rmse_results[ftype]['orientation'], marker='x', label=f'{ftype} Orientation')

    ax.set_xlabel('Dataset Number')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE Comparison Across Different Filter Types')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels)
    ax.legend()
    plt.show()

    # Save the plot
    fig.savefig(os.path.join(img_dir_template, 'PF_type_rmse_comparison.png'))

if __name__ == "__main__":
    main()
