import os
import numpy as np
from typing import List
from ImageUtils import traj_plot, pos_ypr_pf_plots, plot_rmse_loss
from utils import interpolate_ground_truth, rmse, Data, Coordinate
from observation_model import ObservationModel, pose_to_ypr, load_mat_data
from Particle_filter import ParticleFilter
from time import time

def main():
    # Configure paths and directories
    base_dir = "/home/lucifer/WPI/Spring_courses/Advanced_Navigation/Particle_filter"
    data_dir = os.path.join(base_dir, "data")
    img_dir_template = os.path.join(base_dir, "hw4", "imgs", "task1")
    os.makedirs(img_dir_template, exist_ok=True)
    
    # Load dataset
    # dataset = os.path.join(data_dir, "studentdata0.mat")
    # drone_data, gt = load_mat_data(dataset)
    # # print(f"1st row of data for debugging: {drone_data[0]}")
    # dataset_name = os.path.basename(dataset).split('.')[0]

    # Initialize observation model
    obsmodel = ObservationModel()
    for i in range(8):

        dataset = os.path.join(data_dir, f"studentdata{i}.mat")
        img_dir = os.path.join(img_dir_template, f"task1_{i}")
        os.makedirs(img_dir, exist_ok=True)
        print(f"Processing dataset: {dataset}")
        # Load dataset
        drone_data, gt = load_mat_data(dataset)
        dataset_name = os.path.basename(dataset).split('.')[0]

        positions = []
        orientations = []
        times = []
        data = []
        interpolated_gt = []
        cam_estimates = []

        # Data processing
        for datum in drone_data:
            if len(datum.tags) == 0:
                continue
            try:
                interpolated_gt.append(interpolate_ground_truth(gt, datum))
            except Exception:
                continue
            data.append(datum)
            orientation, position = obsmodel.estimate_pose(datum.tags)
            cam_estimates.append(np.concatenate([position, orientation]))
            positions.append(position)
            orientations.append(pose_to_ypr(orientation))
            times.append(datum.timestamp)

        particle_filter = ParticleFilter(num_particles=10000)
        pf_estimates = particle_filter.run_filter(data, type = "weighted_average")

        ground_truth_positions = np.array([[gti[0], gti[1], gti[2]] for gti in interpolated_gt[1:]]).reshape(-1, 3)
        estimates_positions = np.squeeze(pf_estimates)[:, :3] 
        
        min_length = min(len(interpolated_gt), len(pf_estimates), len(positions), len(orientations), len(times))
        interpolated_gt = interpolated_gt[:min_length]
        pf_estimates = [pf_estimates[i] for i in range(min_length)]
        positions = positions[:min_length]
        orientations = orientations[:min_length]
        timestamps = times[:min_length]  # Adjusting timestamps to match the data length

        ground_truth_data = np.array([[gt[0], gt[1], gt[2], gt[3], gt[4], gt[5]] for gt in interpolated_gt])
        estimates_data = np.array([[est[0], est[1], est[2], est[3], est[4], est[5]] for est in pf_estimates])
        camera_data = np.array([[pos[0], pos[1], pos[2], ori[0], ori[1], ori[2]] for pos, ori in zip(positions, orientations)])

        # Plotting
        pos_fig, orient_fig = pos_ypr_pf_plots(
            timestamps, 
            ground_truth_data,
            camera_data,
            estimates_data
        )

        #Plot RMSE loss comparison for Camera and PF
        pos_rmse, orientation_rmse = plot_rmse_loss(interpolated_gt, camera_data, estimates_data, timestamps)

        isometric = traj_plot(
            "Isometric View of Ground Truth and Estimated Positions",
            "Ground Truth",
            [Coordinate(x=gt[i].x, y=gt[i].y, z=gt[i].z) for i in range(len(gt))],
            "PF Estimation",
            [Coordinate(x=position[0], y=position[1], z=position[2]) for position in estimates_positions]
        )

        # Save the figures
        pos_fig.savefig(os.path.join(img_dir, f"{dataset_name}_position_comparison.png"))
        orient_fig.savefig(os.path.join(img_dir, f"{dataset_name}_orientation_comparison.png"))
        pos_rmse.savefig(os.path.join(img_dir, f"{dataset_name}_position_rmse.png"))
        orientation_rmse.savefig(os.path.join(img_dir, f"{dataset_name}_orientation_rmse.png"))
        isometric.savefig(os.path.join(img_dir, f"{dataset_name}_pf_trajecory.png"))

        print(f"Position RMSE for {dataset_name}: {rmse(ground_truth_positions, estimates_positions)}")
        print(f"Orientation RMSE for {dataset_name}: {rmse(ground_truth_data[:, 3:], estimates_data[:, 3:])}")
        print(f"All Done!")
if __name__ == "__main__":
    main()

