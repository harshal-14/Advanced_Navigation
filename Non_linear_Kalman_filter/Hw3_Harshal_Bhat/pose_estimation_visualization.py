import os
import numpy as np
from typing import List
from ImageUtils import pos_ypr_plots, traj_plot
from utils import interpolate_ground_truth
from observation_model import ObservationModel, pose_to_ypr, load_mat_data

def main():
    # Configure paths and directories
    base_dir = "/home/lucifer/WPI/Spring_courses/Advanced_Navigation/Non_linear_Kalman_filter"
    data_dir = os.path.join(base_dir, "data")
    img_dir = os.path.join(base_dir, "hw3", "imgs", "task1_2")
    os.makedirs(img_dir, exist_ok=True)
    
    # Load dataset
    dataset = os.path.join(data_dir, "studentdata0.mat")
    data, gt = load_mat_data(dataset)
    dataset_name = os.path.basename(dataset).split('.')[0]

    # Initialize observation model
    atagmap = ObservationModel()
    positions : List[np.ndarray] = []
    orientations : List[np.ndarray] = []
    times : List[float] = []
    interpolated_gt : List[np.ndarray] = []

    # Data processing
    for datum in data:
        if not datum.tags:
            continue
        try:
            interpolated_gt.append(interpolate_ground_truth(gt, datum))
        except IndexError: # No ground truth at this time
            continue
        
        orientation, position = atagmap.estimate_pose(datum.tags)
        positions.append(position)
        orientations.append(pose_to_ypr(orientation))
        times.append(datum.timestamp)

    # Visualization and output
    create_and_save_plots(dataset_name, interpolated_gt, positions, orientations, times, gt, img_dir)

def create_and_save_plots(dataset_name, interpolated_gt, positions, orientations, times, gt, output_dir):
    position_figure, orientation_figure = pos_ypr_plots(
        interpolated_gt,
        [np.hstack((pos, ori)) for pos, ori in zip(positions, orientations)],
        times,
    )

    # Save figures
    position_figure.savefig(os.path.join(output_dir, f"{dataset_name}_trajectory_merged.png"))
    orientation_figure.savefig(os.path.join(output_dir, f"{dataset_name}_orientation_merged.png"))
    
    # Isometric plots
    Trajectory_plots = [
        ("Ground Truth Trajectory", "Ground Truth", [[gt_point.x, gt_point.y, gt_point.z] for gt_point in gt]),
        ("Estimated Trajectory", "Camera Estimate", [[pos[0], pos[1], pos[2]] for pos in positions]),
        ("Trajectories", "Ground Truth", [[gt_point.x, gt_point.y, gt_point.z] for gt_point in gt], "Camera Estimate", [[pos[0], pos[1], pos[2]] for pos in positions])
    ]
    for title, *data in Trajectory_plots:
        fig = traj_plot(title, *data)
        fig.savefig(os.path.join(output_dir, f"{dataset_name}_{title.lower().replace(' ', '_')}.png"))

if __name__ == "__main__":
    main()
