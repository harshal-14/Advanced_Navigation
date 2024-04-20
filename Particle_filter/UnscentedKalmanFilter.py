import os
from time import time
from typing import List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
from math import cos, sin, degrees, radians
import numpy as np
from ImageUtils import traj_plot, plot_rmse_loss, pos_ypr_plots, pos_ypr_pf_plots
from scipy.linalg import sqrtm, cholesky
from utils import interpolate_ground_truth, rmse
from observation_model import Coordinate, Data, ObservationModel, pose_to_ypr, load_mat_data

# Cov matrix below is calculated in covariance_estimation.py through
# the analyzing the data of the drone' experiments
estimated_cov = np.array([[ 0.00787652,  0.0002269,   0.00462655,  0.00362929,  0.00697831, -0.00029106],
 [ 0.0002269,   0.00483996, -0.00177252, -0.00401243,  0.0024168,   0.00014734],
 [ 0.00462655, -0.00177252,  0.00731506,  0.00361734,  0.00315442, -0.00074367],
 [ 0.00362929, -0.00401243,  0.00361734,  0.00530187,  0.00134563, -0.00023165],
 [ 0.00697831,  0.0024168,   0.00315442,  0.00134563,  0.0072551,  -0.00016959],
 [-0.00029106,  0.00014734, -0.00074367, -0.00023165, -0.00016959,  0.00010396]])


class UnscentedKalmanFilter:

    def __init__(
        self,
        measurement_cov_matrix: Optional[np.ndarray] = None,
        kappa = 1,
        alpha = 0.01,
        beta = 2.0,
    ):

        self.measurement_cov_matrix = measurement_cov_matrix if measurement_cov_matrix is not None else estimated_cov
 
        self.n = 15 # n is the dimensionality of our given x vector
        self.kappa = kappa                  # kappa is a tuning value for the filter
        self.alpha = alpha                  # alpha is used to calculate initial cov weights
        self.beta = beta

        self.n_sigma_pts = 2 * self.n + 1  # The number of sigma points we do are typically 2*n + 1,
        self.last_x = np.zeros((self.n, 1))
        self.measurement_noise = np.identity(self.n) * 1e-3

        self.ng = np.zeros((3, 1))          # ng and na are the biases from the Idist_mean
        self.na = np.zeros((3, 1))

        self.dist_mean = np.zeros((self.n, 1)) # dist_mean is the mean of the distribution

        self.obsmodel = ObservationModel()

        self.init_weights()
    
    def init_weights(self):
        """Initialize weights for the sigma points in the UKF."""
        lambda_ = self.alpha ** 2 * (self.n + self.kappa) - self.n
        self.W_mean = np.full(self.n_sigma_pts, 1 / (2 * (self.n + lambda_)))
        self.W_mean[0] = lambda_ / (self.n + lambda_)
        self.W_cov = np.copy(self.W_mean)
        self.W_cov[0] += 1 - self.alpha ** 2 + self.beta

    def compute_sigma_pts(self, dist_mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        #Ensure that the matrix is Positive semi deiﬁnite
        try:
            #Attempting Cholesky decomposition to ensure PSD
            S = cholesky((self.n + self.kappa)* sigma)
        except np.linalg.LinAlgError:
            eigen_values, eigne_vectors = np.linalg.eigh(sigma)
            jitter = 1e-3
            sigma_jitt = eigne_vectors @ np.diag(eigen_values + jitter) @eigne_vectors.T
            S = sqrtm((self.n + self.kappa) * sigma_jitt)
        
        sigma_pts = np.zeros((self.n_sigma_pts, self.n, 1))
        sigma_pts[0] = dist_mean #Setting the 1st sigma point to be the mean

        for i in range(self.n):
            direction = S[:, i:i+1] #Extract direction maintaining the shape
            sigma_pts[i+1] = dist_mean + direction #Adding scaled_cov to mean
            sigma_pts[self.n + i + 1] = dist_mean - direction #Subtracting scaled_cov from mean
        
        return sigma_pts

    def compute_drone_x(self, x: np.ndarray, dt: float, ua: np.ndarray, uw: np.ndarray) -> np.ndarray:
        """
        Process model to update the x based on given accelerations, delta time,
        and current x, handling the dynamics of the system.
        """
        # print(f"x shape: {x.shape}")
        # print(f"ua shape: {ua.shape}")
        # print(f"uw shape: {uw.shape}")
        # Reshape inputs to ensure they are column vectors
        ua = np.reshape(ua, (3, 1))
        uw = np.reshape(uw, (3, 1))
        # print(f"ua - x[9:12]: {ua - x[9:12]}")
        # print(f"uw - x[9:12]: {uw - x[9:12]}")
        # Extract orientation and velocity from the x
        theta, phi, psi = x[3:6].flatten()  # roll, pitch, yaw
        velocities = x[6:9]

        # Compute the rotation matrix from drone frame to world frame
        # G(q) - drone frame to world frame rotation matrix
        G = np.zeros((3, 3))
        G[0, 0] = cos(theta)
        G[0, 2] = -cos(phi) * sin(theta)
        G[1, 1] = 1
        G[1, 2] = sin(phi)
        G[2, 0] = sin(theta)
        G[2, 2] = cos(phi) * cos(theta)

        # R(q) - world frame to drone frame rotation matrix
        R = np.zeros((3, 3))
        R[0, 0] = cos(psi) * cos(theta) - sin(psi) * sin(theta) * sin(phi)
        R[0, 1] = -cos(phi) * sin(psi)
        R[0, 2] = cos(psi) * sin(theta) + cos(theta) * sin(phi) * sin(psi)
        R[1, 0] = cos(theta) * sin(psi) + cos(psi) * sin(phi) * sin(theta)
        R[1, 1] = cos(psi) * cos(phi)
        R[1, 2] = sin(psi) * sin(theta) - cos(psi) * cos(theta) * sin(phi)
        R[2, 0] = -cos(phi) * sin(theta)
        R[2, 1] = sin(phi)
        R[2, 2] = cos(phi) * cos(theta)

        # Differential changes in x
        xdot = np.zeros(x.shape)
        xdot[0:3] = velocities  # velocity updates directly to position differential
        xdot[3:6] = np.linalg.inv(G)  @ uw  # Orientation change influenced by angular velocity
        xdot[6:9] = np.array([[0], [0], [-9.81]]) + R @ ua  # Acceleration affects the velocity
        # Update the x using the calculated differentials
        new_x = x + xdot * dt

        return new_x

    def predict(
        self, sigma_pts: np.ndarray, ua: np.ndarray, uw: np.ndarray, dt: float
    ) -> Tuple[np.ndarray]:
        """
        predict takes the sigma points and returns the mean and covariance
        of the distribution after the prediction step
        Input:
        sigma_pts - the sigma points to predict from
        ua - the accelerometer readings
        uw - the gyroscope readings
        dt - the time since the last prediction
        Output:
        dist_mean - the mean of the distribution after the prediction
        sigma - the covariance of the distribution after the prediction
        transformed_pts - the sigma points after the prediction
        """
        transformed_pts = np.zeros_like(sigma_pts)
        for idx in range(sigma_pts.shape[0]):
            transformed_pts[idx, :] = self.compute_drone_x(sigma_pts[idx], dt, uw, ua)

        dist_mean = np.zeros((self.n, 1))
        for idx in range(0, self.n_sigma_pts):
            dist_mean += self.W_mean[idx] * transformed_pts[idx]

        Q = np.random.normal(scale=5e-1, size=(15, 15))
        differences = transformed_pts - dist_mean
        sigma = np.zeros((self.n, self.n))
        for idx in range(0, self.n):
            sigma += self.W_cov[idx] * np.dot(differences[idx], differences[idx].T)
        sigma += Q

        return dist_mean, sigma, transformed_pts

    def update(self, x, dist_mean, sigma, sigma_pts):
        """
        update takes the current state, the mean of the distribution after the prediction,
        Input: x - the current state
        dist_mean - the mean of the distribution after the prediction
        sigma - the covariance of the distribution after the prediction
        sigma_pts - the sigma points after the prediction
        Output:
        curr_pos - the current position after the update
        cov - the covariance of the current position after the update
        """
        # Apply the measurement function across each new sigma point
        z_pts = np.zeros_like(sigma_pts) # z_pts is an array of measurement points
        for i in range(self.n_sigma_pts):
            z_pts[i] = self.noise_adjust(sigma_pts[i])

        z_hat = np.zeros((self.n, 1))
        for i in range(0, self.n_sigma_pts):
            z_hat += self.W_mean[i] * z_pts[i]

        R = np.zeros((self.n, self.n))
        R[0:6, 0:6] = np.diag(self.measurement_cov_matrix)
        St = np.zeros((self.n, self.n))
        z_diff = z_pts - z_hat
        for i in range(0, self.n_sigma_pts):
            St += self.W_cov[i] * np.dot(
                z_diff[i], z_diff[i].T
            )
        St += R

        sigmahat_t = np.zeros((self.n, self.n))
        x_diff = sigma_pts - dist_mean
        for i in range(0, self.n_sigma_pts):
            sigmahat_t += self.W_cov[i] * np.dot(
                x_diff[i], z_diff[i].T
            )

        K = np.dot(sigmahat_t, np.linalg.pinv(St))

        # Update the mean and cov
        curr_pos = dist_mean + np.dot(K, x - z_hat)
        cov = sigma - np.dot(K, St).dot(K.T)

        cov = (cov + cov.T) / 2  # Make symmetric
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        eig_vals[eig_vals < 0] = 1e-3  # Replace negative eigenvalues with small positive value
        cov = eig_vecs @ np.diag(eig_vals) @ eig_vecs.T  # Reconstruct the matrix

        return curr_pos, cov

    def noise_adjust(self, x: np.ndarray) -> np.ndarray:
        """
        noise_adjust is used to adjust the noise based on the current state, since the noise dimensionality and application
        is different than the state dimensionality
        Input: x - the current state
        Output: adjusted_noise - the adjusted noise based on the current state
        """
        adjusted_noise = np.zeros((self.n, 1))
        c = np.zeros((6, self.n))
        c[0:6, 0:6] = np.eye(6)

        R = np.diag(self.measurement_cov_matrix).reshape(6, 1)
        adjusted_noise[0:6] = np.dot(c, x) + R

        return adjusted_noise

    def run_filter(self, estimated_positions: List[Data]) -> List[np.ndarray]:
        """
        Given a set of estimated positions, return the estimated positions after run_filterning
        the Unscented Kalman Filter over the data
        """
        filtered_positions: List[np.ndarray] = []

        if not estimated_positions:
            return filtered_positions

        # Initialize the first x based on the first data point
        first_data = estimated_positions[0]
        orientation, position = self.obsmodel.estimate_pose(first_data.tags)
        x = np.zeros((self.n, 1))
        x[0:3] = np.array(position).reshape((3, 1))
        x[3:6] = np.array(pose_to_ypr(orientation)).reshape((3, 1))
        x[6:9] = np.zeros((3, 1))  # Initial velocities assumed to be zero
        previous_x_timestamp = first_data.timestamp

        # Initialize covariance matrix
        process_cov_matrix = np.eye(self.n) * 1e-3

        # Iterate through all provided data points starting from the second one
        for idx, data in enumerate(estimated_positions[1:], start=1):
            orientation, position = self.obsmodel.estimate_pose(data.tags)
            x_curr = np.zeros((self.n, 1))
            x_curr[0:3] = np.array(position).reshape((3, 1))
            x_curr[3:6] = np.array(pose_to_ypr(orientation)).reshape((3, 1))
            x_curr[6:9] = np.zeros((3, 1))  # Assume the velocities can be recalculated or are measured

            ua = data.acc
            uw = data.omg

            # Delta t since our last prediction
            dt = data.timestamp - previous_x_timestamp
            previous_x_timestamp = data.timestamp

            # Get our sigma points
            sigma_pts = self.compute_sigma_pts(x, process_cov_matrix)

            # Run_filter the prediction step based off of our process model and state transition
            dist_meanbar, sigmabar, transformed_pts = self.predict(
                sigma_pts, ua, uw, dt
            )

            # Run_filter the update step to filter our estimated position and resulting sigma
            x, process_cov_matrix = self.update(x_curr, dist_meanbar, sigmabar, transformed_pts)

            filtered_positions.append(x)

        return filtered_positions

if __name__ == "__main__":
    # Creating directories more cleanly
    base_dir = "/home/lucifer/WPI/Spring_courses/Advanced_Navigation/Non_linear_Kalman_filter"
    image_dir = f"{base_dir}/outputs/task4"
    os.makedirs(image_dir, exist_ok=True)

    # Generating dataset paths using list comprehension
    data_dir = f"{base_dir}/data"
    datasets = [f"{data_dir}/studentdata{i}.mat" for i in range(8)]

    total_cam_positional_rmses = []
    all_ukf_pos_rmses = []
    total_cam_orientation_rmses = []
    all_ukf_orientation_rmses = []
    
    # Lists to store all errors for overall statistics
    total_trans_errors = []
    total_rot_errors_rad = []
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        base_data, gt = load_mat_data(dataset)
        dataset_name = dataset.split("/")[-1].split(".")[0]

        obsmodel = ObservationModel()
        x = UnscentedKalmanFilter()

        positions = []
        orientations = []
        times = []
        data = []
        interpolated_gt = []
        cam_estimates = []
        cam_pos_rmses = []
        ukf_pos_rmses = []
        cam_orientation_rmses = []
        ukf_orientation_rmses = []
        dataset_trans_errors = []
        dataset_rot_errors_rad = []

        for datum in base_data:
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

            pos_error = np.linalg.norm(interpolated_gt[-1][0:3] - cam_estimates[-1][0:3])
            orient_error_rad = np.linalg.norm(interpolated_gt[-1][3:6] - cam_estimates[-1][3:6])
            
            dataset_trans_errors.append(pos_error)
            dataset_rot_errors_rad.append(orient_error_rad)
            pos_rmse = rmse(interpolated_gt[-1][0:3], cam_estimates[-1][0:3])
            orientation_rmse = rmse(interpolated_gt[-1][3:6], cam_estimates[-1][3:6])
            cam_pos_rmses.append(pos_rmse)
            cam_orientation_rmses.append(orientation_rmse)
            total_cam_positional_rmses.append(pos_rmse)
            total_cam_orientation_rmses.append(orientation_rmse)

        results = x.run_filter(data)
        # Append dataset errors to total errors
        total_trans_errors.extend(dataset_trans_errors)
        total_rot_errors_rad.extend(dataset_rot_errors_rad)
        for idx, result in enumerate(results):
            pos_rmse = rmse(interpolated_gt[idx][0:3], result[0:3])
            orientation_rmse = rmse(interpolated_gt[idx][3:6], result[3:6])
            ukf_pos_rmses.append(pos_rmse)
            ukf_orientation_rmses.append(orientation_rmse)
            all_ukf_pos_rmses.append(pos_rmse)
            all_ukf_orientation_rmses.append(orientation_rmse)

        def compute_stats(errors, cam_pos_rmses=cam_pos_rmses, ukf_pos_rmses=ukf_pos_rmses, cam_orientation_rmses=cam_orientation_rmses, ukf_orientation_rmses=ukf_orientation_rmses):
            return {
                'Maximum': np.max(errors),
                'Mean': np.mean(errors),
                'Median': np.median(errors),
                'Minimum': np.min(errors),
                'Standard Deviation': np.std(errors),
                'Camera Positional RMSE' : np.mean(cam_pos_rmses),
                'UKF Positional RMSE' : np.mean(ukf_pos_rmses),
                'Camera Orientation RMSE' : np.mean(cam_orientation_rmses),
                'UKF Orientation RMSE' : np.mean(ukf_orientation_rmses)
            }

        # Print statistics for the current dataset
        dataset_trans_stats = compute_stats(dataset_trans_errors)
        dataset_rot_stats = compute_stats(dataset_rot_errors_rad)
        print(f"\nStatistics for {dataset}:")
        print(f"{'Statistic':<30}{'Translational (Meters)':<25}{'Rotational (Radians)':<25}")
        for stat in dataset_trans_stats:
            print(f"{stat:<30}{dataset_trans_stats[stat]:<25.3f}{dataset_rot_stats[stat]:<25.3f}")

        print(f"-" * 50)
        # Plotting and saving logic
        pos_rmse, orientation_rmse = plot_rmse_loss(interpolated_gt, cam_estimates, results, times)
        pos_rmse.savefig(f"{image_dir}/{dataset_name}_ukf_pos_rmse.png")
        orientation_rmse.savefig(f"{image_dir}/{dataset_name}_ukf_orientation_rmse.png")

        # positions_plot, orientations_plot = pos_ypr_plots(interpolated_gt, results, times)
        positions_plot, orientations_plot = pos_ypr_pf_plots(interpolated_gt, cam_estimates, results, times)
        positions_plot.savefig(f"{image_dir}/{dataset_name}_ukf_positions.png")
        orientations_plot.savefig(f"{image_dir}/{dataset_name}_ukf_orientations.png")

        isometric = traj_plot(
            "Isometric View of Ground Truth and Estimated Positions",
            "Ground Truth",
            [Coordinate(x=gt[i].x, y=gt[i].y, z=gt[i].z) for i in range(len(gt))],
            "UKF Estimation",
            [Coordinate(x=position[0], y=position[1], z=position[2]) for position in results]
        )
        isometric.savefig(f"{image_dir}/{dataset_name}_ukf_trajectory.png")
    # Compute overall statistics
    overall_trans_stats = compute_stats(total_trans_errors)
    overall_rot_stats = compute_stats(total_rot_errors_rad)
    overall_rmse_stats = compute_stats(total_cam_positional_rmses, all_ukf_pos_rmses, total_cam_orientation_rmses, all_ukf_orientation_rmses)
    
    # Print overall statistics
    print("\nOverall Absolute Error Statistics:")
    print(f"{'Statistic':<30}{'Translational (Meters)':<25}{'Rotational (Radians)':<25}")
    for stat in overall_trans_stats:
        print(f"{stat:<30}{overall_trans_stats[stat]:<25.3f}{overall_rot_stats[stat]:<25.3f}")
    
    print(f"All Done!")
