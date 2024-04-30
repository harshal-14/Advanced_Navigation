from __future__ import annotations

from typing import List, Tuple,  NamedTuple

import numpy as np
from numpy import identity, zeros
from earth import RATE, gravity_n, principal_radii
from haversine import Unit, haversine
from scipy.linalg import sqrtm
from scipy.spatial.transform import Rotation
import csv


class Data(NamedTuple):
    time: float
    true_lat: float
    true_lon: float
    true_alt: float
    true_roll: float
    true_pitch: float
    true_heading: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    accel_x: float
    accel_y: float
    accel_z: float
    z_lat: float
    z_lon: float
    z_alt: float
    z_VN: float
    z_VE: float
    z_VD: float

def load_data(filename: str) -> List[Data]:
    """
    Read trajectory data from a CSV file and return a list of Data tuples.
    """
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader) 
        return [
            Data(
                time=float(row[0]),
                true_lat=float(row[1]), true_lon=float(row[2]), true_alt=float(row[3]),
                true_roll=float(row[4]), true_pitch=float(row[5]), true_heading=float(row[6]),
                gyro_x=float(row[7]), gyro_y=float(row[8]), gyro_z=float(row[9]),
                accel_x=float(row[10]), accel_y=float(row[11]), accel_z=float(row[12]), 
                z_lat=float(row[13]), z_lon=float(row[14]), z_alt=float(row[15]),
                z_VN=float(row[16]), z_VE=float(row[17]), z_VD=float(row[18])
            ) for row in reader if row
        ]
    
class INSGNSS:
    """
    INSGNSS is a class that implements an Unscented Kalman Filter for
    the integration of INS and GNSS data.
    In this class, we have implemented the UKF for the integration of
    """
    def __init__(self, kappa = 1.0, alpha = 1.0, beta  = 0.4, model_type = "FeedBack", measurement_noise_scale = 5e-3, pred_noise_scale = 1e-3,):
        self.model_type = model_type
        self.n = 12 if model_type == "FeedForward" else 15
        self.measurement_noise_scale = measurement_noise_scale
        self.pred_noise_scale = pred_noise_scale
        self.num_sigma_pts = 2 * self.n + 1

        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta

        self.measurement_noise = identity(self.n) * 1e-3
        self.ng = zeros((3, 1))
        self.na = zeros((3, 1))
        self.mu = zeros((self.n, 1))

        self.initialize_weights()

    def initialize_weights(self):

        """
        Initialize the weights for the sigma points, mean, and covariance
        """

        self._lambda = self.alpha**2 * (self.n + self.kappa) - self.n
        W_mean_0 = self._lambda / (self.n + self._lambda)
        W_cov_0 = W_mean_0 + (1 - self.alpha**2 + self.beta)
        W_i = 1 / (2 * (self.n + self._lambda))

        self.W_mean = zeros(self.num_sigma_pts)
        self.W_cov = zeros(self.num_sigma_pts)
        self.W_mean[0] = W_mean_0
        self.W_mean[1:] = W_i
        self.W_cov[0] = W_cov_0
        self.W_cov[1:] = W_i

    def get_measurements(self, x, gnss):
        """
        Given a state x and a gnss measurement, return the expected
        measurements for the state x.
        """
        adj_noise = np.zeros((self.n, 1))

        if self.model_type == "FeedBack":
            noise_scale = 2e-3
        else:
            noise_scale = 5e-3
        noise = np.random.normal( scale=self.measurement_noise_scale, size=(self.n, self.n))
        cov_m = np.zeros((self.n, self.n))
        cov_m[0:6, 0:6] = np.eye(6)
        if self.model_type == "FeedBack":
            cov_m[6:9, 6:9] = np.eye(3)

        R = np.diag(noise).reshape(self.n, 1)
        adj_noise[0 : self.n] = np.dot(cov_m, x) + R

        return adj_noise

    def compute_sigma_pts(self, dist_mean, sigma):
        """
        Compute sigma points for the given mean and covariance using Julier's method
        """
        sigma_pts = np.zeros((self.num_sigma_pts, self.n, 1))

        sigma_pts[0] = dist_mean

        try:
            S = sqrtm((self.n + self.kappa) * sigma)
        except:
            raise "Matrix is not positive definite."

        for i in range(self.n):
            sigma_pts[i + 1] = dist_mean + S[i].reshape((self.n, 1))
            sigma_pts[self.n + i + 1] = dist_mean - S[i].reshape((self.n, 1))

        return sigma_pts

    def propagation_model(self, x, dt, fb, wb):
        """
        Given a x, dt, and existing accelerations, return
        the predicted x after dt time has passed.
        Input: x - the current state
                dt - the time step
                fb - the feedback accelerations
                wb - the angular velocities
        Output: the predicted state after dt time has passed
        """
        # Get the current state x params
        L = x[0]
        lambda_ = x[1]
        h = x[2]
        phi = x[3]
        theta = x[4]
        psi = x[5]
        Vn = x[6]
        Ve = x[7]
        Vd = x[8]

        v_n = np.array([Vn, Ve, Vd]).reshape(3, 1)

        if self.model_type == "FeedBack":
            fb -= x[9:12]
            wb -= x[12:15]

        R_nb_minus_1 = Rotation.from_euler("xyz", np.array([phi, theta, psi]).reshape((3,)), degrees=True).as_matrix()

        # Attitude dynamics equations to update the attitude
        w_ei_skew = np.array([[0, -RATE, 0], [RATE, 0, 0], [0, 0, 0]])

        Rn, Re, RecosL = principal_radii(L, h)

        w_ne = np.zeros((3,))
        w_ne[0] = Ve / Re
        w_ne[1] = -Vn / Rn
        w_ne[2] = -(Ve * np.tan(np.deg2rad(L))) / Re

        w_ne_skew = np.array([ [0, -w_ne[2], w_ne[1]],
                                    [w_ne[2], 0, -w_ne[0]],
                                    [-w_ne[1], w_ne[0], 0],])

        w_ne = w_ne.reshape((3, 1))

        wb = wb.reshape((3,))
        w_bi_skew = np.array([[0, -wb[2], wb[1]], 
                                [wb[2], 0, -wb[0]], 
                                [-wb[1], wb[0], 0]])
        wb = wb.reshape((3, 1))

        R_nb = R_nb_minus_1 * (np.eye(3) + w_bi_skew * dt) - ( (w_ei_skew + w_ne_skew) * dt * R_nb_minus_1 )

        # Velocity Update
        f_nt = 1 / 2 * np.dot(R_nb_minus_1 + R_nb, fb)

        v_nt = v_n + dt * (
            f_nt
            + gravity_n(L, h).reshape((3, 1))
            - np.dot(w_ne_skew + 2 * w_ei_skew, v_n)
        )

        # Position Update
        h_next = h - (dt / 2) * (Vd + v_nt[2])
        Rnnew, _, _ = principal_radii(L, h_next)
        L_next = L
        L_next += (dt / 2) * (Vn / Rn + v_nt[0] / Rn)
        L_next += (dt / 2) * (Vn / Rn + v_nt[0] / Rnnew)

        _, _, RecosLnext = principal_radii(L_next, h_next)

        lambda_next = lambda_
        lambda_next += (dt / 2) * (Ve / RecosL)
        lambda_next += (dt / 2) * (Ve / RecosLnext)

        phi, theta, psi = Rotation.as_euler(Rotation.from_matrix(R_nb), "xyz", degrees=True,)

        new_x = np.zeros((self.n, 1))
        new_x[0] = L_next
        new_x[1] = lambda_next
        new_x[2] = h_next
        new_x[3] = phi
        new_x[4] = theta
        new_x[5] = psi
        new_x[6:9] = v_nt.reshape((3, 1))
        new_x[9:] = x[9:]

        return new_x

    def update(self, gnss, dist_mean, sigma, sigma_pts,):
        """
        update takes the gnss, dist_meanbar, and sigmabar and performs the
        update step
        """
        # Apply the measurement function across each new sigma point
        measured_pts = np.zeros_like(sigma_pts)
        for i in range(self.num_sigma_pts):
            measured_pts[i] = self.get_measurements(sigma_pts[i], gnss)

        # Calculate the mean of the measurement points by their respective
        # weights. The weights have a 1/N term so the mean is calculated
        # through their addition
        zhat = np.zeros((self.n, 1))
        for i in range(0, self.num_sigma_pts):
            zhat += self.W_mean[i] * measured_pts[i]

        St = np.zeros((self.n, self.n))
        z_diff = measured_pts - zhat
        for i in range(0, self.num_sigma_pts):
            St += self.W_cov[i] * np.dot(z_diff[i], z_diff[i].T)

        sigmapred_t = np.zeros((self.n, self.n))
        x_diff = sigma_pts - dist_mean
        for i in range(0, self.num_sigma_pts):
            sigmapred_t += self.W_cov[i] * np.dot(
                x_diff[i], z_diff[i].T
            )

        K = np.dot(sigmapred_t, np.linalg.pinv(St))

        # Update the mean and cov
        curr_pos = dist_mean + np.dot(K, gnss - zhat)
        cov = sigma - np.dot(K, St).dot(K.T)
        # Ensure the covariance matrix is symmetric and positive definite
        cov = (cov + cov.T) / 2
        eig_val, eig_vec = np.linalg.eig(cov)
        eig_val[eig_val < 0] = 0
        eig_val += 1e-3  # Adjust this jitter value as needed
        cov = eig_vec.dot(np.diag(eig_val)).dot(eig_vec.T)

        return curr_pos, cov

    def predict(self, sigma_pts, feedback, wb, dt):
        """
        predict takes the current sigma points for the estimate and performs
        the x transition across them. We then compute the mean and the
        cov of the resulting transformed sigma points.
        """
        # For each sigma point, run them through our x transition function
        transformed_pts = np.zeros_like(sigma_pts)
        for i in range(sigma_pts.shape[0]):
            transformed_pts[i, :] = self.propagation_model(sigma_pts[i], dt, wb, feedback)

        # Calculate the mean of the transitioned points by their respective weights. The weights
        # contain a 1/N term so we are effectively finding the average. We expect a Nx1 output
        dist_mean = np.zeros((self.n, 1))
        for i in range(0, self.num_sigma_pts):
            dist_mean += self.W_mean[i] * transformed_pts[i]

        if self.model_type == "FeedForward":
            noise_scale = 1e-3
        else:
            noise_scale = 5e-5
        Q = np.random.normal(scale=self.pred_noise_scale, size=(self.n, self.n))

        diff = transformed_pts - dist_mean
        sigma = np.zeros((self.n, self.n))
        for i in range(0, self.n):
            sigma += self.W_cov[i] * np.dot(diff[i], diff[i].T)
        sigma += Q

        return dist_mean, sigma, transformed_pts

    def data_from_sensor(self, data: Data) -> dict:
        """
        Given a data point, return the IMU, GNSS, and pose data in a dictionary
        """
        imu_data = np.zeros((6, 1))
        imu_data[0] = data.accel_x
        imu_data[1] = data.accel_y
        imu_data[2] = data.accel_z
        imu_data[3] = data.gyro_x
        imu_data[4] = data.gyro_y
        imu_data[5] = data.gyro_z

        gnss_data = np.zeros((self.n, 1))
        gnss_data[0] = data.z_lat
        gnss_data[1] = data.z_lon
        gnss_data[2] = data.z_alt
        gnss_data[6] = data.z_VN
        gnss_data[7] = data.z_VE
        gnss_data[8] = data.z_VD

        pose_data = np.zeros((self.n, 1))
        pose_data[0] = data.true_lat
        pose_data[1] = data.true_lon
        pose_data[2] = data.true_alt
        pose_data[3] = data.true_roll
        pose_data[4] = data.true_pitch
        pose_data[5] = data.true_heading

        return {
            "imu": imu_data,
            "gnss": gnss_data,
            "pose": pose_data
        }

    def run(self, data: List[Data]):
        """
        Run the UKF for the given data and return the filtered positions
        and the haversine distances between the filtered positions and
        the ground truth.
        """
        filtered_pos = []
        haversine_distances = []

        # Initialization with the first data point
        sensor_data = self.data_from_sensor(data[0])
        x = sensor_data["pose"]
        previous_x_timestamp = data[0].time
        process_cov_matrix = np.eye(self.n) * 1e-3

        for index, read in enumerate(data):
            if index == 0:
                continue

            sensor_data = self.data_from_sensor(read)
            dt = read.time - previous_x_timestamp
            previous_x_timestamp = read.time
            feedback = sensor_data["imu"][0:3]
            wb = sensor_data["imu"][3:6]
            gnss = sensor_data["gnss"]

            sigma_pts = self.compute_sigma_pts(x, process_cov_matrix)
            dist_meanbar, sigmabar, transformed_pts = self.predict(sigma_pts, feedback, wb, dt)
            if self.model_type == "FeedForward":
                dist_meanbar
            dist_mean, sigma = self.update(gnss, dist_meanbar, sigmabar, transformed_pts)

            process_cov_matrix = sigma
            x = dist_mean

            if self.model_type == "FeedForward":
                dist_meanbar[9:12] = dist_meanbar[0:3] - gnss[0:3]

            filtered_pos.append(x)
            haversine_distances.append(
                haversine((x[0], x[1]), (sensor_data["pose"][0], sensor_data["pose"][1]), unit=Unit.DEGREES,)
            )

        return filtered_pos, [sd["pose"] for sd in [self.data_from_sensor(d) for d in data]], haversine_distances
