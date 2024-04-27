from earth import principal_radii, gravity, gravity_n, curvature_matrix, rate_n
import numpy as np
import scipy.spatial.transform as tf
import pandas as pd
from math import cos, tan
import os
from scipy.spatial.transform import Rotation as R
from scipy.linalg import cholesky, sqrtm

# INS_GNSS.py
class INSGNSS:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()
        self.lat = None
        self.alt = None
        self.RATE = 7.2921159e-5
        self.n = 15 # Number of states
        self.n_sigma_pts = 2*self.n + 1 
        self.alpha = 0.01
        self.beta = 1.
        self.kappa = 0.5
        self.ng = np.zeros((3, 1))          # ng and na are the biases from the accelerometer and gyroscope
        self.na = np.zeros((3, 1))
        self.dist_mean = np.zeros((self.n, 1)) # Mean of the distribution
        self.init_weights()
        self.sigma = np.eye(self.n) # Covariance matrix of the distribution

    def init_weights(self):
        """Initialize weights for the sigma points in the UKF."""
        lambda_ = self.alpha ** 2 * (self.n + self.kappa) - self.n
        self.W_mean = np.full(self.n_sigma_pts, 1 / (2 * (self.n + lambda_)))
        self.W_mean[0] = lambda_ / (self.n + lambda_)
        self.W_cov = np.copy(self.W_mean)
        self.W_cov[0] += 1 - self.alpha ** 2 + self.beta

    def load_data(self):
        # Ensure the file exists
        if os.path.exists(self.file_path):
            # Load data using pandas
            data = pd.read_csv(self.file_path)
            return data
        else:
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        
    def attitude_update(self, R_b_n_minus_1, w_i_b, b_g, dt, v_n, lat, alt):
        # Subtract gyroscope biases from the measurements to get bias-corrected rates
        w_i_b_corrected = w_i_b - b_g

        # Earth's rotation rate (skew-symmetric matrix)
        Omega_e_i = np.array([[0, -self.RATE, 0],
                            [self.RATE, 0, 0],
                            [0, 0, 0]])

        # Transport rate (requires velocity and latitude)
        rn, re, _ = principal_radii(lat, alt)  
        w_e_n = np.array([v_n[1] / (rn + alt), -v_n[0] / (re + alt), -v_n[1] * tan(np.deg2rad(lat)) / (rn + alt)])
        #skew symmetric matrix of w_e_n
        Omega_e_n = np.array([[0, -w_e_n[2], w_e_n[1]],
                            [w_e_n[2], 0, -w_e_n[0]],
                            [-w_e_n[1], w_e_n[0], 0]])
        
        # Gyroscope measurements (body rates in skew-symmetric matrix)
        Omega_i_b = np.array([[0, -w_i_b_corrected[2], w_i_b_corrected[1]],
                            [w_i_b_corrected[2], 0, -w_i_b_corrected[0]],
                            [-w_i_b_corrected[1], w_i_b_corrected[0], 0]])

        # Rotation matrix update using first-order approximation
        R_n_b = R_b_n_minus_1 @ (np.eye(3) + Omega_i_b * dt) - (Omega_e_i + Omega_e_n) @ R_b_n_minus_1 * dt

        return R_n_b, Omega_e_n, Omega_e_i

    def velocity_update(self, v_n_minus_1, f_b, dt, R_b_n_minus_1, R_n_b, lat, alt, Omega_e_n, Omega_e_i):
        # Correct accelerometer measurements for biases if you have accelerometer bias b_a available
        b_a = np.array([0, 0, 0])
        f_b_corrected = f_b - b_a

        g_n = gravity_n(lat, alt)
        f_n = 0.5 * (R_b_n_minus_1 + R_n_b) @ f_b_corrected

        # Omega_ie_n = rate_n(lat)
        Omega_e_i = np.array([[0, -self.RATE, 0],
                    [self.RATE, 0, 0],
                    [0, 0, 0]])

        coriolis = -(Omega_e_n + 2*Omega_e_i) @ v_n_minus_1

        v_n = v_n_minus_1 + dt * (f_n + g_n + coriolis)

        return v_n
    
    def predict(self, x, P, dt):
        # Create sigma points
        sigma_points = self.compute_sigma_pts(x, P)
        
        # Propagate each sigma point
        for i in range(self.n_sigma_pts):
            sigma_points[i] = self.propagate_sigma_point(sigma_points[i], dt)
        
        # Predict state mean
        x_pred = np.dot(self.W_mean, sigma_points)
        
        # Predict state covariance
        P_pred = np.zeros((self.n, self.n))
        for i in range(self.n_sigma_pts):
            diff = sigma_points[i] - x_pred
            P_pred += self.W_cov[i] * np.outer(diff, diff)
        
        return x_pred, P_pred

    def propagate_sigma_point(self, x, dt):
        # Unpack the state vector
        lat, lon, alt, phi, theta, psi, vn, ve, vd, bax, bay, baz, bgx, bgy, bgz = x.ravel()

        # Convert Euler angles to rotation matrix
        euler_angles = np.array([phi, theta, psi])
        R_b_n = R.from_euler('xyz', euler_angles, degrees=False).as_matrix() # Ensure that angles are in radians

        # Prepare the angular rates and specific forces for the attitude update
        w_i_b = np.array([bgx, bgy, bgz]) # The angular rates
        f_b = np.array([bax, bay, baz]) 

        # Propagate the state using the dynamic model
        R_n_b_updated, Omega_e_n, Omega_e_i = self.attitude_update(R_b_n, w_i_b, np.array([bgx, bgy, bgz]), dt, np.array([vn, ve, vd]), lat, alt)
        v_n_updated = self.velocity_update(np.array([vn, ve, vd]), f_b, dt, R_b_n, R_n_b_updated, lat, alt, Omega_e_n, Omega_e_i)
        h_t, lat_t, lon_t, euler_angles_updated = self.position_update(v_n_updated, np.array([vn, ve, vd]), alt, dt, lat, lon, R_n_b_updated)

        # Construct the updated state vector and return
        x_updated = np.hstack([lat_t, lon_t, h_t, euler_angles_updated, v_n_updated, [bax, bay, baz, bgx, bgy, bgz]])
        return x_updated

    def position_update(self, v_n, v_n_minus_1, h_t_minus_1, dt, lat_t_minus_1, lon_t_minus_1, R_b_n):
        rn, re, _ = principal_radii(lat_t_minus_1, h_t_minus_1)

        h_t = h_t_minus_1 - 0.5 * dt * (v_n[2] + v_n_minus_1[2])
        
        lat_t = lat_t_minus_1 + 0.5 * dt * ((v_n[0] / (rn + h_t_minus_1)) + (v_n_minus_1[0] / (rn + h_t)))
        
        lon_t = lon_t_minus_1 + 0.5 * dt * ((v_n[1] / ((re + h_t_minus_1) * np.cos(np.deg2rad(lat_t_minus_1)))) + 
                                            (v_n_minus_1[1] / ((re + h_t) * np.cos(np.deg2rad(lat_t)))))
        q_t = self.rotation_matrix_to_euler_angles(R_b_n)
        
        # Update self.lat and self.lon with the new values
        self.lat = lat_t
        self.lon = lon_t
        
        return h_t, lat_t, lon_t, q_t

    @staticmethod
    def rotation_matrix_to_euler_angles(R_i):
        # Create a Rotation object from the rotation matrix
        r = R.from_matrix(R_i)
        # Get the Euler angles in degrees
        euler_angles = r.as_euler('xyz', degrees=True)
        return euler_angles

    def compute_sigma_pts(self, dist_mean, sigma):
        lambda_ = self.alpha ** 2 * (self.n + self.kappa) - self.n
        scaling_factor = self.n + self.kappa
        try:
            # Attempting Cholesky decomposition
            S = cholesky(scaling_factor * sigma)
        except np.linalg.LinAlgError:
            jitter = np.eye(self.n) * 1e-3 
            while True:
                try:
                    S = cholesky(scaling_factor * sigma + jitter)
                    break  # Success, exit the loop
                except np.linalg.LinAlgError:
                    jitter *= 1.01
            
        sigma_pts = np.zeros((self.n_sigma_pts, self.n))
        sigma_pts[0] = dist_mean
        for i in range(self.n):
            spread = S[:, i]
            sigma_pts[i + 1] = dist_mean + spread
            sigma_pts[i + 1 + self.n] = dist_mean - spread
        
        return sigma_pts
    
    def measurement_model(self, x):
        # Return the position part of the state
        return x[:3]
    
    def update(self, x_pred, P_pred, z):
        # Generate sigma points from predicted state and covariance
        sigma_points_pred = self.compute_sigma_pts(x_pred, P_pred)
        
        # Predict measurements for each sigma point
        Z_pred = np.array([self.measurement_model(sp) for sp in sigma_points_pred])
        
        # Measurement prediction
        z_pred = np.dot(self.W_mean, Z_pred)
        
        # Measurement covariance
        P_zz = np.zeros((3, 3))
        for i in range(self.n_sigma_pts):
            diff_z = Z_pred[i] - z_pred
            P_zz += self.W_cov[i] * np.outer(diff_z, diff_z)
        
        # State-measurement cross-covariance
        P_xz = np.zeros((self.n, 3))
        for i in range(self.n_sigma_pts):
            diff_x = sigma_points_pred[i] - x_pred
            diff_z = Z_pred[i] - z_pred
            P_xz += self.W_cov[i] * np.outer(diff_x, diff_z)
        
        # Kalman gain
        K = np.dot(P_xz, np.linalg.inv(P_zz))
        
        # Update state and covariance with actual measurement
        x_updated = x_pred + np.dot(K, (z - z_pred))
        P_updated = P_pred - np.dot(K, np.dot(P_zz, K.T))
        
        return x_updated, P_updated

    def visualize():
        #Visualze the trajectory on a map
        pass
        
def main():
    ins_gnss = INSGNSS("/home/lucifer/WPI/Spring_courses/Advanced_Navigation/INS-GNSS/trajectory_data.csv")

    # Initial state x and covariance P need to be defined
    x = np.zeros(ins_gnss.n)  # Small initial state
    P = np.eye(ins_gnss.n) * 0.001  # Small initial uncertainty

    previous_time = ins_gnss.data.iloc[0]['time']  # Initial time
    for i in range(1, len(ins_gnss.data)):
        current_time = ins_gnss.data.iloc[i]['time']
        dt = current_time - previous_time
        z = ins_gnss.data.iloc[i][['true_lat', 'true_lon', 'true_alt']].to_numpy()  # Extract true position as measurement

        # Prediction Step
        x_pred, P_pred = ins_gnss.predict(x, P, dt)
        
        # Update Step
        x, P = ins_gnss.update(x_pred, P_pred, z)
        print(f"Updated state at time {current_time}: {x}")
        print(f"Updated covariance at time {current_time}: \n{P}")

        previous_time = current_time  # Update the time for the next iteration

# def main():
#     ins_gnss = INSGNSS("/home/lucifer/WPI/Spring_courses/Advanced_Navigation/INS-GNSS/trajectory_data.csv")
#     data = ins_gnss.data

    # # Initialize state variables
    # R_b_n_minus_1 = np.eye(3)  # Initial rotation matrix
    # v_n_minus_1 = np.zeros(3)  # Initial velocity
    # h_t_minus_1 = data.iloc[0]['true_alt']  # Initial altitude
    # lat_t_minus_1 = data.iloc[0]['true_lat']  # Initial latitude
    # lon_t_minus_1 = data.iloc[0]['true_lon']  # Initial longitude

    # # Time step (assuming data is in seconds and is sequential)
    # dt = 1 

    # for i in range(1, len(data)):
    #     # Extract the measurements for this timestep
    #     w_i_b = data.loc[i, ['gyro_x', 'gyro_y', 'gyro_z']].to_numpy()
    #     f_b = data.loc[i, ['accel_x', 'accel_y', 'accel_z']].to_numpy()
    #     lat = data.loc[i, 'true_lat']
    #     alt = data.loc[i, 'true_alt']
    #     lon = data.loc[i, 'true_lon']

    #     # Perform the updates
    #     R_b_n = ins_gnss.attitude_update(R_b_n_minus_1, w_i_b, np.zeros(3), dt, v_n_minus_1, lat_t_minus_1, h_t_minus_1)
    #     # print(f"R_b_n: {R_b_n}")
    #     v_n = ins_gnss.velocity_update(v_n_minus_1, f_b, dt, R_b_n_minus_1, R_b_n, lat_t_minus_1, h_t_minus_1)
    #     # print(f"v_n: {v_n}")
    #     h_t, lat_t, lon_t, q_t = ins_gnss.position_update(v_n, v_n_minus_1, h_t_minus_1, dt, lat_t_minus_1, lon_t_minus_1, R_b_n)

    #     # Prepare for the next iteration
    #     R_b_n_minus_1 = R_b_n
    #     v_n_minus_1 = v_n
    #     h_t_minus_1 = h_t
    #     lat_t_minus_1 = lat_t
    #     lon_t_minus_1 = lon_t

if __name__ == "__main__":
    main()

