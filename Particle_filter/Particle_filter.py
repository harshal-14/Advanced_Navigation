import numpy as np
import matplotlib.pyplot as plt
from observation_model import ObservationModel, pose_to_ypr
from utils import Data, Marker, ActualData, Coordinate, PixelCoordinate
import time
from math import cos, sin
from scipy.stats import multivariate_normal
from typing import List, NamedTuple, Union
from tqdm import tqdm

estimated_cov = np.array([[ 0.00787652,  0.0002269,   0.00462655,  0.00362929,  0.00697831, -0.00029106],
 [ 0.0002269,   0.00483996, -0.00177252, -0.00401243,  0.0024168,   0.00014734],
 [ 0.00462655, -0.00177252,  0.00731506,  0.00361734,  0.00315442, -0.00074367],
 [ 0.00362929, -0.00401243,  0.00361734,  0.00530187,  0.00134563, -0.00023165],
 [ 0.00697831,  0.0024168,   0.00315442,  0.00134563,  0.0072551,  -0.00016959],
 [-0.00029106,  0.00014734, -0.00074367, -0.00023165, -0.00016959,  0.00010396]])


class ParticleFilter:
    def __init__(self, num_particles, final_min_num_particles = 0.5, covariance_matrix = estimated_cov):
        self.num_particles = num_particles
        self.final_min_num_particles = 0.5 * num_particles
        self.covariance_matrix = covariance_matrix
        self.observation_model = ObservationModel()
                      
    def predict(
        self,
        particles: np.ndarray,
        dt: float,
        acc: np.ndarray,
        gyro: np.ndarray,
    ) -> np.ndarray:
        """
        Predict the next state of the particles using the state transition model
        Inputs:
            particles: np.ndarray of shape (num_particles, 15, 1)
            dt: float, the time step for the prediction
            acc: np.ndarray of shape (3,), the accelerometer measurement
            gyro: np.ndarray of shape (3,), the gyroscope measurement
        Outputs:
            particles: np.ndarray of shape (num_particles, 15, 1), the predicted particles
        """

        # We create a noise vector to add to our particles at
        # the end of the state transition process
        noise = np.zeros((self.num_particles, 6, 1))

        noise[:, 0:3] = np.random.normal(
            scale=105, size=(self.num_particles, 3, 1)
        )
        noise[:, 3:6] = np.random.normal(
            scale=0.01, size=(self.num_particles, 3, 1)
        )

        xdot = np.zeros((self.num_particles, 15, 1))
        # ua and uw are the bias from the acc and gyro
        # respectively.
        ua = np.tile(acc.reshape((3, 1)), (self.num_particles, 1, 1))
        uw = np.tile(gyro.reshape(3, 1), (self.num_particles, 1, 1))
        g = -9.81

        # Add noise to the accelerometer, gyro will add noise at the end
        ua = ua + noise[:, 0:3]

        # Extract the orientation, and velocities from the state
        orientations = particles[:, 3:6]
        velocities = particles[:, 6:9]

        thetas = orientations[:, 0]
        phis = orientations[:, 1]
        psis = orientations[:, 2]


        #Analytically calculated G_inverse for faster computation than np.linalg.inv and
        # also resolve the issue of singular matrix, not symmetric matrix. This is much better than np.linalg.inv
        # taken from Keith's implementation
        G_inv = np.zeros((self.num_particles, 3, 3, 1))
        G_inv[:, 0, 0] = np.cos(thetas)
        G_inv[:, 0, 2] = np.sin(thetas)
        G_inv[:, 1, 0] = np.sin(phis) * np.sin(thetas) / np.cos(phis)
        G_inv[:, 1, 1] = 1.0
        G_inv[:, 1, 2] = -np.cos(thetas) * np.sin(phis) / np.cos(phis)
        G_inv[:, 2, 0] = -np.sin(thetas) / np.cos(phis)
        G_inv[:, 2, 2] = np.cos(thetas) / np.cos(phis)

        R = np.zeros((self.num_particles, 3, 3, 1))
        R[:, 0, 0] = np.cos(psis) * np.cos(thetas) - np.sin(phis) * np.sin(
            phis
        ) * np.sin(thetas)
        R[:, 0, 1] = -np.cos(phis) * np.sin(psis)
        R[:, 0, 2] = np.cos(psis) * np.sin(thetas) + np.cos(thetas) * np.sin(
            phis
        ) * np.sin(psis)
        R[:, 1, 0] = np.cos(thetas) * np.sin(psis) + np.cos(psis) * np.sin(
            phis
        ) * np.sin(thetas)
        R[:, 1, 1] = np.cos(phis) * np.cos(psis)
        R[:, 1, 2] = np.sin(psis) * np.sin(thetas) - np.cos(psis) * np.cos(
            thetas
        ) * np.sin(phis)
        R[:, 2, 0] = -np.cos(phis) * np.sin(thetas)
        R[:, 2, 1] = np.sin(phis)
        R[:, 2, 2] = np.cos(phis) * np.cos(thetas)

        xdot[:, 0:3] = velocities

        # Vector Broadcasting through Numpy fails due to the shape of the matrices,
        # So we iterate through all the particles to calculate the xdot
        # for each particle. This is a bit slower than vectorized operations, however it works!
        for i in range(self.num_particles):
            xdot[i, 3:6] = np.dot(G_inv[i].reshape((3, 3)), uw[i].reshape((3, 1)))
            xdot[i, 6:9] = np.array([0, 0, g]).reshape((3, 1)) + np.dot(
                R[i].reshape((3, 3)), ua[i].reshape((3, 1))
            )

        # Add our xdot delta to our particles

        xdot = xdot * dt
        # Add noise again for gyro
        xdot[:, 3:6] = xdot[:, 3:6] + noise[:, 3:6]
        particles = particles + xdot

        return particles

    def update(self, measurement):
        #Use the observation model and the measurement to calculate its importance weight for each particle and then use the low varinance resampling algorithm to find updated particle set
        # Use the covariance values as a starting point for the observation model.
        measurement_matrix = np.zeros((6, 15))
        measurement_matrix[0:6, 0:6] = np.eye(6)
        updated_measurement = (measurement_matrix @ measurement) + np.diag(self.covariance_matrix).reshape(6, 1)
        return updated_measurement

    def resampling(self, particles, W):
        #Use the low variance resampling algorithm to find the updated particle set
        # W is the weights of the particles
        # Initialize the resampled particles and weights sizes to be the same as the original particles
        resampled_particles = np.zeros_like(particles)
        resampled_weights = np.zeros((self.num_particles, 1))
        # Calculate the cumulative sum of the weights to use for the resampling
        cumulative_sum = np.cumsum(W)
        idx = 0
        step = 1 / self.num_particles
        r = np.random.uniform(0, step)
        # Resample the particles based on the cumulative sum of the weights
        for i in range(self.num_particles):
            sample_point = r + i * step
            while idx < len(W) and cumulative_sum[idx] < sample_point:
                idx += 1

            resampled_particles[i] = particles[idx]
            resampled_weights[i] = W[idx]
        # Normalize the weights
        W_total = np.sum(resampled_weights)
        if W_total > 0:
            resampled_weights /= W_total

        return resampled_particles, resampled_weights
    
    def get_measurment(self, data: Data):
        # Use the observation model to estimate the position and orientation of the drone from the data
        if data.tags is None:
            raise ValueError("No tags found in the data")
        # Estimate the position and orientation of the drone from the data
        orientation, position = self.observation_model.estimate_pose(data.tags)
        x = np.zeros((15, 1))
        x[0:3] = position.reshape(3,1)
        x[6:9] = np.array(orientation).reshape(3,1)
        return x

    def update_weights(self, particles, pred):
        #Calculate the weights of the particles based on the difference between the predicted and actual measurements
        # One issue is weight overflow, adding jitter to the weights could help but that messes up the filter estimates
        ERROR = particles[:, 0:6] - pred[0:6]
        # We calculate the weights using the exponential of the negative of the squared error
        weights = np.exp(-0.5* np.sum(ERROR **2, axis=1))
        # Normalize the weights
        norm_weights = weights / np.sum(weights)

        return norm_weights
    
    def weighted_average(
        self, particles: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the weighted average of the particles based on their weights
        """
        # We multiply each particle by its weight and sum them all up
        weighted_particles = particles * weights.reshape(self.num_particles, 1, 1)
        summed_particles = np.sum(weighted_particles, axis=0)

        return summed_particles / np.sum(weights)

    def initial_particles(self):
        """
        We initialize the particles by sampling from a uniform distribution, with the following limits:
        x, y, z: 0 to 3
        roll, pitch, yaw: -0.5*pi to 0.5*pi
        """
        x_lims = (0., 3)
        y_lims = (0., 3)
        z_lims = (0., 1.5)

        yaw_lims = (-0.5*np.pi, 0.5*np.pi)
        pitch_lims = (-0.5*np.pi, 0.5*np.pi)
        roll_lims = (-0.5*np.pi, 0.5*np.pi)

        mins = np.array([x_lims[0], y_lims[0], z_lims[0], roll_lims[0], pitch_lims[0], yaw_lims[0]])
        maxss = np.array([x_lims[1], y_lims[1], z_lims[1], roll_lims[1], pitch_lims[1], yaw_lims[1]])

        particles = np.random.uniform(
            low=mins, high=maxss, size=(self.num_particles, 6)
        )
        particles = np.hstack((particles, np.zeros((self.num_particles, 9))))

        particles = particles.reshape((self.num_particles, 15, 1))
        return particles
    
    def run_filter(self, est_pos: List[Data], type = "weighted_average"):
        estimates = np.zeros((len(est_pos) - 1, 15, 1))  # Pre-allocate space for all estimates
        particles = self.initial_particles()
        W = np.ones((self.num_particles, 1)) / self.num_particles
        time = est_pos[0].timestamp

        for idx, pos in tqdm(enumerate(est_pos[1:]), total=len(est_pos)-1, desc="Filtering Particles"):
            dt = pos.timestamp - time
            time = pos.timestamp
            particles = self.predict(particles, dt, pos.acc, pos.omg)

            measurement = self.get_measurment(pos)
            pred = self.update(measurement)
            pred = np.concatenate((pred, np.zeros((9,1))))

            W = self.update_weights(particles, pred)
            if type == "weighted_average":
                estimate = self.weighted_average(particles, W)
            elif type == "max":
                estimate = particles[np.argmax(W)]
            elif type == "average":
                estimate = (np.sum(particles, axis=0) / self.num_particles)
            # estimate = self.weighted_average(particles, W)
            particles, W = self.resampling(particles, W)

            estimates[idx - 1] = estimate  # Store each estimate

        return estimates
