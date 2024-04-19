import numpy as np
import matplotlib.pyplot as plt
from observation_model import ObservationModel, pose_to_ypr
from utils import Data, Marker, ActualData, Coordinate, PixelCoordinate
import time
from math import cos, sin
from scipy.stats import multivariate_normal
from typing import List, NamedTuple, Union

class ParticleFilter:
    def __init__(self, num_particles, final_min_particle_count, covariance_matrix):
        self.num_particles = num_particles
        self.final_min_particle_count = 0.5 * num_particles
        self.covariance_matrix = covariance_matrix
        self.observation_model = ObservationModel()
        
    def predict(self, particles, dt, acc, omg):
        #For each particle, take the original particle, sample from the noise distribution and then apply the motion model
        #Use Measured inputs plus noise to determine the future state of the particle using process model
        #Use noise Covariance values as starting point for the process noise
        noise = np.zeros((self.num_particles, 6, 1))
        noise[:, 0:3] = np.random.normal(105, size=(self.num_particles, 3, 1)) # 105 is the standard deviation of the acceleration
        noise[:, 3:6] = np.random.normal(0.01, size=(self.num_particles, 3, 1)) # 0.01 is the standard deviation of the gyroscope
        
        xdot = np.zeros((self.num_particles, 15, 1))
        #Bias from the accelerometer and gyroscope
        ua = np.tile(acc.reshape((3,1)), (self.num_particles, 1, 1))
        uw = np.tile(omg.reshape((3,1)), (self.num_particles, 1, 1))

        ua = ua + noise[:, 0:3]
        uw = uw + noise[:, 3:6]

        orientations = particles[:, 3:6]
        vel = particles[:, 6:9]

        theta = orientations[:, 0]
        phi = orientations[:, 1]
        psi = orientations[:, 2]

        #Calculate the rotation matrix
        R = np.zeros((self.num_particles, 3, 3, 1))
        R[:, 0, 0] = cos(psi) * cos(theta) - sin(phi) * sin(phi)
        R[:, 0, 1] = -cos(phi) * sin(psi)
        R[:, 0, 2] = cos(psi) * sin(theta) + sin(phi) * cos(theta) * sin(psi)    
        R[:, 1, 0] = cos(theta) * sin(psi) + cos(psi) * sin(phi) * sin(theta)
        R[:, 1, 1] = cos(phi) * cos(psi)
        R[:, 1, 2] = sin(psi) * sin(theta) - sin(phi) * cos(theta) * cos(psi)
        R[:, 2, 0] = -cos(phi) * sin(theta)
        R[:, 2, 1] = sin(phi)
        R[:, 2, 2] = cos(phi) * cos(theta)
        
        #Calculate G matrix
        # G[0, 0] = cos(theta)
        # G[0, 2] = -cos(phi) * sin(theta)
        # G[1, 1] = 1
        # G[1, 2] = sin(phi)
        # G[2, 0] = sin(theta)
        # G[2, 2] = cos(phi) * cos(theta)
        G = np.zeros((self.num_particles, 3, 3, 1))
        G[:, 0, 0] = cos(theta)
        G[:, 0, 2] = -cos(phi) * sin(theta)
        G[:, 1, 1] = 1
        G[:, 1, 2] = sin(phi)
        G[:, 2, 0] = sin(theta)
        G[:, 2, 2] = cos(phi) * cos(theta)

        xdot[:, 0:3] = vel
        for i in range(self.num_particles):
            xdot[i, 3:6] = np.linalg.inv(G[i]).reshape((3,3)) @ uw[i].reshape((3,1))
            xdot[i, 6:9] = np.array([0, 0, 9.81]).reshape((3,1)) + R[i].reshape((3,3)) @ ua[i].reshape((3,1))

        xdot = xdot * dt
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
        resampled_particles = np.zeros_like(particles)
        resampled_weights = np.zeros((self.num_particles, 1))

        cumulative_sum = np.cumsum(W)
        idx = 0
        step = 1 / self.num_particles
        r = np.random.uniform(0, step)
        for i in range(self.num_particles):
            sample_point = r + i * step
            while idx < len(W) and cumulative_sum[idx] < sample_point:
                idx += 1

            resampled_particles[i] = particles[idx]
            resampled_weights[i] = W[idx]
        
        W_total = np.sum(resampled_weights)
        if W_total > 0:
            resampled_weights /= W_total

        return resampled_particles, resampled_weights
    
    def get_measurment(self, data: Data):
        if data.tags is None:
            raise ValueError("No tags found in the data")
        
        orientation, position = self.observation_model.estimate_pose(data.tags)
        x = np.zeros((15, 1))
        x[0:3] = position.reshape(6,1)
        x[6:9] = orientation.reshape(6,1)
        return x

    def update_weights(self, particles, pred):
        ERROR = particles[:, 0:6] - pred[0:6]

        weights = np.exp(-0.5* np.sum(ERROR **2, axis=1))
        norm_weights = weights / np.sum(weights)

        return norm_weights
    
    def weighted_average(
        self, particles: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """
        Given the particles and their weights, calculate a singular point
        from their weighted average
        """
        # We need to reshape the weights to a (particle_count, 1, 1) to
        # make the multiplication work the way we'd expect across each
        # particle.
        weighted_particles = particles * weights.reshape(self.particle_count, 1, 1)
        summed_particles = np.sum(weighted_particles, axis=0)

        return summed_particles / np.sum(weights)

    def initial_particles(self):
        x_range = (0., 3)
        y_range = (0., 3)
        z_range = (0., 1.5)

        yaw_range = (-0.5*np.pi, 0.5*np.pi)
        pitch_range = (-0.5*np.pi, 0.5*np.pi)
        roll_range = (-0.5*np.pi, 0.5*np.pi)

        lows = np.array([x_range[0], y_range[0], z_range[0], roll_range[0], pitch_range[0], yaw_range[0]])
        highs = np.array([x_range[1], y_range[1], z_range[1], roll_range[1], pitch_range[1], yaw_range[1]])

        particles = np.random.uniform(lows, highs, size = (self.num_particles, 0))
        particles = np.hstack((particles, np.zeros((self.num_particles, 9))))

        particles = particles.reshape((self.num_particles, 15, 1))
        return particles
    
    def run_filter(self, est_pos: List[Data]):
        for idx, pos in enumerate(est_pos):
            if not pos.tags:
                raise ValueError("No tags found in the data")
        
        estimates = np.zeros((len(est_pos) -1, 15, 1))
        particle_history = np.zeros((len(est_pos), self.num_particles, 15, 1))
                         
        particles = self.initial_particles()
        particle_history[0] = particles
        time = est_pos[0].timestamp

        W = np.ones((self.num_particles, 1)) / self.num_particles

        for idx, pos in enumerate(est_pos[1:]):
            print(f"\r{idx+1}/{len(est_pos)}", end="")
            dt = pos.timestamp - time
            time = pos.timestamp
            particles = self.predict(particles, dt, pos.acc, pos.omg)
            particle_history[idx] = particles

            measurement = self.get_measurment(pos)

            pred = self.update(measurement)

            pred = np.concatenate((pred, np.zeros((9,1))))

            W = self.update_weights(particles, pred)

            estimate = self.weighted_average(particles, W)

            particles, W = self.resampling(particles, W)
            
            estimates[idx -1] = estimate

        return estimates, particle_history