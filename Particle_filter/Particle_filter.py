import numpy as np
import matplotlib.pyplot as plt
from observation_model import ObservationModel, pose_to_ypr
from utils import Data, Marker, ActualData, Coordinate, PixelCoordinate
import time
from scipy.stats import multivariate_normal

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
        pass

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
    
    def plot(self):
        pass
    