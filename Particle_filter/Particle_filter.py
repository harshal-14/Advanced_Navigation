import numpy as np
import matplotlib.pyplot as plt
from Non_linear_Kalman_filter.Hw3_Harshal_Bhat import observation_model

class ParticleFilter:
    def __init__(self, num_particles, x0, P0, Q, R):
        pass

    def predict(self):
        #For each particle, take the original particle, sample from the noise distribution and then apply the motion model
        #Use Measured inputs plus noise to determine the future state of the particle using process model
        #Use noise Covariance values as starting point for the process noise
        pass

    def update(self):
        #Use the observation model and the measurement to calculate its importance weight for each particle and then use the low varinance resampling algorithm to find updated particle set
        # Use the covariance values as a starting point for the observation model.
        pass

    def resample(self):
        pass

    def plot(self):
        pass
    