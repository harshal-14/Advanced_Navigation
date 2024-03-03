#Load the data from the txt files
import glob
import numpy as np
import matplotlib.pyplot as plt

class Data_loader:
    def __init__(self, path):
        self.path = path
        self.data = []
    def load_data(self):
        for file in glob.glob(self.path + "*.txt"):
            self.data.append(np.loadtxt(file, delimiter=","))
        return self.data
    
class Kalman:
    def __init__(self):


    def get_position(self, u, t):
        self.u = u
        self.z = z
        self.t = t

        self.x = []
        self.y = []
        self.z = []
        initial_velocity = 0
        for i in range(len(t)):
            x = initial_velocity*t + 0.5*u1*t**2
            y = initial_velocity*t + 0.5*u2*t**2
            z = initial_velocity*t + 0.5*u3*t**2
            self.x.append(x)
            self.y.append(y)
            self.z.append(z)
        return self.x, self.y, self.z
       
if __name__ == "__main__":
    path = "../Kalman_filter/Data/"
    data_mocap = Data_loader(path).load_data()
    t, u1, u2, u3, z1, z2, z3 = data_mocap[0].T
    kalman = Kalman()