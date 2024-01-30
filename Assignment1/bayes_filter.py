import matplotlib.pyplot as plt
from tqdm import tqdm

class RobotBeliefModel:
    def __init__(self, z_prob, x_prob):
        self.z_prob = z_prob
        self.x_prob = x_prob

    def predict(self, tasks, initial_belief=None):
        door_predicted_open = (self.x_prob['open'][tasks[0]][tasks[1]] * initial_belief[0]
                     + self.x_prob['open'][tasks[0]][tasks[2]] * initial_belief[1])
        door_predicted_closed = (self.x_prob['closed'][tasks[0]][tasks[1]] * initial_belief[0]
                       + self.x_prob['closed'][tasks[0]][tasks[2]] * initial_belief[1])

        return [door_predicted_open, door_predicted_closed]

    def update(self, pred, sense=None):
        x_open_meas = self.z_prob[sense]['open'] * pred[0]
        x_closed_meas = self.z_prob[sense]['closed'] * pred[1]

        normalize = 1 / (x_open_meas + x_closed_meas)

        return [x_open_meas * normalize, x_closed_meas * normalize]

    def run_tasks(self, action_tasks, measurement_type, initial_belief=[0.5, 0.5]):
        total_belief = []

        while initial_belief[0] < 0.9999:
            pred = self.predict(action_tasks, initial_belief)
            initial_belief = self.update(pred, measurement_type)
            total_belief.append(initial_belief[0])
        print(f"It took {len(total_belief)} iterations to reach belief of {initial_belief[0]}")
        print(f"-------------------------------------------------------------------------------")
        return total_belief
    
    def run_steady_state_tasks(self, action_tasks, measurement_type, initial_belief=[0.5, 0.5], tau=0.0001):
        total_belief = []
        relative_change = tau + 1
        iterations = 0
        while relative_change > tau and iterations < 1000:
            pred = self.predict(action_tasks, initial_belief)
            new_belief = self.update(pred, measurement_type)
            relative_change = abs(new_belief[0] - initial_belief[0])
            initial_belief = new_belief
            total_belief.append(initial_belief[0])
            iterations += 1

        print(f"It took {len(total_belief)} iterations to reach a steady state belief of {initial_belief[0]}")
        print(f"-------------------------------------------------------------------------------")

        return total_belief


def plot_total_belief(x_values, total_belief, action_type, measurement_type):
    plt.plot(x_values, total_belief)
    plt.xlabel('Iteration')
    plt.ylabel('Belief door open')
    plt.title(f'U={action_type} and Z={measurement_type}')
    plt.show()


# Defining probabilities
z_prob = {"measure_open": {"open": 0.6, "closed": 0.2}, "measure_closed": {"open": 0.4, "closed": 0.8}}
x_prob = {"open": {"push": {"open": 1, "closed": 0.8}, "do_nothing": {"open": 1, "closed": 0}},
          "closed": {"push": {"open": 0, "closed": 0.2}, "do_nothing": {"open": 0, "closed": 1}}}

robot_model = RobotBeliefModel(z_prob, x_prob)

# tasks 1
initial_belief = [0.5, 0.5]
total_belief = robot_model.run_tasks(['do_nothing', 'open', 'closed'], 'measure_open', initial_belief)
x_values = list(range(1, len(total_belief) + 1))
plot_total_belief(x_values, total_belief, 'do_nothing', 'open')

# tasks 2
initial_belief = [0.5, 0.5]
total_belief = robot_model.run_tasks(['push', 'open', 'closed'], 'measure_open', initial_belief)
x_values = list(range(1, len(total_belief) + 1))
plot_total_belief(x_values, total_belief, 'push', 'open')

# tasks 3
initial_belief = [0.5, 0.5]
# tau = 0.001 #tolerance for reaching steady state
# relative_change = tau + 1
total_belief = robot_model.run_steady_state_tasks(['push', 'open', 'closed'], 'measure_closed', initial_belief)
x_values = list(range(1, len(total_belief) + 1))
plot_total_belief(x_values, total_belief, 'push', 'closed')
