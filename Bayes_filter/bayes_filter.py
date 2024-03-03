import matplotlib.pyplot as plt
from tqdm import tqdm

class RobotBeliefModel:
    
    def __init__(self, z_prob, x_prob):
        '''
        Initializes the RobotBeliefModel with the given probabilities
        Inputs:
        z_prob: dictionary of dictionaries, where the first key is the measurement type and the second key is the state of the door. 
        The value is the probability of the measurement given the state of the door.
        x_prob: dictionary of dictionaries, where the first key is the state of the door, the second key is the action, and the third 
        key is the resulting state of the door. The value is the probability of the resulting state of the door given the initial state and the action.
        '''
        self.z_prob = z_prob
        self.x_prob = x_prob

    def predict(self, tasks, initial_belief=None):
        '''
        Predicts the probability of the door being open or closed after performing the given tasks, given the initial belief.
        Inputs:
        tasks: list of strings, where each string is the action to be performed in the corresponding step.
        initial_belief: list of two floats, where the first float is the probability of the door being open and the second float is the 
        probability of the door being closed.
        '''
        door_predicted_open = (self.x_prob['open'][tasks[0]][tasks[1]] * initial_belief[0]
                     + self.x_prob['open'][tasks[0]][tasks[2]] * initial_belief[1])
        door_predicted_closed = (self.x_prob['closed'][tasks[0]][tasks[1]] * initial_belief[0]
                       + self.x_prob['closed'][tasks[0]][tasks[2]] * initial_belief[1])

        return [door_predicted_open, door_predicted_closed]

    def update(self, pred, sense=None):
        '''
        Updates the probability of the door being open or closed after performing the given measurement, given the predicted probabilities.
        Inputs:
        pred: list of two floats, where the first float is the probability of the door being open and the second float is the probability
        of the door being closed after performing the tasks.
        sense: string, the measurement type.
        '''
        x_open_meas = self.z_prob[sense]['open'] * pred[0]
        x_closed_meas = self.z_prob[sense]['closed'] * pred[1]

        normalize = 1 / (x_open_meas + x_closed_meas)

        return [x_open_meas * normalize, x_closed_meas * normalize]

    def run_tasks(self, action_tasks, measurement_type, initial_belief=[0.5, 0.5]):
        '''
        Runs the given tasks and measurement type and returns the belief of the door being open after each iteration.
        Inputs:
        action_tasks: list of strings, where each string is the action to be performed in the corresponding step.
        measurement_type: string, the measurement type.
        initial_belief: list of two floats, where the first float is the probability of the door being open and the second float is the
        probability of the door being closed.
        '''
        total_belief = []

        while initial_belief[0] < 0.9999:
            pred = self.predict(action_tasks, initial_belief)
            initial_belief = self.update(pred, measurement_type)
            total_belief.append(initial_belief[0])
        print(f"It took {len(total_belief)} iterations to reach belief of {initial_belief[0]}")
        print(f"-------------------------------------------------------------------------------")
        return total_belief
    
    def run_steady_state_tasks(self, action_tasks, measurement_type, initial_belief=[0.5, 0.5], tau=0.0001):
        '''
        Runs the given tasks and measurement type and returns the belief of the door being open after each iteration.
        Inputs:
        action_tasks: list of strings, where each string is the action to be performed in the corresponding step.
        measurement_type: string, the measurement type.
        initial_belief: list of two floats, where the first float is the probability of the door being open and the second float is the
        probability of the door being closed.
        '''
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

        print(f"It took {len(total_belief)} iterations to reach a steady state belief of {total_belief[-1]}")
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
