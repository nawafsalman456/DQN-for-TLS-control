import math
import random
import argparse
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import traffic_network

parser = argparse.ArgumentParser()
parser.add_argument('--gui', action='store_true')
parser.add_argument('--print_reward', action='store_true')
parser.add_argument('--plot_loss', action='store_true')
parser.add_argument('--plot_space_time', action='store_true')
args = parser.parse_args()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[stay0exp,next0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor
# TAU is the update rate of the target network
# LR is the learning rate of the ``RMSprop`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.001
LR = 0.01


env = traffic_network.TrafficNetwork()

# Get the number of state observations and actions
state = env.reset(is_gui=False)
n_observations = len(state)
n_actions = env.get_num_actions()

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.RMSprop(policy_net.parameters(), lr=LR)

memory = ReplayMemory(500000)


num_steps = 0

def select_action(state):
    global num_steps
    num_steps += 1
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * num_steps / EPS_DECAY)
    eps_threshold = max(eps_threshold, 0.05)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([env.sample_random_action()], device=device, dtype=torch.long)


losses = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch. This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat(list(filter(lambda s: s is not None, batch.next_state)))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
    optimizer.step()
    
    # save it for the plotting
    losses.append(loss.item())

def print_reward(total_reward, max_total_reward, i_episode):
    print(f"\ntotal_reward = {total_reward}\n")
    print(f"max_total_reward = {max_total_reward}\n")
    print(f"i_episode = {i_episode}\n")

def plot_loss(losses, pause_time = 2):
    assert(False)   # not working. instead plot reward per episode
    plt.plot(losses)
    plt.title('Training ...')
    plt.xlabel('step')
    plt.ylabel('Loss')
    plt.pause(pause_time)    # Wait for "pause_time" second before closing the window
    plt.clf()  # Clear the current figure
    plt.close() # Close the current figure window

def plot_space_time():
    # Data collection of the last simulation
    vehicle_positions, vehicle_velocities, distance_from_tls = env.get_aggregated_data()
    # Plot space-time diagram
    plt.figure(figsize=(10, 5))
    scatter = plt.scatter(distance_from_tls, vehicle_positions, c=vehicle_velocities, cmap='viridis', marker='.')
    plt.colorbar(scatter, label='Velocity (m/s)')
    plt.xlabel('Simulation Time (s)')
    plt.ylabel('Distance From TLS')
    plt.title('Space-Time Diagram')
    plt.pause(10)    # Wait for "pause_time" second before closing the window
    plt.clf()  # Clear the current figure
    plt.close() # Close the current figure window

if torch.cuda.is_available():
    num_episodes = 600
else:
    num_episodes = 100

max_total_reward = -float("inf")

#  main training loop
 # TODO - run episodes in parallel ?
for i_episode in range(num_episodes):
    # Initialize the environment and get its state

    # # change this to True if need to open GUI
    # # TODO - need to control this bit in run time, between 2 consecutive episodes
    # is_gui = False

    state = env.reset(is_gui=args.gui)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0

    # # clear the losses of previous episode
    # losses = []

    for t in count():
        action = select_action(state)
        observation, reward, terminated = env.step(action.item())
        total_reward += reward
        reward = torch.tensor([reward], device=device)

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if terminated:
            max_total_reward = max(max_total_reward, total_reward)
            if args.print_reward:
                print_reward(total_reward, max_total_reward, i_episode)
            if args.plot_loss:
                plot_loss(losses)
            if args.plot_space_time:
                plot_space_time()
            break
           

print('Complete')
plt.plot(losses)
plt.title('Final Results')
plt.xlabel('step')
plt.ylabel('Loss')
plt.show()
