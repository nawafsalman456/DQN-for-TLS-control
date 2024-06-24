import os
import math
import random
import argparse
import matplotlib
import signal
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import tensorflow as tf
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import traffic_network

# Define a context manager to temporarily block interrupts
class BlockInterrupts:
    def __enter__(self):
        self.old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        return self

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)


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
        self.layer1 = nn.Linear(n_observations, 128) # TODO: init params ?
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[stay0exp,next0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


PROJECT_ROOT = os.environ.get('PROJECT_ROOT')

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor
# TAU is the update rate of the target network
# LR is the learning rate of the ``RMSprop`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 50
TAU = 0.001
LR = 0.001

SAVED_MODEL_PATH = f"{PROJECT_ROOT}/saved_models/RL_model_{LR}_{TAU}"


# initialize traffic environment
env = traffic_network.TrafficNetwork()

# parse arguments
args = env.parse_args()

# set constant seed
random.seed(args.seed)

if not args.test or args.debug:
    # initialize Tensorflow writers
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f'{PROJECT_ROOT}/logs/train/' + current_time + f"_LR_{LR}_TAU_{TAU}"
    # test_log_dir = f'{PROJECT_ROOT}/logs/test/' + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # test_summary_writer = tf.summary.create_file_writer(test_log_dir)


# Get the number of state observations and actions
state = env.reset()
n_observations = len(state)
n_actions = env.get_num_actions()

# initialize policy and target networks
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

# initialize RMSprop optimizer
optimizer = optim.RMSprop(policy_net.parameters(), lr=LR)

# initialize memory of size 500000000. make sure that it is big enough to save all transitions.
memory = ReplayMemory(500000000)

num_episodes = 200

max_total_reward = -float("inf")

rewards = []

# num of total steps on all episodes (different from env.curr_step which counts steps in current episode only)
num_total_steps = 0

def select_action(state):
    global num_total_steps
    num_total_steps += 1
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * num_total_steps / EPS_DECAY)
    eps_threshold = max(eps_threshold, 0.05)
    if args.test or sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([env.sample_random_action()], device=device, dtype=torch.long)


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

    # with train_summary_writer.as_default():
    #     with torch.no_grad():
    #         for name, param in policy_net.named_parameters():
    #             if param.grad is not None:
    #                 tf.summary.scalar(f'gradient norm/{name}', param.grad.norm(), num_total_steps)

    # gradients norm clipping
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 100)

    # with train_summary_writer.as_default():
    #     with torch.no_grad():
    #         for name, param in policy_net.named_parameters():
    #             if param.grad is not None:
    #                 tf.summary.scalar(f'clipped gradient norm/{name}', param.grad.norm(), num_total_steps)


    optimizer.step()


def print_reward(total_reward, max_total_reward, i_episode):
    print(f"\ntotal_reward = {total_reward}\n")
    print(f"max_total_reward = {max_total_reward}\n")
    print(f"i_episode = {i_episode}\n")

def plot_rewards():
    plt.plot(rewards)
    plt.title('Rewards')
    plt.xlabel('episode')
    plt.ylabel('Total Reward')
    # plt.show()
    plt.savefig(f'imgs/rewards.png')

def load_model():
    global rewards, memory
    if not os.path.exists(SAVED_MODEL_PATH):
        return
    checkpoint = torch.load(SAVED_MODEL_PATH)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    target_net.load_state_dict(checkpoint['target_net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    memory = checkpoint['memory']
    rewards = checkpoint['rewards'].tolist()
    # print("rewards = ", rewards)
    # print("len(memory) = ", len(memory))
    if args.test:
        policy_net.eval()
        target_net.eval()
    else:
        policy_net.train()
        target_net.train()

def save_model():
    with BlockInterrupts():
        torch.save({
                    'policy_net_state_dict': policy_net.state_dict(),
                    'target_net_state_dict': target_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'memory': memory,
                    'rewards' : torch.tensor(rewards, dtype=torch.int),
                    }, SAVED_MODEL_PATH)

def finalize_episode(total_reward, i_episode):
    global max_total_reward
    if not args.test:
        rewards.append(total_reward)
        max_total_reward = max(max_total_reward, total_reward)
    if args.print_reward:
        print_reward(total_reward, max_total_reward, i_episode)
    if args.plot_rewards:
        plot_rewards()
    if args.plot_space_time:
        env.plot_space_time()
    # if not args.test:
    #     with train_summary_writer.as_default():
    #         with torch.no_grad():
    #             tf.summary.scalar('Total Reward', total_reward, i_episode)
    #             for name, param in policy_net.named_parameters():
    #                 if param.grad is not None:
    #                     tf.summary.histogram(name + "/gradient", param.grad.cpu(), i_episode)
    #                     tf.summary.histogram(name, param, i_episode)
    #             # for name, param in target_net.named_parameters():
    #             #     if param.grad is not None:
    #             #         tf.summary.histogram(name + "/gradient_target_net", param.grad.cpu(), i_episode)
    #             #         tf.summary.histogram(name + "/target_net", param, i_episode)
    if args.save_model:
        print(f"Saving the model ...")
        save_model()
        print(f"*** Saved checkpoint at episode {i_episode+1}")

def train_model(num_episodes):
    for i_episode in range(num_episodes):
        # collect per-step data only in last episode.
        is_last_episode = (i_episode == num_episodes-1)
        # Initialize the environment and get its state
        state = env.reset(is_gui=args.gui, collect_data=is_last_episode)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0

        while True:
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
                finalize_episode(total_reward, i_episode)
                break

def test_model():

    # Initialize the environment and get its state
    state = env.reset(is_gui=args.gui, collect_data=True)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0

    while True:
        action = select_action(state)
        observation, reward, terminated = env.step(action.item())
        total_reward += reward
        reward = torch.tensor([reward], device=device)

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Move to the next state
        state = next_state

        if terminated:
            finalize_episode(total_reward, 0)
            break


if (args.test) and (not args.load_model):
    raise Exception("can't test without loading a given model. run script with --test --load_model")
if (args.test) and (args.save_model):
    raise Exception("can't save a model in test mode. run without --save_model")

if args.load_model:
    print("Loading the model ...")
    load_model()
    print("rewards = ", rewards)
    print("len(memory) = ", len(memory))

if not args.test:
    # main training loop
    print("Training ...")
    train_model(num_episodes)
else:
    test_model()

if args.save_model:
    print(f"Saving the Final model ...")
    save_model()

print('Complete')
plot_rewards()
env.plot_space_time(0)