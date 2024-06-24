import os
import sys
import random
from generic_tls import *
import traffic_network

root = os.environ.get('PROJECT_ROOT')

# initialize traffic environment
env = traffic_network.TrafficNetwork()

# parse arguments
args = env.parse_args()

# set constant seed
random.seed(args.seed)

state = env.reset(is_gui=args.gui, collect_data=True)

total_reward = 0

while True:
    observation, reward, terminated = env.step()
    total_reward += reward

    if terminated:
        break

print(f"\ntotal_reward = {total_reward}\n")
env.plot_space_time(0)
