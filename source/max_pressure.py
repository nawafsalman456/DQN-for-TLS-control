import random
from itertools import cycle
from collections import deque

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

env.MIN_TIME_IN_PHASE = 1
env.MAX_TIME_IN_PHASE = 99

while True:
    
    next_phase = env.tls.max_pressure()
    
    action = 0
    if env.tls_curr_phase != next_phase:
        action = 1

    observation, reward, terminated = env.step(action)
    total_reward += reward

    if terminated:
        break

print(f"\ntotal_reward = {total_reward}\n")
env.plot_space_time(0)
