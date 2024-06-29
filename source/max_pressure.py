import random
from itertools import cycle
from collections import deque

from generic_tls import *
import traffic_network
import matplotlib.pyplot as plt


root = os.environ.get('PROJECT_ROOT')

# initialize traffic environment
env = traffic_network.TrafficNetwork()

# parse arguments
args = env.parse_args()

# set constant seed
random.seed(args.seed)

state = env.reset(is_gui=args.gui, collect_data=False)


env.MIN_TIME_IN_PHASE = 5
env.MAX_TIME_IN_PHASE = 35

green_phases = env.tls.get_tls_green_phases()

rewards = []

num_episodes = 300
for t in range(num_episodes):
    total_reward = 0
    curr_phase = 0
    time_in_yellow = 0
    time_in_red = 0
    env.reset()
    while True:
        
        next_phase = env.tls.max_pressure(curr_phase)
        
        if next_phase != curr_phase and env.tls.get_curr_phase_spent_time() < env.MIN_TIME_IN_PHASE:
            next_phase = curr_phase
            
        # Force change phase if max time in the current phase is exceeded
        if next_phase == curr_phase and env.tls.get_curr_phase_spent_time() > env.MAX_TIME_IN_PHASE:
            next_phase = env.tls.max_pressure(curr_phase)

        # Implement yellow phase logic
        if next_phase != curr_phase and time_in_yellow < 2:
            time_in_yellow += 1
            next_phase = 4
        elif time_in_yellow >= 2:
            time_in_yellow = 0  # Reset yellow phase timer
            
            
        # Implement red phase logic
        if next_phase != curr_phase and time_in_red < 1:
            time_in_red += 1
            next_phase = 5
        elif time_in_red >= 1:
            time_in_red = 0  # Reset red phase timer


        env.tls.set_tls_phase(next_phase)

        curr_phase = next_phase
        
        # do the action
        env.tls.do_one_simulation_step()

        if env.collect_data:
            env.tls.aggregate_data()

        env.curr_step += 1
        weighted_reward = -env.tls.get_weighted_num_vehicles()
        total_reward += weighted_reward
        is_terminated =  (env.curr_step >= env.SIM_STEPS)
        if is_terminated:
            rewards.append(total_reward)
            env.terminate()
            plt.plot(rewards)
            plt.title('Rewards')
            plt.xlabel('episode')
            plt.ylabel('Total Reward')
            # plt.show()
            plt.savefig(f'imgs/rewards_max_pressure.png')
            break
    

print(rewards)
plt.plot(rewards)
plt.title('Rewards')
plt.xlabel('episode')
plt.ylabel('Total Reward')
# plt.show()
plt.savefig(f'imgs/rewards_max_pressure.png')

print(f"\ntotal_reward = {total_reward}\n")
env.plot_space_time(0)
