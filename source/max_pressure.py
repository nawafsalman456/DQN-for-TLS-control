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

# sim_file = f"{root}/verif/sim/try/try.sumocfg"
sim_file = f"{root}/verif/sim/single_tls_4_way/single_tls_4_way.sumocfg"
state = env.reset(is_gui=args.gui, collect_data=False, sim_file=sim_file)


env.MIN_TIME_IN_PHASE = 5
env.MAX_TIME_IN_PHASE = 50

green_phases = env.tls.get_tls_green_phases()

rewards = []

num_episodes = 1
for t in range(num_episodes):
    total_reward = 0
    curr_phase = 0
    time_in_yellow = 0
    time_in_red = 0
    env.reset(sim_file=sim_file)
    while True:
        
        next_phase = env.tls.max_pressure()

        colors = env.tls.get_curr_colors()
        is_red_phase = not("G" in colors) and not("g" in colors) and not("y" in colors)
        is_yellow_phase = ("y" in colors)
        
        if next_phase != curr_phase and env.tls.get_curr_phase_spent_time() < env.MIN_TIME_IN_PHASE and not(is_red_phase) and not(is_yellow_phase):
            next_phase = curr_phase
            
        # Force change phase if max time in the current phase is exceeded
        if next_phase == curr_phase and env.tls.get_curr_phase_spent_time() > env.MAX_TIME_IN_PHASE:
            next_phase = env.tls.max_pressure()

        # Implement yellow phase logic
        if not(is_red_phase) and ((next_phase != curr_phase and time_in_yellow == 0) or (time_in_yellow == 1)):
            time_in_yellow += 1
            next_phase = 1#env.tls.get_next_yellow_phase()
        elif time_in_yellow >= 2:
            time_in_yellow = 0  # Reset yellow phase timer
            
            
        # Implement red phase logic
        if (is_yellow_phase and time_in_yellow == 0 and time_in_red < 1):
            time_in_red += 1
            next_phase = 2
        elif time_in_red >= 1:
            time_in_red = 0  # Reset red phase timer
            
        # print("next_phase = ", next_phase)
        env.tls.set_tls_phase_max_pressure(curr_phase, next_phase)

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
            print(f"\ntotal_reward = {total_reward}\n")
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
