import os
import sys
import argparse
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), f"{os.environ.get('SUMO_HOME')}\\tools"))

import traci
from sumolib import checkBinary
import generic_tls
import random
import io
import torch

root = os.environ.get('PROJECT_ROOT')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrafficNetwork:

    def __init__(self):
        # TODO - move TLS names to test file
        self.SIM_STEPS = 3600
        self.MIN_TIME_IN_PHASE = 5
        self.MAX_TIME_IN_PHASE = 30
        self.tls = generic_tls.GenericTLS('my_traffic_light')   # TODO - for now assume 1 tls, later create a list of all TLSs in the network ??
        self.tls_curr_phase = 0  # start from phase 0
        self.collect_data = False

        self.curr_step = 0
        self.reward = 0
        self.vehicles_alive_time = {}   # map from vehicle_id to the time it is alive in simulation
        self.vehicle_enter_time = {}
        self.state = []
        self.vehicle_positions = []
        self.vehicle_velocities = []
        self.simulation_times = []
        self.action_space = [0, 1]  # action space for each TLS : 0 - stay in current state. 1 - move to next state.

    def reset(self, is_gui=False, collect_data=False):
        # TODO - move below 2 to test file
        sim_file = f"{root}\\verif\sim\\single_tls_4_way\\single_tls_4_way.sumocfg"
        try:
            self.tls.start_simulation(sim_file, is_gui)
        except traci.exceptions.TraCIException:
            self.terminate()
            self.tls.start_simulation(sim_file, is_gui)

        self.vehicle_positions = []
        self.vehicle_velocities = []
        self.simulation_times = []
        self.curr_step = 0
        self.reward = 0
        self.vehicles_alive_time = {}
        self.vehicle_enter_time = {}
        self.state = self.get_curr_state()
        self.collect_data = collect_data
        return self.state

    def step(self, action = None):
        current_colors = self.tls.get_curr_colors()
        time_in_curr_phase = self.tls.get_curr_phase_spent_time()

        if ("y" in current_colors): # if yellow phase, stay for 2 seconds
            MIN = MAX = 2
        elif ("G" in current_colors or "g" in current_colors):
            MIN = self.MIN_TIME_IN_PHASE
            MAX = self.MAX_TIME_IN_PHASE
        else:
            MIN = MAX = 1   # if red phase, stay for 1 seconds

        if time_in_curr_phase >= MAX:
            # if stuck in current phase for a long time, move to next phase
            action = 1
        if time_in_curr_phase <= MIN:
            # stay in current phase for at least MIN seconds. to prevent fast transitions.
            action = 0

        if action is not None:
            num_tls_phases = len(self.tls.get_tls_all_phases())
            tls_next_phase = (self.tls_curr_phase + action) % num_tls_phases  # if action=0 - stay in current state. if action=1 - move to next phase
            self.tls.set_tls_phase(tls_next_phase)

        # do the action
        self.tls.do_one_simulation_step()
        # self.update_vehicles_alive_time()
        self.tls_curr_phase = self.tls.get_curr_phase()

        if self.collect_data:
            self.tls.aggregate_data()

        self.curr_step += 1
        self.state = self.get_curr_state()
        self.reward = -traci.vehicle.getIDCount()
        is_terminated =  (self.curr_step >= self.SIM_STEPS)
        if is_terminated:
            self.terminate()
        return  (self.state, self.reward, is_terminated)

    def get_curr_state(self):
        state_list = self.tls.get_curr_phase_encoding() + \
                     [self.tls.get_curr_phase_spent_time()] + \
                     self.tls.get_num_vehicles_on_each_lane() + \
                     self.tls.get_all_lanes_waiting_vehicles() + \
                     self.tls.get_min_vehicle_distance_on_each_lane() + \
                     [self.curr_step] 
        return state_list

    def get_num_actions(self):
        return 2

    def sample_random_action(self):
        return random.sample(self.action_space, k=1)

    def terminate(self):
        self.tls.end_simulation()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--gui', action='store_true')
        parser.add_argument('--save_model', action='store_true')
        parser.add_argument('--load_model', action='store_true')
        parser.add_argument('--print_reward', action='store_true')
        parser.add_argument('--plot_rewards', action='store_true')
        parser.add_argument('--plot_space_time', action='store_true')
        parser.add_argument("--seed", type=int, default=42, help="Seed for random number generation (default: 42)")
        return parser.parse_args()

    def plot_space_time(self, pause_time = 10):
        # Data collection of the last simulation
        vehicle_positions, vehicle_velocities, distance_from_tls = self.tls.get_aggregated_data()
        # Plot space-time diagram
        plt.figure(figsize=(10, 5))
        scatter = plt.scatter(distance_from_tls, vehicle_positions, c=vehicle_velocities, cmap='viridis', marker='.')
        plt.colorbar(scatter, label='Velocity (m/s)')
        plt.xlabel('Simulation Time (s)')
        plt.ylabel('Distance From TLS')
        plt.title('Space-Time Diagram')
        if pause_time == 0:
            plt.show()
        else:
            plt.pause(pause_time)    # Wait for "pause_time" second before closing the window
            plt.clf()  # Clear the current figure
            plt.close() # Close the current figure window
