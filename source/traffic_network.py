import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), f"{os.environ.get('SUMO_HOME')}\\tools"))

import traci
from sumolib import checkBinary
import generic_tls
import random
import io
import torch

root = os.environ.get('PROJECT_ROOT')

SIM_STEPS = 3600
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrafficNetwork:

    def __init__(self):
        # TODO - move TLS names to test file
        self.tls = generic_tls.GenericTLS('my_traffic_light')   # TODO - for now assume 1 tls, later create a list of all TLSs in the network ??
        self.tls_curr_phase = 0  # start from phase 0

        self.curr_step = 0
        self.reward = 0
        self.vehicles_alive_time = {}   # map from vehicle_id to the time it is alive in simulation
        self.vehicle_enter_time = {}
        self.state = []
        self.vehicle_positions = []
        self.vehicle_velocities = []
        self.simulation_times = []
        self.action_space = [0, 1]  # action space for each TLS : 0 - stay in current state. 1 - move to next state. TODO - when network contains a list of TLSs. action space will be a dict of all permutation of actions of all TLSs in network ?

    def reset(self, is_gui):
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
        return self.state

    # TODO - when network contains a list of TLSs. "action" will be dict. in each entry - {"tls_id" : action}
    def step(self, action):
        current_colors = self.tls.get_curr_colors()
        MIN = 10 if ("G" in current_colors or "g" in current_colors) else 2     # if current color are only red and yellow. Min = 2
        MAX = 30
        time_in_curr_phase = self.tls.get_curr_phase_spent_time()

        if time_in_curr_phase > MAX:
            # if stuck in current phase for a long time, move to next phase
            action = 1
        if time_in_curr_phase < MIN:
            # stay in current phase for at least MIN seconds. to prevent fast transitions.
            action = 0

        num_tls_phases = len(self.tls.get_tls_all_phases())
        self.tls_curr_phase = (self.tls_curr_phase + action) % num_tls_phases   # if action=0 - stay in current state. if action=1 - move to next phase
        self.tls.set_tls_phase(self.tls_curr_phase)

        # do the action
        self.tls.do_one_simulation_step()
        # self.update_vehicles_alive_time()

        self.curr_step += 1
        self.state = self.get_curr_state()
        self.reward = -len(traci.vehicle.getIDList()) #-sum(self.vehicles_alive_time.values())
        is_terminated =  (self.curr_step >= SIM_STEPS)
        if is_terminated:
            self.terminate()
        return  (self.state, self.reward, is_terminated)

    def get_aggregated_data(self):
        return self.tls.get_aggregated_data()

    def get_curr_state(self):
        state_list = self.get_curr_phase_encoding() + \
                     [self.tls.get_curr_phase_spent_time()] + \
                     self.tls.get_all_phases_waiting_vehicles()# + \
                    #  [self.curr_step] + \
                    #  self.tls.get_num_vehicles_on_each_lane() + \
                    #  self.tls.get_min_vehicle_distance_on_each_lane()
        return state_list

    def get_curr_phase_encoding(self):
        num_phases = len(self.tls.get_tls_all_phases())
        curr_phase_index = self.tls.get_curr_phase()
        phases_encoding_list = [0]*num_phases
        phases_encoding_list[curr_phase_index] = 1
        return phases_encoding_list

    def get_num_actions(self):
        return 2

    def sample_random_action(self):
        return random.sample(self.action_space, k=1)

    def terminate(self):
        self.tls.end_simulation()

    # def update_vehicles_alive_time(self):
    #     # remove vehicles that exited from simulation
    #     for arrived_vehicle_id in traci.simulation.getArrivedIDList():
    #         self.vehicle_enter_time.pop(arrived_vehicle_id)
    #         self.vehicles_alive_time.pop(arrived_vehicle_id)

    #     for vehicle_id in traci.vehicle.getIDList():
    #         if not (vehicle_id in self.vehicles_alive_time):
    #             self.vehicle_enter_time[vehicle_id] = self.curr_step
    #             self.vehicles_alive_time[vehicle_id] = 1
    #         self.vehicles_alive_time[vehicle_id] += (self.curr_step - self.vehicle_enter_time[vehicle_id])  # higher punishment to "older" vehicles


