import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), f"{os.environ.get('SUMO_HOME')}\\tools"))

import traci
from sumolib import checkBinary

class GenericTLS:

    def __init__(self, tls_id):
        self.tls_id = tls_id
        self.phases_spent_time = [] # time spent in each phase. only counts time spent in current phases loop.
        self.phases_waiting_vehicles = []
        self.changed_phase = False

    def start_simulation(self, scenario_file, is_gui = False):
        gui = "-gui" if is_gui else ""
        sumo_binary = checkBinary('sumo'+ gui)
        traci.start([sumo_binary, '-c', scenario_file])
        self.phases_spent_time = [0]*len(self.get_tls_all_phases())
        self.phases_waiting_vehicles = [0]*len(self.get_tls_all_phases())
        self.changed_phase = False
        self.vehicle_distance_from_tls = []
        self.vehicle_velocities = []
        self.simulation_times = []

    def end_simulation(self):
        traci.close()

    def do_one_simulation_step(self):
        traci.simulationStep()
        self.update_curr_phase_spent_time()
        self.update_curr_phase_waiting_vehicles()
        self.aggregate_data()

    # return list of vehicles waiting for this TLS, each entry:
    # (vehicle_id, distance_to_tls, tls_color)
    # tls_color : r, y, g, G
    def get_waiting_vehicles_on_tls(self):
        waiting_vehicles = []
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        for lane_id in controlled_lanes:
            vehicles_on_lane = traci.lane.getLastStepVehicleIDs(lane_id)
            for vehicle_id in vehicles_on_lane:
                if traci.vehicle.getSpeed(vehicle_id) == 0:
                    next_tls_tuple = traci.vehicle.getNextTLS(vehicle_id)
                    next_tls_id, tls_index, tls_distance, tls_color = next_tls_tuple[0]
                    assert(self.tls_id == next_tls_id)
                    waiting_vehicles.append((vehicle_id, tls_distance, tls_color))
        return waiting_vehicles

    def get_time_loss(self):
        total_time_loss = 0
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        for lane_id in controlled_lanes:
            vehicles_on_lane = traci.lane.getLastStepVehicleIDs(lane_id)
            for vehicle_id in vehicles_on_lane:
                total_time_loss += traci.vehicle.getTimeLoss(vehicle_id)
        return total_time_loss

    def get_num_controlled_lanes(self):
        return len(traci.trafficlight.getControlledLanes(self.tls_id))

    def get_num_vehicles_on_each_lane(self):
        num_vehicles = []
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        for lane_id in controlled_lanes:
            num_vehicles_on_lane = len(traci.lane.getLastStepVehicleIDs(lane_id))
            num_vehicles.append(num_vehicles_on_lane)
        return num_vehicles

    def get_waiting_time_on_each_lane(self):
        waiting_time = []
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        for lane_id in controlled_lanes:
            waiting_time.append(traci.lane.getWaitingTime(lane_id))
        return waiting_time

    def get_min_vehicle_distance_on_each_lane(self):
        min_distances = []
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        for lane_id in controlled_lanes:
            min_distance_to_tls = 200
            vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            for vehicle_id in vehicles:
                next_tls_tuple = traci.vehicle.getNextTLS(vehicle_id)
                next_tls_id, tls_index, tls_distance, tls_color = next_tls_tuple[0]
                assert(self.tls_id == next_tls_id)
                min_distance_to_tls = min(min_distance_to_tls, tls_distance)
            min_distances.append(min_distance_to_tls)
        return min_distances

    def get_tls_all_phases(self):
        all_program_logics = traci.trafficlight.getAllProgramLogics(self.tls_id)
        assert(all_program_logics != () and all_program_logics != None)
        return all_program_logics[0].getPhases()

    def get_curr_phase(self):
        return traci.trafficlight.getPhase(self.tls_id)

    def get_curr_phase_spent_time(self):
        curr_phase_index = self.get_curr_phase()
        return self.phases_spent_time[curr_phase_index]

    def get_all_phases_waiting_vehicles(self):
        return self.phases_waiting_vehicles


    def update_curr_phase_spent_time(self):
        curr_phase_index = self.get_curr_phase()
        if self.changed_phase:
            self.phases_spent_time[curr_phase_index] = 0
            self.changed_phase = False
        self.phases_spent_time[curr_phase_index] += 1

    def update_curr_phase_waiting_vehicles(self):
        curr_phase_index = self.get_curr_phase()
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        total_waiting_vehicles = 0
        for lane_id in controlled_lanes:
            num_waiting_vehicles = traci.lane.getLastStepHaltingNumber(lane_id)
            total_waiting_vehicles += num_waiting_vehicles
        self.phases_waiting_vehicles[curr_phase_index] = total_waiting_vehicles

    def aggregate_data(self):
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        for lane_id in controlled_lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            for vehicle_id in vehicles:
                velocity = traci.vehicle.getSpeed(vehicle_id)
                next_tls_tuple = traci.vehicle.getNextTLS(vehicle_id)
                next_tls_id, tls_index, tls_distance, tls_color = next_tls_tuple[0]
                assert(self.tls_id == next_tls_id)
                self.vehicle_distance_from_tls.append(tls_distance)
                self.vehicle_velocities.append(velocity)
                self.simulation_times.append(traci.simulation.getTime())


    def set_tls_phase(self, phase_index):
        prev_phase = self.get_curr_phase()
        traci.trafficlight.setPhase(self.tls_id, phase_index)
        curr_phase = self.get_curr_phase()
        if curr_phase != prev_phase:
            self.changed_phase = True

    def get_aggregated_data(self):
        return (self.vehicle_distance_from_tls, self.vehicle_velocities, self.simulation_times)

    def get_curr_colors(self):
        return traci.trafficlight.getRedYellowGreenState(self.tls_id)