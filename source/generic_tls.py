import os
import sys
import random
import torch
import numpy as np
from collections import defaultdict, Counter
sys.path.append(os.path.join(os.path.dirname(__file__), f"{os.environ.get('SUMO_HOME')}\\tools"))
import time
import traci
from sumolib import checkBinary

class GenericTLS:

    def __init__(self, tls_id):
        self.tls_id = tls_id
        self.phases_spent_time = [] # time spent in each phase. only counts time spent in current phases loop.
        self.phases_total_spent_time = []
        self.changed_phase = False
        self.next_phase = None
        self.num_cars = 0
        self.num_buses = 0
        self.vtypes_map = {}

    def start_simulation(self, scenario_file, is_gui = False):
        gui = "-gui" if is_gui else ""
        sumo_binary = checkBinary('sumo'+ gui)
        traci.start([sumo_binary, '-c', scenario_file])
        self.phases_spent_time = [0]*len(self.get_tls_all_phases())
        self.phases_total_spent_time = [0]*len(self.get_tls_all_phases())
        self.changed_phase = False
        self.next_phase = None
        self.num_cars = 0
        self.num_buses = 0
        self.vtypes_map = {}
        self.vehicle_distance_from_tls = []
        self.vehicle_velocities = []
        self.simulation_times = []
        self.green_phases = self.get_tls_green_phases()
        self.green_phases_mask = self.get_tls_green_phases_mask()
        self.max_pressure_lanes = self.get_max_pressure_lanes()

    def end_simulation(self):
        traci.close()

    def do_one_simulation_step(self):
        prev_phase = self.get_curr_phase()
        if self.next_phase is not None:
            traci.trafficlight.setPhase(self.tls_id, self.next_phase)
            self.next_phase = None

        traci.simulationStep()

        curr_phase = self.get_curr_phase()
        if curr_phase != prev_phase:
            self.changed_phase = True

        self.update_total_num_cars_and_buses()
        self.update_curr_phase_spent_time()

    def get_tls_green_phases_mask(self):
        green_phases_mask = [0] * len(self.get_tls_all_phases())
        for i, p in enumerate(self.get_tls_all_phases()):
            if ("g" in p.state) or ("G" in p.state):
                green_phases_mask[i] = 1
        return green_phases_mask

    def get_num_lanes(self):
        return len(traci.trafficlight.getControlledLanes(self.tls_id))

    # return (in_vehicles, out_vehicles)
    # in_vehicles: number of vehicles entering the TLS from each lane
    # out_vehicles: number of vehicles leaving the TLS from each lane
    def get_in_out_vehicles_on_each_lane(self):
        in_vehicles = {}
        out_vehicles = {}
        lanes = traci.trafficlight.getControlledLinks(self.tls_id)
        for lane in lanes:
            for link in lane:
                incoming_lane = link[0]
                outgoing_lane = link[1]
                in_vehicles[incoming_lane] = traci.lane.getLastStepVehicleNumber(incoming_lane)
                out_vehicles[outgoing_lane] = traci.lane.getLastStepVehicleNumber(outgoing_lane)
        return in_vehicles, out_vehicles

    def get_num_vehicles_on_each_lane(self):
        num_vehicles = []
        controlled_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        for lane_id in controlled_lanes:
            vehicles = traci.lane.getLastStepVehicleNumber(lane_id)
            num_vehicles.append(vehicles)
        return num_vehicles
    
    def get_num_cars_and_buses_on_each_lane(self):
        num_cars = []
        num_buses = []
        lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        for lane in lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            num_cars_curr_lane = 0
            num_buses_curr_lane = 0
            for vehicle_id in vehicles:
                vtype = traci.vehicle.getTypeID(vehicle_id)
                num_cars_curr_lane += (vtype == "CAR_TYPE")
                num_buses_curr_lane += (vtype == "BUS_TYPE")
            num_cars.append(num_cars_curr_lane)
            num_buses.append(num_buses_curr_lane)
        return num_cars, num_buses

    def get_tls_all_phases(self):
        all_program_logics = traci.trafficlight.getAllProgramLogics(self.tls_id)
        assert(all_program_logics != () and all_program_logics != None)
        return all_program_logics[0].getPhases()

    def get_tls_green_phases(self):
        green_phases = []
        for i, p in enumerate(self.get_tls_all_phases()):
            if ("g" in p.state) or ("G" in p.state):
                phase = (i, p.state)
                green_phases.append(phase)
        return green_phases

    def get_curr_phase(self):
        return traci.trafficlight.getPhase(self.tls_id)

    def get_curr_phase_spent_time(self):
        curr_phase_index = self.get_curr_phase()
        return self.phases_spent_time[curr_phase_index]
    
    def get_phases_total_spent_time(self):
        return self.phases_total_spent_time

    def update_curr_phase_spent_time(self):
        curr_phase_index = self.get_curr_phase()
        if self.changed_phase:
            self.phases_spent_time[curr_phase_index] = 0
            self.changed_phase = False
        self.phases_total_spent_time[curr_phase_index] += 1
        self.phases_spent_time[curr_phase_index] += 1

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

    def get_num_vehicle_of_each_type(self):
        vehicles = traci.vehicle.getIDList()
        vehicle_types = [traci.vehicle.getTypeID(vehicle_id) for vehicle_id in vehicles]

        # Convert list to numpy array for efficient operations
        vehicle_types_array = np.array(vehicle_types)

        # Get unique vehicle types and their counts
        unique_vehicle_types, counts = np.unique(vehicle_types_array, return_counts=True)

        # Convert to dictionary
        vehicle_types_num = dict(zip(unique_vehicle_types, counts))

        return vehicle_types_num

    # don't use, very slow in case of high pressure
    def get_total_num_cars_and_buses_old_dont_use(self):

        vehicle_types_num = self.get_num_vehicle_of_each_type()

        num_cars = vehicle_types_num.get("CAR_TYPE", 0)
        num_buses = vehicle_types_num.get("BUS_TYPE", 0)

        # print("num_cars = ", num_cars)
        # print("num_buses = ", num_buses)
        # assert(num_cars + num_buses == traci.vehicle.getIDCount())
        
        return num_cars, num_buses
    
    def get_total_num_cars_and_buses(self):
        return self.num_cars, self.num_buses
    
    def update_total_num_cars_and_buses(self):
        # Get the list of vehicles that have entered (departed) and left (arrived)
        departed_vehicles = traci.simulation.getDepartedIDList()
        arrived_vehicles = traci.simulation.getArrivedIDList()

        # Update the count based on departed vehicles
        for veh_id in departed_vehicles:
            vtype = traci.vehicle.getTypeID(veh_id)
            self.vtypes_map[veh_id] = vtype  # Store the type
            self.num_cars += (vtype == "CAR_TYPE")
            self.num_buses += (vtype == "BUS_TYPE")

        # Update the count based on arrived vehicles (subtract vehicles that have left)
        for veh_id in arrived_vehicles:
            vtype = self.vtypes_map.pop(veh_id, None)  # Retrieve and remove the type
            self.num_cars -= (vtype == "CAR_TYPE")
            self.num_buses -= (vtype == "BUS_TYPE")
            
    def set_tls_phase(self, phase_index):
        self.next_phase = phase_index

    def set_max_pressure_tls_phase(self, phase_index):
        if (self.get_curr_phase() != phase_index):
            self.green_phases_mask[self.get_curr_phase()] = 0
        # if selected all green phases in this round, re-set all green phases bits
        if (sum(self.green_phases_mask) == 1):
            self.green_phases_mask = self.get_tls_green_phases_mask()
        self.next_phase = phase_index
        

    def get_aggregated_data(self):
        return (self.vehicle_distance_from_tls, self.vehicle_velocities, self.simulation_times)

    def get_curr_colors(self):
        return traci.trafficlight.getRedYellowGreenState(self.tls_id)

    def get_phase_colors(self, phase_index):
        return self.get_tls_all_phases()[phase_index].state

    def get_curr_phase_encoding(self):
        num_phases = len(self.get_tls_all_phases())
        curr_phase_index = self.get_curr_phase()
        phases_encoding_list = [0]*num_phases
        phases_encoding_list[curr_phase_index] = 1
        return phases_encoding_list
    
    def get_green_lanes_in_curr_phase(self):
        curr_colors = self.get_curr_colors()
        green_lanes = []
        for color in curr_colors:
            if color.upper() == "G":
                green_lanes.append(1)
            else:
                green_lanes.append(0)
        return green_lanes

    def get_max_pressure_lanes(self):
        """for each green phase, get all incoming
        and outgoing lanes for that phase, store
        in dict for max pressure calculation
        """
        max_pressure_lanes = {}
        for green_phase in self.green_phases:
            phase_index, phase_state = green_phase
            green_lanes_indexes = [index for index, char in enumerate(phase_state) if char in ['g', 'G']]
            # Incoming lanes: Lanes where vehicles enter the intersection
            incoming_lanes = []
            # Outgoing lanes: Lanes where vehicles exit the intersection
            outgoing_lanes = []
            for green_lane_index in green_lanes_indexes:
                green_lanes = traci.trafficlight.getControlledLinks(self.tls_id)[green_lane_index]
                for link in green_lanes:
                    incoming_lane = link[0]
                    outgoing_lane = link[1]

                    if incoming_lane not in incoming_lanes:
                        incoming_lanes.append(incoming_lane)
                    if outgoing_lane not in outgoing_lanes:
                        outgoing_lanes.append(outgoing_lane)
            max_pressure_lanes[phase_index] = {'inc':incoming_lanes, 'out':outgoing_lanes}
        return max_pressure_lanes

    def max_pressure(self):
        phase_pressure = {}
        no_vehicle_phases = []
        #compute pressure for all green movements
        inc, out = self.get_in_out_vehicles_on_each_lane()
        for green_phase in self.green_phases:
            phase_index, phase_state = green_phase
            # if already selected this phase in current round, skip it.
            if self.green_phases_mask[phase_index] == 0:
                continue
            inc_lanes = self.max_pressure_lanes[phase_index]['inc']
            out_lanes = self.max_pressure_lanes[phase_index]['out']
            # print("phase_index = ", phase_index)
            # print("inc_lanes = ", inc_lanes)
            # print("inc = ", inc)


            #pressure is defined as the number of vehicles in a lane
            inc_pressure = sum([ inc[l] for l in inc_lanes])
            out_pressure = sum([ out[l] for l in out_lanes])
            # print("inc_pressure = ", inc_pressure)
            # print("out_pressure = ", out_pressure)
            phase_pressure[phase_index] = inc_pressure# - out_pressure
            if inc_pressure == 0 and out_pressure == 0:
                no_vehicle_phases.append(phase_index)

        selected_phase = None
        ###if no vehicles randomly select a phase
        if len(no_vehicle_phases) == len(self.green_phases):
            selected_phase = random.choice(self.green_phases)[0]
        else:
            #choose phase with max pressure
            #if two phases have equivalent pressure
            #select one with more green movements
            #return max(phase_pressure, key=lambda p:phase_pressure[p])
            phase_pressure = [ (p, phase_pressure[p]) for p in phase_pressure]
            phase_pressure = sorted(phase_pressure, key=lambda p:p[1], reverse=True)
            phase_pressure = [ p for p in phase_pressure if p[1] == phase_pressure[0][1] ]
            selected_phase = random.choice(phase_pressure)[0]

        # print("selected_phase = ", selected_phase)
        # mark selected phase in the mask, so we don't select it in same round
        # if selected_phase != self.get_curr_phase():
        #     self.green_phases_mask[selected_phase] = 0
        return selected_phase
