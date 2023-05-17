import numpy as np
import pandas as pd
import carla
import os
import sys
import sympy as sp
from constants import *

try:
    sys.path.append(os.path.dirname((os.getcwd())) + os.sep + "carla")
except IndexError:
    pass

from agents.navigation.basic_agent import BasicAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner


class AggressiveDriver:
    """This class represents the aggressive driver. 
    It contains the vehicle and the agent that will control it. The latter will have such parameters to satisfy the target aggressiveness index.
    The class offers the possibility to change the target aggressiveness index and update the agent accordingly. Methods to compute the correct 
    controller parameters are protected.
    """
    
    def __init__(self, world, vehicle, waypoints, target_aggIn = 107, opt_dict = {}):  
        """Constructor method used to spawn the vehicle, create the route from the list of waypoints,
        create the agent and link it to the computed route.

        --------------
        ### Parameters
            `world` : carla.World
                The world object representing the simulation. NOTE that the world must be in synchronous mode 
                 and have the same dt as the one specified in the constructor.
            `waypoints` : list of carla.Waypoint
                List of waypoints that will be used to create the route that the vehicle will follow.
            `target_aggIn` : int, optional
                Target aggressivity index (between 70 and 160). Defaults to 107.
            `vehicle_bp_ID` : str, optional
                String that identifies the blueprint of the vehicle to be used. NOTE that the system has been thought for
                 single-speed transmission. Defaults to 'vehicle.tesla.model3'.
            `dt` : float, optional
                Delta time of the simulation. Defaults (and highly recommended) to 0.005 [sec].
            `f_long_update` : float, optional
                Frequency of longitudinal control update. Defaults (and highly recommended) to 10 [Hz].
            `opt_dict` : dict, optional 
                Contains some possible options for the agent. Defaults is empty.

        ----------
        ### Raises
            `ValueError` 
                If the world's settings are incorrect or if the target aggressiveness index is out of range.

        """        
        
        # check the vehicle, then teleport it to the first waypoint
        if vehicle is None:
            raise ValueError("The vehicle has not be spawned!")
        self._world = world
        self._vehicle = vehicle
        self._vehicle.set_location(waypoints[0].location)
           
        # create the route
        self._route = []
        self._start_location = waypoints[0].location
        self._end_location = waypoints[-1].location      
        self._waypoints = waypoints  
        self._waypoints.pop(0)    # removes the first waypoint, which is the spawn point
        grp = GlobalRoutePlanner(world.get_map(), 2.0)
        for i in range(len(self._waypoints) - 1):
            partial_route = grp.trace_route(self._waypoints[i].location, self._waypoints[i+1].location)
            partial_route.pop(0)    # prevents first element of every partial route to be added twice, resulting in unexpected
                                    # decelerations of the vehicle when it gets to the chosen spawn points
            for x in partial_route:
                self._route.append(x)
               
        if target_aggIn < AGG_IN_MIN or target_aggIn > AGG_IN_MAX:
            raise ValueError("Aggressiveness index out of range")
        
        # compute the best PID and create the agent
        self._aggIn = target_aggIn
        self._PID_parameters = self._bestPID()
        self._agent = BasicAgent(self._vehicle, opt_dict={'dt' : 1 / F_CTRL_UPDATE, 
                                                          'longitudinal_control_dict' : {'K_P': self._PID_parameters['KP'],
                                                                                         'K_I': self._PID_parameters['KI'], 
                                                                                         'K_D': self._PID_parameters['KD'], 
                                                                                         'dt': 1 / F_CTRL_UPDATE, 
                                                                                         'anti_windup': True                }   }   )    
        self._agent.set_global_plan(plan=self._route, stop_waypoint_creation=False)    
        self._opt_dict = opt_dict
        if 'ignore_traffic_lights' in opt_dict:
            self._agent.ignore_traffic_lights(active = opt_dict['ignore_traffic_lights'])
        if 'ignore_stop_signs' in opt_dict:
            self._agent.ignore_stop_signs(active = opt_dict['ignore_stop_signs'])
        if 'ignore_vehicles' in opt_dict:
            self._agent.ignore_vehicles(active = opt_dict['ignore_vehicles'])
        if 'follow_speed_limits' in opt_dict:
            self._agent.follow_speed_limits(value = opt_dict['follow_speed_limits'])
        # attach IMU sensor to the vehicle (NOTE that the IMU still has to be activated by specifying its callback function)   
        imu_location = carla.Location(0,0,0)
        imu_rotation = carla.Rotation(0,0,0)
        imu_transform = carla.Transform(imu_location, imu_rotation)
        self._imu = self._world.spawn_actor(self._world.get_blueprint_library().find('sensor.other.imu'), imu_transform, 
                                            attach_to=self._vehicle, 
                                            attachment_type=carla.AttachmentType.Rigid)   

    def run_step(self):
        return self._agent.run_step()
        
    def set_target_speed(self, target):
        self._agent.set_target_speed(target)        
        
    def get_start_location(self):
        return self._start_location
    
    def get_end_location(self):
        return self._end_location
        
    def get_aggIn(self):       
        return self._aggIn   
    
    def get_imu(self):
        return self._imu
    
    def get_vehicle(self):
        return self._vehicle
    
    def get_PID_parameters(self):
        return self._PID_parameters

    def get_agent_options(self):
        return self._opt_dict

    
    def set_plan(self, waypoints):
        if self._vehicle is None:
            raise ValueError("The vehicle has not be spawned!")
        self._vehicle.set_location(waypoints[0].location)
        self._start_location = waypoints[0].location
   
        # create the route
        self._route = []
        waypoints.pop(0)    # removes the first waypoint, which is the spawn point
        grp = GlobalRoutePlanner(self._world.get_map(), 2.0)
        for i in range(len(waypoints) - 1):
            partial_route = grp.trace_route(waypoints[i].location, waypoints[i+1].location)
            partial_route.pop(0)    # prevents first element of every partial route to be added twice, resulting in unexpected
                                    # decelerations of the vehicle when it gets to the chosen spawn points
            for x in partial_route:
                self._route.append(x)
                
        self._agent.set_global_plan(plan=self._route, stop_waypoint_creation=False)
        
        
    def set_agent_options(self, opt_dict):
        self._opt_dict.update(opt_dict)
        if 'ignore_traffic_lights' in opt_dict:
            self._agent.ignore_traffic_lights(active = opt_dict['ignore_traffic_lights'])
        if 'ignore_stop_signs' in opt_dict:
            self._agent.ignore_stop_signs(active = opt_dict['ignore_stop_signs'])
        if 'ignore_vehicles' in opt_dict:
            self._agent.ignore_vehicles(active = opt_dict['ignore_vehicles'])
        if 'follow_speed_limits' in opt_dict:
            self._agent.follow_speed_limits(value = opt_dict['follow_speed_limits'])
    
    
    def set_target_aggIn(self, target_aggIn):
        if target_aggIn < AGG_IN_MIN or target_aggIn > AGG_IN_MAX:
            raise ValueError("Aggressiveness index out of range")
        # set the target aggressiveness index and compute the best PID controller that satisfies it
        self._aggIn = target_aggIn
        self._PID_parameters = self._bestPID()
        # update the agent with the new PID controller
        self._agent = BasicAgent(self._vehicle, opt_dict={'dt' : 1 / self._f_long_update,
                                                          'longitudinal_control' : {'K_P': self._PID_parameters['KP'],
                                                                                    'K_I': self._PID_parameters['KI'], 
                                                                                    'K_D': self._PID_parameters['KD'], 
                                                                                    'dt': 1 / self._f_long_update, 
                                                                                    'anti_windup': True}})
        # apply back the options
        for o in self._opt_dict:
            if o == 'ignore_traffic_lights':
                self._agent.ignore_traffic_lights(active = self._opt_dict[o])
            elif o == 'ignore_stop_signs':
                self._agent.ignore_stop_signs(active = self._opt_dict[o])
            elif o == 'ignore_vehicles':
                self._agent.ignore_vehicles(active = self._opt_dict[o])
            elif o == 'follow_speed_limits':
                self._agent.follow_speed_limits(active = self._opt_dict[o])
    
    def reset(self):
        # reset the location of the vehicle
        self._vehicle.set_location(self._start_location)


    def _predict_aggIn(self, KP, KD):
        """This function computes an approximation of the aggressiveness index based on the KP and KD values of the PID controller,
        while considering the KI value to be 0.01. The equation has been obtained by fitting a polynomial of degree 3 to the data obtained
        from the simulation of the PID controller.

        --------------
        ### Parameters
            `KP` : float
                Proporcional term of the PID controller.
            `KD` : float
                Derivative term of the PID controller.

        -----------
        ### Returns
            `out` : float
                Approximation of the aggressiveness index of the PID controller if it would be used to follow FTP-75 driving cycle.
        """
        args = OPT_PARAMETERS
        
        return args[0] + args[1]*KP + args[2]*KD + args[3]*KP*KD + args[4]*KP**2 + args[5]*KD**2 + args[6]*KP**3 + args[7]*KD**3 + args[8]*KP*KD**2 + args[9]*KP**2*KD



    def _possible_PIDs(self, k = 10):
        """Provides a list of at most k possible PID controllers that satisfy the target aggressiveness index.
        To do so it generates a random KP from a normal distribution with mean depending on the target aggressiveness index 
        and then finds the possible KD values that satisfy the target aggressiveness index with the current KP.
        The solutions are filtered to keep only the real ones, that are in the range and that are not too close to the ones already in the list.
        The number of elements in the list can be smaller than k to speed up the process when the loop keeps finding the too similar solutions.

        --------------
        ### Parameters
            `k` : int, optional
                The maximum number of solutions to return. Defaults to 10.

        -----------
        ### Returns
            `out` : list
                List of at most k possible PID controllers that satisfy the target aggressiveness index.

        ----------
        ### Raises
            `ValueError`
                When aggressiveness index is out of range (must be between 0 and 1).
        """
        MAX_ITER = 300
        PRECISION_on_KP = 0.003
        
        res_list = []
        i = 0
        
        while (len(res_list) < k or (i < MAX_ITER and len(res_list) <= 1)):
            # generate KP from a random normal distribution with mean depending on normalized aggressiveness index   
            KP = 0
            while(KP < KP_A or KP > KP_B):
                KP = np.random.normal((KP_B - KP_A) * (1 + 2 * (self._aggIn - AGG_IN_MIN) / (AGG_IN_MAX - AGG_IN_MIN)) / 4, (KP_B - KP_A) / 3)
                
            # find possible KD values that sarisfy the target aggressiveness index with the current KP
            KD = sp.Symbol('KD')
            eq = sp.Eq(self._predict_aggIn(KP, KD), self._aggIn)
            sol = sp.solve(eq, KD)
            
            # filter the solutions to keep only the real ones and the ones in the range
            sol = [s for s in sol if s.is_real and s >= KD_A and s <= KD_B]
            
            # append the solutions to the result list only if they are not too close to the ones already in the list
            for s in sol:
                if len(res_list) < 1 or len([d for d in res_list if abs(d['KP'] - KP) < PRECISION_on_KP]) == 0:
                    res_list.append({'KP' : float(KP), 'KI' : DEF_KI, 'KD' : float(s)}) 
                else:
                    k -= 1
            i += 1
            
        return res_list


    def _bestPID(self):
        """Given a target aggressiveness index, it returns the best PID controller that satisfies it.
        To do so it computes a list of possible PIDs calling the `possible_PIDs()` function and then it finds the one that has a KP parameter
        closest to the center of the range, shifted according to normalized aggIn.

        -----------
        ### Returns
            `out` : dict
                A dictionary containing the KP, KI, KD parameters of the best PID controller that satisfies the 
                target aggressiveness index
        """
        KP_center = (KP_B - KP_A) / 2
        shifted_KP_center = KP_A + KP_center + (KP_B - KP_A) * ((self._aggIn - AGG_IN_MIN) / (AGG_IN_MAX - AGG_IN_MIN) - 0.5) * 0.8
        
        possible_PIDs_list = self._possible_PIDs()
        
        min_dist = float('inf')
        best_PID = {}
        
        for PID in possible_PIDs_list:
            dist = np.abs(shifted_KP_center - PID['KP'])
            if dist < min_dist:
                min_dist = dist
                best_PID.clear()
                best_PID.update(PID)    
        
        return best_PID
    
    
    