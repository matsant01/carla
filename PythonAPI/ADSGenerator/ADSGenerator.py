import numpy as np
import pandas as pd
import carla
import os
import sys
import sympy as sp
from constants import *
import AggressiveDriver

try:
    sys.path.append(os.path.dirname((os.getcwd())) + os.sep + "carla")
except IndexError:
    pass

from agents.navigation.basic_agent import BasicAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner

class ADSGenerator:
    def __init__(self, world, vehicle, waypoints, target_aggIn = 107, dt = 0.005, opt_dict = {}):  
        """Constructor method.

        Parameters
        ----------
            world : carla.World
                The world object representing the simulation. NOTE that the world must be in synchronous mode 
                 and have the same dt as the one specified in the constructor.
            waypoints : list of carla.Waypoint
                List of waypoints that will be used to create the route that the vehicle will follow.
            target_aggIn : int, optional
                Target aggressivity index (between 70 and 160). Defaults to 107.
            vehicle_bp_ID : str, optional
                String that identifies the blueprint of the vehicle to be used. NOTE that the system has been thought for
                 single-speed transmission. Defaults to 'vehicle.tesla.model3'.
            dt : float, optional
                Delta time of the simulation. Defaults (and highly recommended) to 0.005 [sec].
            f_long_update : float, optional
                Frequency of longitudinal control update. Defaults (and highly recommended) to 10 [Hz].
            opt_dict : dict, optional 
                Contains some possible options for the agent. Defaults is empty.

        Raises
        ------
            ValueError 
                If the world's settings are incorrect or if the target aggressiveness index is out of range.

        """
           
        # check if the world is consistent with the parameters
        settings = world.get_settings()
        if not settings.synchronous_mode:
            raise ValueError("The world must be in synchronous mode!")
        if settings.fixed_delta_seconds != dt:
            raise ValueError("The world must have the same dt as the one specified in the constructor!")
        self._world = world
        self._dt = dt
        
        # check if the vehicle has been spawned correctly
        self._vehicle = vehicle
        if self._vehicle is None:
            raise ValueError("The vehicle has not be spawned!")
        
        self._agg_driver = AggressiveDriver.AggressiveDriver(self._world, self._vehicle, waypoints, target_aggIn, opt_dict)
        
    
        
    def sim(self, duration): 
        self._agg_driver.reset_position()
              
        velocity = 0
        time_array = []
        
        t0 = self._world.get_snapshot().timestamp.elapsed_seconds
        t1 = t0
        
        control = carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0)         

        while True:
            # Retrieve time and velocity from the simulation. Throttle and brake are computed by
            # the agent locally, so they don't need to be retrieved from the simulation. 
            t = self._world.get_snapshot().timestamp.elapsed_seconds
            
            # Collect signals
                # velocity = self._vehicle.get_velocity()
                # velocity_array.append(3.6 * np.sqrt((float(velocity.x)) ** 2 + (float(velocity.y)) ** 2 + (float(velocity.z)) ** 2))
                # target_velocity_array.append(vehicle.get_speed_limit())
                # throttle_array.append(control.throttle)
                # brake_array.append(control.brake)
            time_array.append(t - t0)
            
            # Update the control that will be applied        
            if t - t1 >= 1 / F_CTRL_UPDATE:
                t1 = t
                control = self._agg_driver.get_agent().run_step()
            else:
                control.steer = self._agg_driver.get_agent().run_step().steer
            # Apply control
            self._agg_driver.get_vehicle().apply_control(control)
            
            # Go ahead with simulation
            self._world.tick()
            
            # Check wheter simulation is finished
            if time_array[-1] >= duration:
                break