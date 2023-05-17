import numpy as np
import pandas as pd
import carla
import os
import sys
import csv
import asyncio
from constants import *
from aggressive_driver import AggressiveDriver
try:
    sys.path.append(os.path.dirname((os.getcwd())) + os.sep + "carla")
except IndexError:
    pass

from agents.navigation.basic_agent import BasicAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner

import nest_asyncio
nest_asyncio.apply()

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
        
        self._agg_driver = AggressiveDriver(self._world, self._vehicle, waypoints, target_aggIn, opt_dict)
        
        self._time_array = []
        self._imu_array = []
        self._velocity_array = []
        self._target_velocity_array = []
        self._throttle_array = []
        self._brake_array = []
        
        
    def _init_cycle(self):
        # Start and then stop the vehicle
        time_array = []
        target_velocity = 10
        duration = 10
        self._agg_driver.set_agent_options({'follow_speed_limits' : False})            
        
        t0 = self._world.get_snapshot().timestamp.elapsed_seconds
        while True:
            t = self._world.get_snapshot().timestamp.elapsed_seconds
            time_array.append(t - t0)
            if time_array[-1] >= duration - 2.5:
                self._agg_driver.set_target_speed(0)
            else:
                self._agg_driver.set_target_speed(target_velocity)
            self._agg_driver.get_vehicle().apply_control(self._agg_driver.run_step())
            self._world.tick()

            if time_array[-1] >= duration:
                break   
        
        self._agg_driver.reset()
        # Force the vehicle to brake for 3 seconds
        time_array = []
        t0 = self._world.get_snapshot().timestamp.elapsed_seconds
        while True:
            t = self._world.get_snapshot().timestamp.elapsed_seconds
            time_array.append(t - t0)
            self._agg_driver.get_vehicle().apply_control(carla.VehicleControl(throttle=0, brake=0.3, steer=0))
            self._world.tick()

            if time_array[-1] >= 3:
                self._agg_driver.get_vehicle().apply_control(carla.VehicleControl(throttle=0, brake=0, steer=0))
                break   
        
        print("Initialization cycle completed!")

            
        
    async def sim(self, filename, max_duration, speed_profile = None, speed_profile_dt = 0.1, stop_at_end_pos = True):
        """Starts the simulation by running concurrently the main simulation loop and an async printer that saves the data to a csv file.

        Parameters
        ----------
            filename : str
                Path of the csv file where the data will be saved.
            max_duration : float
                Maximum duration of the simulation [sec]. If the stop_at_end parameter is set to True, the simulation will stop when the vehicle
                 reaches the end of the route or when this maximum duration is reached.
            speed_profile : list of float, optional 
                Profile of speed. Defaults to None.
            speed_profile_dt (float, optional): _description_. Defaults to 0.1.
            stop_at_end (bool, optional): _description_. Defaults to True.
        """
        
        if max_duration <= 0:
            raise ValueError("The maximum duration must be positive!")
        
        self._speed_profile = speed_profile
        self._speed_profile_dt = speed_profile_dt
        self._max_duration = max_duration
        self._stop_at_end = stop_at_end_pos

        self._agg_driver.reset()
        # Perform init cycle
        self._init_cycle()
        # Initialize the arrays that will contain the data
        self._time_array.clear()
        self._imu_array.clear()
        self._velocity_array.clear()
        self._target_velocity_array.clear()
        self._throttle_array.clear()
        self._brake_array.clear()
            
        main_task = asyncio.create_task(self._main_loop())
        save_task = asyncio.create_task(self._save_to_csv(main_task, filename))
        done, pending = await asyncio.wait([main_task, save_task], return_when=asyncio.ALL_COMPLETED)    
        
        print("Simulation completed!")
    
        
    async def _main_loop(self): 
        # Initialize the arrays that will contain the data
        self._time_array.clear()
        self._imu_array.clear()
        self._velocity_array.clear()
        self._target_velocity_array.clear()
        self._throttle_array.clear()
        self._brake_array.clear()   
        
        # Make the IMU start listening  
        self._agg_driver.get_imu().listen(lambda imu_data: self._imu_array.append({'AccX'  : imu_data.accelerometer.x,
                                                                                   'AccY'  : imu_data.accelerometer.y,
                                                                                   'AccZ'  : imu_data.accelerometer.z,
                                                                                   'GyroX' : imu_data.gyroscope.x,
                                                                                   'GyroY' : imu_data.gyroscope.y,
                                                                                   'GyroZ' : imu_data.gyroscope.z   }))
        # Initialize control
        control = carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0)         
        # If no speed profile is provided, the agent will follow the speed limits
        if self._speed_profile is None:
            self._agg_driver.set_agent_options(opt_dict={'follow_speed_limits': True})
        else:
            self._agg_driver.set_agent_options(opt_dict={'follow_speed_limits': False})
        
        t0 = self._world.get_snapshot().timestamp.elapsed_seconds
        t1 = t0

        # Run the simulation
        while True:
            # Retrieve time information from the simulation
            t = self._world.get_snapshot().timestamp.elapsed_seconds
            # Compute target velocity
            if self._speed_profile is not None:
                target_velocity = self._speed_profile[int(t / self._speed_profile_dt)]
            else:
                target_velocity = self._agg_driver.get_vehicle().get_speed_limit()
            
            # Retrieve time and velocity from the simulation. Throttle and brake are computed by
            # the agent locally, so they don't need to be retrieved from the simulation. 
            velocity = self._vehicle.get_velocity()
            self._velocity_array.append(3.6 * np.sqrt((float(velocity.x)) ** 2 + (float(velocity.y)) ** 2 + (float(velocity.z)) ** 2))
            self._throttle_array.append(control.throttle)
            self._brake_array.append(control.brake)
            self._time_array.append(t - t0)
            self._target_velocity_array.append(target_velocity)
            
            # Update target speed (needed by the agent to run a step)
            if self._speed_profile is not None:
                self._agg_driver.set_target_speed(target_velocity)    # an else branch is not needed because the agent will follow the speed limits 
            
            # Update the control that will be applied (longitudinal control is updated only every 1 / F_CTRL_UPDATE sec)        
            if t - t1 >= 1 / F_CTRL_UPDATE:
                t1 = t
                control = self._agg_driver.run_step()
            else:
                control.steer = self._agg_driver.run_step().steer
            # Apply control to the vehicle (on the next tick)
            self._agg_driver.get_vehicle().apply_control(control)
            
            # Go ahead with simulation
            self._world.tick()

            # Check if time is over or if the vehicle has reached the end location
            if self._time_array[-1] >= self._max_duration or (self._stop_at_end and self._agg_driver.get_vehicle().get_location().distance(self._agg_driver.get_end_location()) < 0.5):
                break
            
            # Yield control
            await asyncio.sleep(0)
            
        print("Main loop completed!")            
            
            
    async def _save_to_csv(self, main_task, filename):
        with open(filename, mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Time', 'Velocity', 'Target Velocity', 'Throttle', 'Brake', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ'])
            while True:
                await asyncio.sleep(0)  # needed to avoid blocking the event loop
                # if all the arrays have at least one value
                if self._velocity_array and self._target_velocity_array and self._throttle_array and self._brake_array and self._time_array and self._imu_array:
                    # find how many tuples can be written
                    min_len = min(len(self._velocity_array), len(self._target_velocity_array), len(self._throttle_array), len(self._brake_array), len(self._time_array), len(self._imu_array))
                    # write the tuples
                    for i in range(min_len):
                        x = self._imu_array.pop(0)
                        writer.writerow([self._time_array.pop(0), self._velocity_array.pop(0), self._target_velocity_array.pop(0), self._throttle_array.pop(0), self._brake_array.pop(0), x['AccX'], x['AccY'], x['AccZ'], x['GyroX'], x['GyroY'], x['GyroZ']])
                elif main_task.done():
                    print("Exiting save task!")
                    break