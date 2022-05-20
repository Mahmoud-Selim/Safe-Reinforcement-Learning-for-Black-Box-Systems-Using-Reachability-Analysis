#!/usr/bin/env python3
import os
import rospy
import numpy as np
import math
from math import pi
import random

from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel
#import joblib
from tf_agents.trajectories.time_step import TimeStep as tfTimeStep
from tf_agents.trajectories.time_step import StepType as tfStepType
from tf_agents.environments import py_environment
import tf_agents
from brsl_msgs.srv import *

diagonal_dis = math.sqrt(2) * (3.6 + 3.8)
#goal_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..', 'turtlebot3_simulations',
#                              'turtlebot3_gazebo', 'models', 'Target', 'model.sdf')



class TFEnv(py_environment.PyEnvironment):
    def __init__(self, discount=1.0):
        is_training = True
        super().__init__()
        self.one_round_step = 0

        self.discount = discount

        rospy.wait_for_service("/environment/GetActionSpec")
        service_proxy = rospy.ServiceProxy('/environment/GetActionSpec', GetActionSpec)
        msg = GetActionSpecRequest()
        resp = service_proxy(msg)
        self._action_spec = tf_agents.specs.BoundedArraySpec(shape=(len(resp.maximum),), dtype=np.float32, name="action", minimum=resp.minimum, maximum=resp.maximum)

        
        rospy.wait_for_service("/environment/GetObservationSpec")
        service_proxy = rospy.ServiceProxy('/environment/GetObservationSpec', GetObservationSpec)
        msg = GetObservationSpecRequest()
        resp = service_proxy(msg)
        self.observation_length = len(resp.maximum)
        self._observation_spec = tf_agents.specs.BoundedArraySpec(shape=(len(resp.maximum),), dtype=np.float32, name="observation", 
                                                                    minimum=resp.minimum, maximum=resp.maximum)


        rospy.wait_for_service("/environment/EnvReset")
        service_proxy = rospy.ServiceProxy('/environment/EnvReset', EnvReset)
        msg = EnvResetRequest()
        resp = service_proxy(msg)
        self._state = resp.observation
        

        self.collisions_number = 0
        self.arrivals_number = 0
        self.safety_interventions_number = 0
        self.total_stopping_number = 0
        self.episodes_number = 0
        self.acc_robot_speed = 0
        self.logging = False

    def get_odom(self):
        rospy.wait_for_service("/environment/GetOdom")
        service_proxy = rospy.ServiceProxy('/environment/GetOdom', GetOdom)
        msg = GetOdomRequest()
        resp = service_proxy(msg)
        return resp

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def enable_logging(self):
        self.logging = True

    def _reset(self):
        rospy.wait_for_service("/environment/EnvReset")
        service_proxy = rospy.ServiceProxy('/environment/EnvReset', EnvReset)
        msg = EnvResetRequest()
        resp = service_proxy(msg)
        self._state = resp.observation
        self._episode_ended = False
        return tf_agents.trajectories.time_step.restart(np.array([self._state], dtype=np.float32).reshape((self.observation_length,)))


    def _step(self, action):
        
        rospy.wait_for_service("/environment/EnvStep")
        service_proxy = rospy.ServiceProxy('/environment/EnvStep', EnvStep)
        msg = EnvStepRequest()
        msg.action = list(action.astype(np.float32))
        resp = service_proxy(msg)
        self._state = resp.observation

        
        #self.one_round_step +=1
        # done flag happens when the agent hits an obstacle. That is it has failed
        # arrive happens when the agent arrives successfully at the goal location
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            #self.one_round_step = 0
            return self._reset()
        state, reward, episode_ended = np.array(resp.observation).astype(np.float32).reshape((self.observation_length,)), resp.reward, resp.episode_ended

        if episode_ended:
            self._episode_ended = True
            if(resp.arrived and self.logging):
                self.arrivals_number += 1
            elif(resp.collision and self.logging):
                self.collisions_number += 1
            self.episodes_number += 1
            return tf_agents.trajectories.time_step.termination(state, reward)
        else:
            return tf_agents.trajectories.time_step.transition(state, reward, self.discount)


        
