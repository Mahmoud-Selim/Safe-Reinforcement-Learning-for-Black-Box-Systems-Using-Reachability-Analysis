#!/usr/bin/env python3
import os
import rospy
import numpy as np
import math
from math import pi
import random
import time
import glob
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_srvs.srv import Empty
from rosgraph_msgs.msg import Clock
from gazebo_msgs.srv import SpawnModel, DeleteModel
from tf_agents.trajectories.time_step import TimeStep as tfTimeStep
from tf_agents.trajectories.time_step import StepType as tfStepType
from tf_agents.environments import py_environment
import tf_agents
from brsl_msgs.srv import *

diagonal_dis = math.sqrt(2) * (3.6 + 3.8)
goal_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..', 'turtlebot3_simulations',
                              'turtlebot3_gazebo', 'models', 'Target', 'model.sdf')



class TbotEnv(py_environment.PyEnvironment):
    def __init__(self, discount=1.0):
        is_training = True
        super().__init__()
        self.discount = discount

        self._action_spec = tf_agents.specs.BoundedArraySpec(shape=(2,), dtype=np.float32, name="action", minimum=[0, -0.5], maximum=[0.25, 0.5])
        self._observation_spec = tf_agents.specs.BoundedArraySpec(shape=(26,), dtype=np.float32, name="observation", 
                                                                    minimum=[0] * 26,
                                                                    maximum=[1] * 26)

        self.gtime = 0
        self.scan = [0] * 18
        self.number_of_collisions = 0
        self.position = Pose()
        self.goal_position = Pose()
        self.goal_position.position.x = 0.
        self.goal_position.position.y = 0.
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.sub_clock = rospy.Subscriber('clock', Clock, self.get_gtime)
        self.sub_scan  = rospy.Subscriber('scan', LaserScan, self.get_scan)
        #self.sub_imu  = rospy.Subscriber('imu', Imu, self.getImu)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.goal = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.past_distance = 0.
        self.action_pred_X = None
        self.action_pred_X0 = None
        self.action_pred_X1 = None
        self.action_pred_U = None
        self.action_pred_Y = None
        self.last_state = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.last_action = np.array([0, 0])
        self.last_position = np.array([0, 0])
        self.data_size = 0
        self.imu_status = [0, 0, 0, 0]
        self.one_round_step = 0
        self.spawned_obects = []
        self.cleared = True
        self.cylinders_path = glob.glob(os.path.join(os.path.split(os.path.realpath(__file__))[0], "Turtlebot_Models", "cylinder*"))
        self.boxes_path = glob.glob(os.path.join(os.path.split(os.path.realpath(__file__))[0], "Turtlebot_Models", "Box*"))
        self.walls_poses = np.load(os.path.join(os.path.split(os.path.realpath(__file__))[0], "Turtlebot_Models", "Turtlebot_V1.npy"))
        GetActionSpec_srv = rospy.Service('~GetActionSpec', GetActionSpec, self.action_spec)
        GetObservationSpec_srv = rospy.Service('~GetObservationSpec', GetObservationSpec, self.observation_spec)
        EnvReset_srv = rospy.Service('~EnvReset', EnvReset, self._reset)
        EnvStep_srv = rospy.Service('~EnvStep', EnvStep, self._step)
        GetOdom_srv = rospy.Service('~GetOdom', GetOdom, self.GetOdom)
        #print(self.cylinders_path)
        #print(self.boxes_path)
        #print("*"*80)
        if is_training:
            self.threshold_arrive = 0.5
        else:
            self.threshold_arrive = 0.5


    def get_gtime(self, msg):
        self.gtime = msg.clock.nsecs


    def get_scan(self, msg):
        #print(msg)
        self.scan = msg

    def action_spec(self, req):
        response = GetActionSpecResponse()
        response.minimum = [0, -0.5]
        response.maximum = [0.25, 0.5]
        return response

    def observation_spec(self, req):
        
        response = GetObservationSpecResponse()
        response.minimum = [0] * 26
        response.maximum = [1] * 26
        return response


    def getGoalDistace(self):
        goal_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        self.past_distance = goal_distance

        return goal_distance

    def GetOdom(self, req):
        response = GetOdomResponse()
        response.odom = self.odom
        return response


    def getOdometry(self, odom):

        self.odom = [odom.pose.pose.position.x, odom.pose.pose.position. y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w, odom.twist.twist.linear.x, 
                    odom.twist.twist.linear.y, odom.twist.twist.linear.z, odom.twist.twist.angular.z]

        self.position = odom.pose.pose.position
        self.twist_state = odom.twist.twist.linear
        self.twist_ang = odom.twist.twist.angular
        #self.orientation 
        self.orientation = odom.pose.pose.orientation
        q_x, q_y, q_z, q_w = self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w
        yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))))

        if yaw >= 0:
             yaw = yaw
        else:
             yaw = yaw + 360

        rel_dis_x = round(self.goal_position.position.x - self.position.x, 1)
        rel_dis_y = round(self.goal_position.position.y - self.position.y, 1)

        # Calculate the angle between robot and target
        if rel_dis_x > 0 and rel_dis_y > 0:
            theta = math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x > 0 and rel_dis_y < 0:
            theta = 2 * math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y < 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y > 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x == 0 and rel_dis_y > 0:
            theta = 1 / 2 * math.pi
        elif rel_dis_x == 0 and rel_dis_y < 0:
            theta = 3 / 2 * math.pi
        elif rel_dis_y == 0 and rel_dis_x > 0:
            theta = 0
        else:
            theta = math.pi
        rel_theta = round(math.degrees(theta), 2)

        diff_angle = abs(rel_theta - yaw)

        if diff_angle <= 180:
            diff_angle = round(diff_angle, 2)
        else:
            diff_angle = round(360 - diff_angle, 2)

        self.rel_theta = rel_theta
        self.yaw = yaw
        self.diff_angle = diff_angle

    def getState(self, scan):
        scan_range = []
        yaw = self.yaw
        rel_theta = self.rel_theta
        diff_angle = self.diff_angle
        min_range = 0.2
        done = False
        arrive = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        if min_range > min(scan_range):
            done = True

        current_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        if current_distance <= self.threshold_arrive:
            # done = True
            arrive = True

        return scan_range, current_distance, yaw, rel_theta, diff_angle, self.orientation.z, self.orientation.w, \
                self.twist_state.x, self.twist_ang.z, done, arrive


    def spawn_object(self, object_path, name, pose):
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        #print(name)
        try:
            object_urdf = open(object_path, "r").read()
            target = SpawnModel
            target.model_name = name 
            target.model_xml = object_urdf
            objection_position = Pose()
            objection_position.position.x = pose[0]
            objection_position.position.y = pose[1]
            self.goal(target.model_name, target.model_xml, 'namespace', objection_position, 'world')

            if(name == "goal"):
                self.goal_position = objection_position

        except (rospy.ServiceException) as e:
            print("/gazebo/failed to build the target")

    def spawn_goal(self, cylinders_poses, boxes_poses, walls_poses):
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            goal_urdf = open(goal_model_dir, "r").read()
            target = SpawnModel
            target.model_name = "goal" 
            target.model_xml = goal_urdf

            goal_position_valid = False

            while(True):
                pose = np.random.uniform(-5, 5, (2))

                cylinders_distances = np.linalg.norm(cylinders_poses - pose, axis = 1)
                boxes_distances = np.linalg.norm(boxes_poses - pose, axis = 1)
                walls_distances = np.linalg.norm(walls_poses - pose, axis = 1)


                cylinder_threshold = 0.5 + 0.5
                boxes_threshold = 0.5 * np.sqrt(2) + 0.5
                walls_threshold = np.sqrt(0.5**2 + 0.5**2)

                if(np.all(cylinders_distances > cylinder_threshold) and np.all(boxes_distances > boxes_threshold) and \
                   np.all(walls_distances > walls_threshold)):
                    break
            objection_position = Pose()
            objection_position.position.x = pose[0]
            objection_position.position.y = pose[1]
            self.goal(target.model_name, target.model_xml, 'namespace', objection_position, 'world')

            self.goal_position = objection_position

        except (rospy.ServiceException) as e:
            print("/gazebo/failed to build the target")

    def spawn_goal_obstacles(self):
        self.spawned_obects = []
        n_cylinders = int(np.random.choice(7))
        n_boxes = int(np.random.choice(7))

        cylinders = np.random.choice(self.cylinders_path, n_cylinders, False)
        boxes     = np.random.choice(self.boxes_path, n_boxes, False)


        cylinders_poses = np.random.uniform(-8.2, 8.2, (n_cylinders, 2))
        boxes_poses = np.random.uniform(-8.2, 8.2, (n_boxes, 2))
        cylinders_poses[np.argwhere(np.linalg.norm(cylinders_poses, axis = 1) < 1)] += 4
        boxes_poses[np.argwhere(np.linalg.norm(boxes_poses, axis = 1) < 1)] += 4

        for i, cylinder in enumerate(cylinders):
            self.spawn_object(os.path.join(cylinder, "model.sdf"), os.path.basename(cylinder), cylinders_poses[i])
            self.spawned_obects.append(os.path.basename(cylinder))

        for i, box in enumerate(boxes):
            self.spawn_object(os.path.join(box, "model.sdf"), os.path.basename(box), boxes_poses[i])
            self.spawned_obects.append(os.path.basename(box))

        
        
        self.spawn_goal(cylinders_poses, boxes_poses, self.walls_poses)

        self.cleared = False

    def clear_goal_obstacles(self):
        if(self.cleared == False):
            rospy.wait_for_service('/gazebo/delete_model')
            self.del_model('goal')
            for spawned_object in self.spawned_obects:
                self.del_model(spawned_object)
            self.cleared = True


    def setReward(self, done, arrive):
        current_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        distance_rate = (self.past_distance - current_distance)

        reward = 20.*distance_rate 
        #reward = 10/(current_distance + 0.1)
        self.past_distance = current_distance

        if done:
            reward = -100
            self.number_of_collisions += 1
            self.pub_cmd_vel.publish(Twist())

        if arrive:
            reward = 100
            self.pub_cmd_vel.publish(Twist())

            #rospy.wait_for_service('/gazebo/unpause_physics')
            self.goal_distance = self.getGoalDistace()
            arrive = False

        return reward

    def Gstep(self, action):
        start_t = self.gtime

        linear_vel = action[0]
        ang_vel = action[1]

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = ang_vel

        self.unpause_proxy()
        self.pub_cmd_vel.publish(vel_cmd)

        while(np.abs(self.gtime - start_t) < 100):
            continue
        #data = None
        #while data is None:
        #    try:
        #        data = rospy.wait_for_message('scan', LaserScan, timeout=5)
        #    except:
        #        pass

        self.pause_proxy()
        state, rel_dis, yaw, rel_theta, diff_angle, orientation_z, orientation_w, twist_state_x, twist_ang_z, done, arrive = self.getState(self.scan)
        state = [i / 3.5 for i in state]

        state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180, orientation_z/6.29,
                         orientation_w/6.29, twist_state_x/6.29, twist_ang_z/6.29]

        reward = self.setReward(done, arrive)

        return state, reward, done, arrive#np.asarray(state, dtype=np.float32).reshape((14,)), reward, done, arrive


    def Gstep2(self, action):
        start_t = time.time()
        times = []
        linear_vel = action[0]
        ang_vel = action[1]

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        #vel_cmd.linear.y = ang_vel
        vel_cmd.angular.z = ang_vel

        self.pub_cmd_vel.publish(vel_cmd)
        times.append(time.time() - start_t)
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass
        times.append(time.time() - start_t)
        state, rel_dis, yaw, rel_theta, diff_angle, orientation_z, orientation_w, twist_state_x, twist_ang_z, done, arrive = self.getState(data)
        state = [i / 3.5 for i in state]

        #for pa in past_action:
        #    state.append(pa)

        state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180, orientation_z/6.29,
                         orientation_w/6.29, twist_state_x/6.29, twist_ang_z/6.29]
        """
        state,gx, x, gy, y, done, arrive = self.getState(data)
        #done, arrive = state[-2], state[-1]
        state = state + [past_action[0], past_action[1], x, y, gx, gy]
        #print("++++++++++++++++++++", len(state))
        
        """
        reward = self.setReward(done, arrive)


        times.append(time.time() - start_t)
        #print("Step took {}".format(times))
        return state, reward, done, arrive#np.asarray(state, dtype=np.float32).reshape((14,)), reward, done, arrive


    def Greset(self):
        # Reset the env 
        self.clear_goal_obstacles()

        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        self.unpause_proxy()
        # Build the targetz

        self.spawn_goal_obstacles()
        #rospy.wait_for_service('/gazebo/unpause_physics')
        

        self.goal_distance = self.getGoalDistace()
        state, rel_dis, yaw, rel_theta, diff_angle, orientation_z, orientation_w, twist_state_x, twist_ang_z, done, arrive = self.getState(self.scan)
        state = [i / 3.5 for i in state]

        state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180, orientation_z/6.29,
                         orientation_w/6.29, twist_state_x/6.29, twist_ang_z/6.29]
        self.pause_proxy()
        return state



    def position():
        return self.position
        
    def _reset(self, req):
        self._state = self.Greset()
        self._episode_ended = False
        response = EnvResetResponse()
        response.observation = self._state
        return response


    def _step(self, req):
        #print("*"*80)
        #print("Entered Step {}".format(req.action))
        self.one_round_step +=1
        # done flag happens when the agent hits an obstacle. That is it has failed
        # arrive happens when the agent arrives successfully at the goal location
        state, reward, done, arrive = self.Gstep(req.action)
        response = EnvStepResponse()

        if arrive or done or self.one_round_step >= 4000:
            self.one_round_step = 0
            #print("done or arrive")
            if(arrive):
                response.collision = False
                response.arrived = True
                response.time_elapsed = False
            elif(done):
                response.collision = True
                response.arrived = False
                response.time_elapsed = False
            else:
                response.collision = False
                response.arrived = False
                response.time_elapsed = True

            response.episode_ended = True
            response.observation = state 
            response.reward = reward 
            response.episode_ended = True
            return response#tf_agents.trajectories.time_step.termination(state, reward)
        else:
            response.observation = state 
            response.reward = reward 
            response.episode_ended = False
            response.collision = False
            response.arrived = False
            response.time_elapsed = False
            return response#tf_agents.trajectories.time_step.transition(state, reward, self.discount)


def main():
    rospy.init_node("environment")
    turtlebot_environment = TbotEnv()
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass