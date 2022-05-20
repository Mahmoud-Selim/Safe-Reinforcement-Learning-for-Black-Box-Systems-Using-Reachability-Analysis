import numpy as np
from reachability.NLReachability import NLReachability
from multiprocessing import Pool
from helper import check_zono_intersection
import math
from reachability.Zonotope import Zonotope

class Pose:
    def __init__(self, *args):
        if(len(args) == 2):
            self.x = args[0] 
            self.y = args[1] 
        elif(len(args) == 1):
            self.x = args[0][0]
            self.y = args[0][1]
        
    def __sub__(self, *factor):
        result = Pose(self.x, self.y)
        if(len(factor) == 1):
            if(isinstance(factor[0], Pose)):
                factor = factor[0]
                result.x -= factor.x
                result.y -= factor.y                
            else:
                result.x -= factor[0]
                result.y -= factor[0]
        elif(len(factor) == 2):
            result.x -= factor[0]
            result.y -= factor[1]
            
        return result
    
    def __add__(self, *factor):
        result = Pose(self.x, self.y)
        if(len(factor) == 1):
            if(isinstance(factor[0], Pose)):
                factor = factor[0]
                result.x += factor.x
                result.y += factor.y                
            else:
                result.x += factor[0]
                result.y += factor[0]
        elif(len(factor) == 2):
            result.x += factor[0]
            result.y += factor[1]
            
        return result    
    def __mul__(self, factor):
        result = Pose(factor * self.x, factor * self.y)
        return result
    
    def __truediv__(self, factor):
        result = Pose(self.x / factor, self.y / factor)
        return result
        
    
    def as_tuple(self):
        return (self.x, self.y)
    
    def __str__(self):
        result = "X: {}, Y: {}".format(self.x, self.y)
        return result


class SafetyLayer():
    def __init__(self, env = "tb"):
        self.env = env
        self.nonlinear_reachability = NLReachability("/home/mahmoud/exp/Turtlebot/BRSL/Data/")
        self.old_plan = []
        for i in range(10):
            self.old_plan.append(np.array([0] * 8))
        self.old_plan = np.array(self.old_plan)
        self.pool = Pool()
        
        self.fail_safe = []
        for i in range(6):
            self.fail_safe.append([0., 0.25, 0., 0., 0., 0., 0., 0.])
        self.fail_safe = np.array(self.fail_safe)
        #print(self.fail_safe)
        #pass


    def get_rays_poses2d(self, readings):
        angles = [i * (np.pi/(len(readings) - 1)) for i in range(len(readings))]
        poses = []
        
        for angle in angles:
            poses.append(np.array((np.cos(angle), np.sin(angle))))

        objects = []
        max_range = 1
        rays_poses = []
        
        for i, reading in enumerate(readings):
            rays_poses.append(poses[i] * reading)

        return rays_poses

    
    def construct_objects(self, readings, _range = None):
        objects = []

        if(self.env == "tb"):
            active_rays = np.argwhere(readings <= 1).flatten()
            if(_range is None):
                _range = [-90, 90]

            #rays_poses = self.get_rays_poses(readings)

            i = 0
            while i < len(active_rays) - 2:
                if(active_rays[i] + 1 == active_rays[i + 1] and active_rays[i] + 2 == active_rays[i + 2]):
                    objects.append([active_rays[i], active_rays[i + 1], active_rays[i + 2]])
                    i += 3

                elif(active_rays[i] + 1 == active_rays[i + 1]):
                    objects.append([active_rays[i], active_rays[i + 1]])
                    i += 2

                else:
                    i += 1
                    continue

            remaining_active_rays = len(active_rays) - i

            if(remaining_active_rays == 2):
                objects.append([active_rays[i], active_rays[i + 1]])
            elif(remaining_active_rays == 1):
                if(active_rays[i] - 1 == active_rays[i - 1]):
                    objects.append([active_rays[i - 1], active_rays[i]])

        else:
            pass

        return objects


    
    def create_zonotope2d(self, indices, readings):
        zonotopes = []
        poses = self.get_rays_poses2d(readings)
        #print(poses)
        for idx in indices:
            first, last = poses[idx[0]], poses[idx[-1]]
            first_gen = np.array([last[0] - first[0], last[1] - first[1]])
            second_gen = np.array(poses[idx[0]])
            third_gen = np.array(poses[idx[-1]])


            first_gen_center = (poses[idx[0]] + poses[idx[-1]]) / 2
            perpendicular = np.array([-first_gen[1] / first_gen[0], 1])
            perpendicular = math.sqrt(3) * (perpendicular / np.linalg.norm(perpendicular))

            center = np.array((0, 0)).reshape((-1, 1))

            first_gen = first_gen / 2
            #print(np.arctan2(first_gen[1], first_gen[0]))
            second_gen = np.linalg.norm(first_gen) * (second_gen / np.linalg.norm(second_gen))
            third_gen = np.linalg.norm(first_gen) * (third_gen / np.linalg.norm(third_gen))

            generators = np.hstack([first_gen.reshape((-1, 1)), second_gen.reshape((-1, 1)), \
                                                   third_gen.reshape((-1, 1))])
            #print(center)
            #print(first_gen)
            #print(second_gen)
            #print(third_gen)

            centered_zonotope = Zonotope(center, generators)
            center_shift = first_gen_center
            vertices = centered_zonotope.polygon()
            #print(vertices)
            for i in range(vertices.shape[1] - 1):
                for j in range(1, vertices.shape[1]):
                    #print(i, j)
                    line = vertices[:, i] - vertices[:, j]
                    line_center = (vertices[:, i] + vertices[:, j]) / 2
                    #print(line_center, first_gen_center)
                    #print(np.abs(np.arctan2(line[1], line[0]) - np.arctan2(first_gen[1], first_gen[0])) , # Same Slope
                    #   (np.linalg.norm(line) / 2) - np.linalg.norm(first_gen) ,
                    #   np.dot(line_center, first_gen_center) )

                    if(np.abs(np.arctan2(line[1], line[0]) - np.arctan2(first_gen[1], first_gen[0])) < 0.001 and # Same Slope
                       (np.linalg.norm(line) / 2) - np.linalg.norm(first_gen) < 0.001 and
                       line_center @ first_gen_center < 0):
                        #print(line_center, first_gen_center)
                        center_shift -= line_center
                        #print(center_shift)
                        zonotopes.append(Zonotope(center_shift.reshape((-1, 1)), generators))
                    #print(np.arctan2(line[1], line[0]))
                    #print()
        return zonotopes    
    
    def enforce_safety(self, reachability_state, plan, readings):
        #print("in safety layer", readings)
        obstacles_indices = self.construct_objects(1.3 * readings)
        obstacles = self.create_zonotope2d(obstacles_indices, readings)
        plan = np.vstack((plan, self.fail_safe))
        #print(self.fail_safe)
        if(len(obstacles) > 0):
            for j in range(3):
                new_states, derivatives = self.nonlinear_reachability.run_reachability(reachability_state, plan)

                #print("derivatives are")
                #for derivative in derivatives:
                #    print(derivative)
                #print("*"*80)

                pose_states = [Zonotope(i.Z[:plan.shape[1], :plan.shape[1] + 1]) for i in new_states[1:]]

                #print(pose_states[-1].Z)
                new_states_rep = np.array(pose_states).reshape((-1, 1))
                #print("*"*80)
                #print(j)
                
                new_states_rep = np.hstack([new_states_rep] * len(obstacles)).flatten()

                zono_obs_pair = zip(new_states_rep, obstacles * len(new_states[1:]))
                #print(new_states_rep)
                #print(obstacles * len(new_states[1:]))
                ret = self.pool.map(check_zono_intersection, zono_obs_pair)
                ret = np.array(ret, dtype = object)
                #ret = np.hstack((ret[:, 0], ret[:, 1]))
                #print(ret)
                if(any(ret[:, 0]) and j <= 2):
                    #print("Adjusting Plan")
                    ret = ret.reshape((len(new_states[1:]), len(obstacles), 2))
                    #print(ret)
                    plan_updates = np.array([np.zeros_like(np.array([0, 0]))] * len(new_states[1:])).astype(np.float32)

                    upstream_gradient = np.ones_like(np.array([0, 0])).astype(np.float32).reshape((1, -1))

                    # Loop backwards to do gradient updates for the plan
                    for i in range(len(ret) - 1, -1, -1):
                        colliding_obstacles = ret[i][np.where(ret[i][:, 0])]
                        if(len(colliding_obstacles) != 0):
                            avg_gradient = np.mean(colliding_obstacles[:, 1]) * 100
                        else:
                            avg_gradient = np.zeros_like((np.array([0, 0]))).reshape((-1, 1))

                        #print(upstream_gradient.shape)
                        #print(derivatives[i][0].shape)
                        #print("derivatives are", derivatives[i][1])
                        if(i == len(ret) - 1):
                            #print("last = ", avg_gradient.T @ derivatives[i][1])
                            plan_updates[i] = (avg_gradient.T @ derivatives[i][1]).flatten()
                        else:
                            plan_updates[i] = ((avg_gradient.T  + upstream_gradient) @ \
                                               (derivatives[i][1])).flatten()
                        upstream_gradient = upstream_gradient @ derivatives[i][0]
                        #print("grad and updates are ", avg_gradient, plan_updates[i])
                        #print("Upstream grad ", upstream_gradient)


                        """step_obstacles = ret[i][:, 1] * derivatives[i][1]

                        #print(len(obstacles))
                        print(derivatives[i][1])
                        print(ret[i])
                        print(step_obstacles)
                        print(step_obstacles[np.where(step_obstacles[0, :] == True)])

                        avg_gradient = np.mean(step_obstacles[np.where(step_obstacles[0, :] == True)][1, :])
                        plan_updates[i] = upstream_gradient * avg_gradient

                        upstream_gradient * derivatives[i][0]"""
                    #print(plan_updates)
                    plan[:, :2] += (10 * plan_updates)
                    #print(plan)
                    plan[np.where(plan[:, 0] > 0.25), 0] = 0.25
                    plan[np.where(plan[:, 0] < 0), 0]    = 0
                    plan[np.where(plan[:, 1] > 0.5), 1]  = 0.5
                    plan[np.where(plan[:, 1] < -0.5), 1] = -0.5
                    
                    """new_states, derivatives = self.nonlinear_reachability.run_reachability(reachability_state, plan)

                    pose_states = [Zonotope(i.Z[:plan.shape[1], :plan.shape[1] + 1]) for i in new_states[1:]]

                    new_states_rep = np.array(pose_states).reshape((-1, 1))
                    new_states_rep = np.hstack([new_states_rep] * len(obstacles)).flatten()

                    zono_obs_pair = zip(new_states_rep, obstacles * len(new_states[1:]))

                    ret = self.pool.map(check_zono_intersection, zono_obs_pair)
                    ret = np.array(ret, dtype = object)"""
                    #print(j)
                else:
                    break
            #print(ret)
            # when exiting the for loop
            plan[-6:, :] = self.fail_safe
            if(j == 0):
                #print("plan is safe :)")
                action = plan[0, :2]
                plan = np.delete(plan, (0), axis=0)
                plan = np.vstack((plan, np.array([[0] * 8])))
                self.old_plan = plan
                return True, action
            elif(np.any(ret[:, 0])):
                #print("Update failed")
                action = self.old_plan[0, :2]
                self.old_plan = np.delete(self.old_plan, (0), axis=0)
                self.old_plan = np.vstack((self.old_plan, np.array([[0] * 8])))
                return False, action
                #print(plan)
                #print(plan_updates)
            else:
                #print("Update Succedded")
                action = plan[0, :2]
                plan = np.delete(plan, (0), axis=0)
                plan = np.vstack((plan, np.array([[0] * 8])))
                self.old_plan = plan
                return True, action
        else:
            #print("No Obstacles found. Plan is safe")
            action = plan[0, :2]
            plan = np.delete(plan, (0), axis=0)
            plan = np.vstack((plan, np.array([[0] * 8])))
            self.old_plan = plan
            return True, action
                    