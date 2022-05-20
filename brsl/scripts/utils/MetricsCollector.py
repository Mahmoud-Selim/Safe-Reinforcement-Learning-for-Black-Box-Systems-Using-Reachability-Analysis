###################################################################################################
# @Author: Mahmoud-Selim 
# @Date: 2021-09-16 09:53:47 
# @Last Modified by:   Mahmoud-Selim 
# @Last Modified time: 2021-09-16 09:53:47 
###################################################################################################


from tensorboardX import SummaryWriter


class MetricsCollector():
    def __init__(self, safety_dir):
        self.writer = SummaryWriter(safety_dir)
    
    # Each Episode
    def add_collision_rate(self, step, rate):
        self.writer.add_scalar('Safety/collision_rate', rate, step)

    # Each Episode
    def add_goal_rate(self, step, rate):
        self.writer.add_scalar('Safety/Goal_rate', rate, step)

    # Each Episode
    def add_time_to_goal(self, step, time):
        self.writer.add_scalar('Safety/Time_to_goal', time, step)

    # Each Time Step
    def add_safety_interventions(self, step, rate):
        self.writer.add_scalar('Safety/Safey_interventions', rate, step)

    # Each Time Step
    def add_plan_time(self, step, plan_t):
        self.writer.add_scalar('Safety/Planning_time', plan_t, step)

    # Each Time Step
    def add_robot_speed(self, step, speed):
        self.writer.add_scalar('Safety/Robot_speed', speed, step)

    # Each Time Step
    def add_lr(self, step, lr):
        self.writer.add_scalar('Safety/lr', lr, step)



