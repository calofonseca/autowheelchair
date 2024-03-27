import rospy
import math
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose2D
from flatland_msgs.msg import Collisions
from flatland_msgs.srv import MoveModel
from nav_msgs.msg import Odometry 
from gym import Env
from gym.spaces import Discrete, Box, Tuple, MultiDiscrete
import numpy as np
import random
import pandas as pd 
from tf.transformations import euler_from_quaternion
import time
import math

class TwoWheelChairEnvLessActions(Env):

    def __init__(self):
        #current data
        self.distance1 = None
        self.distance2 = None
        self.previous_distance1 = None
        self.previous_distance2 = None

        self.episode = 0
        self.naction1 = 0
        self.naction2 = 0
        self.max_episodes = 300
        self.front_split = 9
        self.back_split = 3
        self.split = self.front_split + self.back_split
        
        self.lidar_sample = []
        self.lidar_sample2 = []
        
        self.collisions = False
        self.min_distance = 0.4
        
        self.finished = False
        self.end_reached = False
        self.finished2 = False
        self.end_reached2 = False
        
        self.action_history = []
        self.rotation_counter = 0
        self.rotations = 0
        self.consecutive_rotations = 0

        self.action_history2= []
        self.rotation_counter2 = 0
        self.rotations2 = 0
        self.consecutive_rotations2 = 0

        self.forward_reward = 0
        
        self.map = 0
        self.start_points = []
        self.position = (0,0,0)
        self.position2 = (0,0,0)

        self.task = 'none'

        self.reset_counters()

        self.reset_data()
        
        #ros topics and services
        rospy.init_node('two_wheelchair_env', anonymous=True)

        self.scan_topic = "/static_laser1"   
        self.twist_topic = "/cmd_vel1"
        self.bumper_topic = "/collisions"
        self.odom_topic = "/odom1"
        self.prox_topic = "/prox_laser1"
        
        self.scan_topic2 = "/static_laser2"   
        self.twist_topic2 = "/cmd_vel2"
        self.odom_topic2 = "/odom2"
        self.prox_topic2 = "/prox_laser2"


        rospy.Subscriber(self.scan_topic, LaserScan, self.sample_lidar, buff_size=10000000, queue_size=1)
        rospy.Subscriber(self.bumper_topic, Collisions, self.check_collisions, buff_size=10000000, queue_size=1)
        rospy.Subscriber(self.prox_topic, LaserScan, self.check_finished, buff_size=10000000, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.update_position, buff_size=10000000, queue_size=1)
        
        rospy.Subscriber(self.scan_topic2, LaserScan, self.sample_lidar2, buff_size=10000000, queue_size=1)
        rospy.Subscriber(self.prox_topic2, LaserScan, self.check_finished2, buff_size=10000000, queue_size=1)
        rospy.Subscriber(self.odom_topic2, Odometry, self.update_position2, buff_size=10000000, queue_size=1)

        rospy.wait_for_service("/move_model")
        self.move_model_service = rospy.ServiceProxy("/move_model", MoveModel)

        self.twist_pub = rospy.Publisher(self.twist_topic, Twist, queue_size=1)
        self.twist_pub2 = rospy.Publisher(self.twist_topic2, Twist, queue_size=1)

        
        #learning env
        self.fixed_linear_speed = 0.3

        self.observation_space = Box(0, 2, shape=(1,(self.split + 7) * 2))
        
        
        while len(self.lidar_sample) != 19  and len(self.lidar_sample2) != 19 :pass
        self.state = np.array(self.lidar_sample + self.lidar_sample2)

        # Assuming these are your maximum and minimum angular velocities
        max_angular_speed = 1.5  # Example max angular speed
        min_angular_speed = -1.5 # Example min angular speed

        # Continuous action space for angular velocity
        self.action_space = Box(low=np.array([min_angular_speed]), high=np.array([max_angular_speed]), dtype=np.float32)

    def change_robot_speed(self, robot, angular):
        """
        Apply the given action to the robot.
        :param action: The action to apply. action[0] contains the angular velocity.
        """
        twist_msg = Twist()
        
        linear = self.fixed_linear_speed # Fixed linear speed 
        twist_msg.linear.x = linear
        twist_msg.angular.z = angular

        if(robot == 1):
            self.twist_pub.publish(twist_msg)
        if(robot == 2):
            self.twist_pub2.publish(twist_msg)

    def step(self, a1, a2):
        """
        Steps the environment to the next time step. It applies actions and calculates rewards
        :param a1: The action to apply for robot 1.
        :param a2: The action to apply for robot 2.
        """
        
        while not (self.naction1 == self.episode + 1 and self.naction2 == self.episode + 1):
            # Optionally, include a small delay to prevent the loop from consuming too much CPU
            time.sleep(0.0001)  
        
        reward=0
        reward1 = 0
        reward2 = 0

        #Apply Actions
        self.change_robot_speed(1, a1)
        self.change_robot_speed(2, a2)

        self.state = np.array(self.lidar_sample + self.lidar_sample2)
        
        done = False
        
        if self.episode < 4:
            self.collisions = False
            self.finished = False
            self.finished2 = False

        if self.end_reached != self.finished:
            if self.end_reached: exit_end = True
            else: enter_end = True
            self.end_reached = self.finished

        if self.end_reached2 != self.finished2:
            if self.end_reached2: exit_end = True
            else: enter_end = True
            self.end_reached2 = self.finished2


        if self.collisions:
            #reward = -400
            #Changed for rewrad or penalizations based on distance to the wall
            print("COLIDED")
            done = True
            self.data['end_condition'][-1] = 'collision'
        elif self.end_reached and self.end_reached2:
            reward = 800 + ((self.max_episodes - ((self.episode) * 200) / self.max_episodes))
            #Changed for reward based on proximity of target
            done = True
            self.data['end_condition'][-1] = 'finished'

        #TODO    
        elif self.episode > self.max_episodes:
            reward = -(600 + self.forward_reward) 
            print("EPISODES GREATER THAN")
            done = True
            #self.data['end_condition'][-1] = 'time out'
            print("GREATERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
        else:  
            reward = 0

        #if enter_end: reward += 100
        #if exit_end: reward -= 100

        #REWARD CALCULATION
        #Penalizing Actions
        reward1 += self.optimized_reward_function(a1, 1)
        reward2 += self.optimized_reward_function(a2, 2)
        
        #Penalizing repetitive Changes in direction #Might disapeer
        #reward1 += self.apply_penalty_direction_changes(self.action_history,a1)
        #reward2 += self.apply_penalty_direction_changes(self.action_history2,a2)   

        #Penalizing Repetitive Actions #Might disapeer
        #reward1 += self.apply_penalty_if_repetitive(self.action_history,a1,reward)
        #reward2 += self.apply_penalty_if_repetitive(self.action_history2,a2,reward)

        self.total_steps += 1
        if a1 == 1:self.forward_steps += 0.5
        if a2 == 1:self.forward_steps += 0.5

        if done:
            self.total_episodes += 1
            if self.end_reached and self.end_reached2: self.success_episodes += 1

        info = {}

        self.action_history.append(a1)
        self.action_history2.append(a2)

        self.data['actions'][-1].append(action)
        self.data['rewards'][-1].append(reward)
        self.data['positions'][-1].append((self.position, self.position2))


        # Split the state for each agent based on your state formation
        observations = [self.lidar_sample, self.lidar_sample2]  # Separate observations for each agent
        rewards = [reward + reward1, reward + reward2]  # If shared rewards, otherwise calculate individually
        dones = done  # Both agents likely share the same done flag in your scenario
        infos = [{}, {}]  # Additional info if any, per agent
        self.episode += 1
        return observations, rewards, dones, infos      

    
    def apply_penalty_direction_changes(self, action_history, a_current):
        #Penalizing Rapid Direction Changes
        reward = 0 
        if len(action_history) > 0:
            last_action = action_history[-1]
            if (last_action == 2 and a_current == 3) or (last_action == 3 and a_current == 2):
                reward -= (300 / self.max_episodes) * 5

        # Return the modified reward
        return reward

    def optimized_reward_function(self, action, chair):
        # Constants
        # Constants
        base_reward = 10  # Constant base reward
        penalty_scale = 2  # Adjusted for quadratic penalty
        progress_reward_scale = 1  # Balanced scale for rewarding progress towards the goal
        stop_penalty_base = -1  # Base penalty for stopping
        significant_goal_reward = 50  # Significant reward for reaching the goal
        safe_distance = 0.40  # Distance considered safe from obstacles
        safe_distance_sides = 0.30 # Distance considered safe from obstacles
        turn_penalty = -10  # Penalty for unnecessary turning
        turn_reward = 5  # Reward for wisely turning away from an obstacle

        # Check terminal state
        end_condition = (chair == 1 and self.end_reached) or (chair == 2 and self.end_reached2)
        if action == 0:  # Success
            if end_condition: 
                return significant_goal_reward  # Reward significantly for reaching the goal
            else: 
                return stop_penalty_base  # Apply stop penalty

        # Initialize reward
        reward = base_reward

        # Get lidar samples for obstacle detection
        lidar = self.lidar_sample if chair == 1 else self.lidar_sample2
        front_obstacles = [lidar[i] for i in range(12, 14)]
        left_obstacles = [lidar[i] for i in range(9, 11)]
        right_obstacles = [lidar[i] for i in range(15, 17)]

        # Determine if there are obstacles in front, left, and right
        front_clear = all(d > safe_distance for d in front_obstacles)
        left_clear = all(d > safe_distance_sides for d in left_obstacles)
        right_clear = all(d > safe_distance_sides for d in right_obstacles)

        # Handle turning actions
        if action == 2:  # Turning left
            if front_clear:
                reward += turn_penalty  # Penalize if turning left unnecessarily
            elif not front_clear and left_clear:
                reward += turn_reward  # Reward if turning away from an obstacle
        elif action == 3:  # Turning right
            if front_clear:
                reward += turn_penalty  # Penalize if turning right unnecessarily
            elif not front_clear and right_clear:
                reward += turn_reward  # Reward if turning away from an obstacle on the right

        # Penalize moving forward when an obstacle is directly in front
        if action == 1 and not front_clear:
            reward += turn_penalty

        # Quadratic penalty for being too close to an obstacle, considering all relevant directions
        min_distances = front_obstacles + left_obstacles + right_obstacles
        min_distance_to_obstacle = min(min_distances)
        if min_distance_to_obstacle < safe_distance:
            penalty = (penalty_scale * (1 - min_distance_to_obstacle / safe_distance)) ** 2
            reward -= penalty


        # Reward or penalize based on progress towards the goal
        current_distance = self.distance1 if chair == 1 else self.distance2
        previous_distance = self.previous_distance1 if chair == 1 else self.previous_distance2
        if previous_distance is not None and current_distance is not None:

            distance_improvement = previous_distance - current_distance  # Positive if closer to the goal

            # Adjust reward or penalty for progress
            if distance_improvement > 0:
                reward += progress_reward_scale * distance_improvement * 100
            else:
                reward += progress_reward_scale * distance_improvement * 50  # Less punitive for negative progress

        # Update previous distance for next step comparison
        if chair == 1:
            self.previous_distance1 = current_distance
        else:
            self.previous_distance2 = current_distance

        return reward


    def render(self): pass
    
    def reset(self):
        self.lidar_sample = []
        self.lidar_sample2 = []

        map = self.start_points[self.map]

        x = map[1][0]
        y = map[1][1]

        x2 = map[1][0]
        y2 = map[1][1]
        theta = random.random() * (math.pi * 2)
       
        if map[1][2] == 'h': 
            x -= 0.2
            x2 += 0.2
            theta = math.pi / 2
        else: 
            y -= 0.2
            y2 += 0.2
            theta = math.pi

        self.change_robot_position("robot1", x, y, theta)
        self.change_robot_position("robot2", x2, y2, theta)

        target_x = map[2][0]
        target_y = map[2][1]
        self.change_robot_position("prox", target_x, target_y, 0)

        self.collisions = False
        self.finished = False
        self.finished2 = False
        self.end_reached = False
        self.end_reached2 = False
        self.episode = 0
        self.naction1 = 0
        self.naction2 = 0
        self.previous_distance1 = None
        self.previous_distance2 = None
        self.distance1 = None
        self.distance2 = None

        while len(self.lidar_sample) != 19 and len(self.lidar_sample2) != 19:pass

        self.state =np.array(self.lidar_sample + self.lidar_sample2)

        self.action_history = []
        self.rotation_counter = 0
        self.rotations = 0
        self.consecutive_rotations = 0

        self.action_history2 = []
        self.rotation_counter2 = 0
        self.rotations2 = 0
        self.consecutive_rotations2 = 0

        self.forward_reward = 0

        self.map += 1
        if self.map == len(self.start_points): self.map = 0

        self.data['map'].append(map[0])
        self.data['task'].append(self.task)
        self.data['target'].append((target_x,target_y))
        self.data['actions'].append([])
        self.data['rewards'].append([])
        self.data['positions'].append([(x,y,theta)])
        self.data['adjacency'].append([])
        self.data['end_condition'].append('time out')
   
        return [self.lidar_sample, self.lidar_sample2]

    def sample_lidar(self,data): #LEFT SIDE ROBOT

        if self.naction1 != self.episode: return
        
        # Check if enough time has passed since the last execution
        current_time = time.time()
        if hasattr(self, 'last_execution_time'):
            elapsed_time = current_time - self.last_execution_time
            if elapsed_time < 0.002:  # Less than 2 seconds have passed
                return
        else:
            self.last_execution_time = current_time  # Initialize if not set

        self.lidar_sample = []

        front_lasers = math.ceil(len(data.ranges) / 2)
        each = front_lasers // (self.front_split - 1)
        front_lasers += each
        front_dist = [each for _ in range(self.front_split)]
        back = False
        for i in range(len(data.ranges) % self.split): 
            if back: front_dist[self.front_split - ((i//2)+1)] += 1
            else: front_dist[i//2] += 1
            back = not back
        
        self.lidar_sample.append((data.ranges[math.ceil(len(data.ranges) / 2) - 1])) #0
        self.lidar_sample.append((data.ranges[(math.ceil(len(data.ranges) / 2) - 1) - (each // 2)])) #1
        self.lidar_sample.append((data.ranges[(math.ceil(len(data.ranges) / 2) - 1) + (each // 2)])) #2
        self.lidar_sample.append((data.ranges[math.ceil(len(data.ranges) / 4) - 1])) #3
        self.lidar_sample.append((data.ranges[math.ceil(len(data.ranges) * (3/4)) - 1])) #4
        self.lidar_sample.append((data.ranges[(math.ceil(len(data.ranges) / 4) - 1) - (each // 2)])) #5
        self.lidar_sample.append((data.ranges[(math.ceil(len(data.ranges) / 4) - 1) + (each // 2)])) #6


        back_lasers = len(data.ranges) - front_lasers
        back_dist = [(back_lasers // self.back_split) for _ in range(self.back_split)]
        for i in range(back_lasers % self.back_split): back_dist[i] += 1

        dist = back_dist[:math.ceil(self.back_split/2)] + front_dist + back_dist[math.ceil(self.back_split/2):]

        if self.back_split % 2 == 0:
            min_range =  0
            max_range = dist[0]
            self.lidar_sample.append(min(data.ranges[min_range:max_range]))
        else:
            min_range =  len(data.ranges) - (dist[0] // 2)
            max_range = (dist[0] // 2) + (dist[0] % 2)
            self.lidar_sample.append(min(data.ranges[min_range:len(data.ranges)] + data.ranges[0:max_range]))
        
        for i in range(1, self.split):
            min_range = max_range
            max_range += dist[i]
            self.lidar_sample.append(min(data.ranges[min_range:max_range]))

        for i in range (len(self.lidar_sample)): self.lidar_sample[i] = min(self.lidar_sample[i], 2)

        #print(len(data.ranges))
        #print("ROBOT 1")
        # Calculate indices for #5 and #6 based on your setup
        start_index = math.ceil(len(data.ranges) / 4) - 1 - (each // 2*4) 
        end_index = math.ceil(len(data.ranges) / 4) - 1 + (each // 2*4) 

        front_distance, back_distance, front_edge_index, back_edge_index = self.find_object_front_and_back(data.ranges, start_index, end_index, 1)
        self.lidar_sample2[5] = back_distance
        self.lidar_sample2[6] = front_distance

        self.naction1 += 1
        self.last_execution_time = current_time  # Update the last execution time


    def sample_lidar2(self,data):

        if self.naction2 != self.episode: return

        # Check if enough time has passed since the last execution
        current_time2 = time.time()
        if hasattr(self, 'last_execution_time2'):
            elapsed_time2 = current_time2 - self.last_execution_time2
            if elapsed_time2 < 0.002:  # Less than 2 seconds have passed
                return
        else:
            self.last_execution_time2 = current_time2  # Initialize if not set

        self.lidar_sample2 = []

        front_lasers = math.ceil(len(data.ranges) / 2)
        each = front_lasers // (self.front_split - 1)
        front_lasers += each
        front_dist = [each for _ in range(self.front_split)]
        back = False
        for i in range(len(data.ranges) % self.split): 
            if back: front_dist[self.front_split - ((i//2)+1)] += 1
            else: front_dist[i//2] += 1
            back = not back
        
        self.lidar_sample2.append((data.ranges[math.ceil(len(data.ranges) / 2) - 1]))
        self.lidar_sample2.append((data.ranges[(math.ceil(len(data.ranges) / 2) - 1) - (each // 2)]))
        self.lidar_sample2.append((data.ranges[(math.ceil(len(data.ranges) / 2) - 1) - (each // 2)]))
        self.lidar_sample2.append((data.ranges[math.ceil(len(data.ranges) / 4) - 1]))
        self.lidar_sample2.append((data.ranges[math.ceil(len(data.ranges) * (3/4)) - 1]))
        self.lidar_sample2.append((data.ranges[(math.ceil(len(data.ranges) * (3/4)) - 1) + (each // 2)]))
        self.lidar_sample2.append((data.ranges[(math.ceil(len(data.ranges) * (3/4)) - 1) - (each // 2)]))
   
        back_lasers = len(data.ranges) - front_lasers
        back_dist = [(back_lasers // self.back_split) for _ in range(self.back_split)]
        for i in range(back_lasers % self.back_split): back_dist[i] += 1

        dist = back_dist[:math.ceil(self.back_split/2)] + front_dist + back_dist[math.ceil(self.back_split/2):]

        if self.back_split % 2 == 0:
            min_range =  0
            max_range = dist[0]
            self.lidar_sample2.append(min(data.ranges[min_range:max_range]))
        else:
            min_range =  len(data.ranges) - (dist[0] // 2)
            max_range = (dist[0] // 2) + (dist[0] % 2)
            self.lidar_sample2.append(min(data.ranges[min_range:len(data.ranges)] + data.ranges[0:max_range]))
        
        for i in range(1, self.split):
            min_range = max_range
            max_range += dist[i]
            self.lidar_sample2.append(min(data.ranges[min_range:max_range]))

        for i in range (len(self.lidar_sample2)): self.lidar_sample2[i] = min(self.lidar_sample2[i], 2)
        
        #print("ROBOT 2")
        # Calculate indices for #5 and #6 based on your setup
        end_index = math.ceil(len(data.ranges) * (3/4)) - 1 + (each // 2*4) 
        start_index = math.ceil(len(data.ranges) * (3/4)) - 1 - (each // 2*4)

        front_distance, back_distance, front_edge_index, back_edge_index = self.find_object_front_and_back(data.ranges, start_index, end_index, 2)
        #print(f"2 - Object front side distance: {front_distance}, back side distance: {back_distance}")
        self.lidar_sample2[5] = back_distance
        self.lidar_sample2[6] = front_distance

        self.naction2 += 1
        self.last_execution_time2 = current_time2  # Update the last execution time


    def find_object_front_and_back(self, data_ranges, start_index, end_index, side):
        # Find the closest point in the segment, which is part of the object
        min_distance = min(data_ranges[start_index:end_index+1])
        closest_point_index = data_ranges.index(min_distance, start_index, end_index+1)

        # Initialize indices for front and back edges
        front_edge_index = closest_point_index
        back_edge_index = closest_point_index

        # Scan backward (towards start_index) to find where distance starts increasing (front edge)
        for i in range(closest_point_index, start_index-1, -1):  # Note: step is -1 to move backwards
            if data_ranges[i] > min_distance * 1.4:  # Arbitrary factor to indicate significant increase
                front_edge_index = i + 1
                break

        # Scan forward (towards end_index) to find where distance starts increasing (back edge)
        for i in range(closest_point_index, end_index+1):
            if data_ranges[i] > min_distance * 1.4:  # Same arbitrary factor as above
                back_edge_index = i - 1
                break
                
        # Verification step: Check if no significant increase is detected
        if front_edge_index == back_edge_index == closest_point_index:
            # No significant increase detected, return -1 for both distances and indices
            return -1, -1, -1, -1

        # Return distances at front and back edges
        front_distance = data_ranges[front_edge_index]
        back_distance = data_ranges[back_edge_index]

        #if side 1, means robot1, so return according to inverted logic of the right side
        if side == 1:
            return back_distance, front_distance, back_edge_index, front_edge_index
        elif side == 2:
            return front_distance, back_distance, front_edge_index, back_edge_index


    def check_collisions(self, data):
        if self.collisions: return
        if len(data.collisions) > 0:
            self.change_robot_speed(1,0,0)
            self.change_robot_speed(2,0,0)

            self.collisions = True

    def check_finished(self, data):
        values = [x for x in data.ranges if not np.isnan(x)]
        if len(values) > 0:
            self.distance1= min(values)
            if self.previous_distance1 is None:
                self.previous_distance1 = min(values)
            self.finished = min(values) < self.min_distance


    def check_finished2(self, data):
        values = [x for x in data.ranges if not np.isnan(x)]
        if len(values) > 0:
            self.distance2= min(values)
            if self.previous_distance2 is None:
                self.previous_distance2 = min(values)
            self.finished2 = min(values) < self.min_distance
                
    def update_position(self, data):
        rotation = euler_from_quaternion([data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w])[2]
        self.position = (data.pose.pose.position.x, data.pose.pose.position.y, rotation)

    def update_position2(self, data):
        rotation = euler_from_quaternion([data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w])[2]
        self.position2 = (data.pose.pose.position.x, data.pose.pose.position.y, rotation)
    






    def reset_robots(self):
        self.change_robot_position('robot1', 11.25, 4.5, 1.57079632679)
        self.change_robot_position('robot2', 11.25, 4.2, 1.57079632679)
        self.change_robot_position('prox', 11.25, 4.8, 0)

    def change_robot_position(self, name, x, y, theta):
        pose = Pose2D()
        pose.x = x
        pose.y = y
        pose.theta = theta
        self.move_model_service(name = name, pose = pose)

    def reset_counters(self):
        self.total_episodes = 0
        self.success_episodes = 0
        self.total_steps = 0
        self.forward_steps = 0
        self.adj_steps = 0

    def reset_data(self):
        self.data = {
            'map':[],
            'task':[],
            'target':[],
            'actions':[],
            'rewards':[],
            'positions':[],
            'adjacency':[],
            'end_condition':[]
        }
    
    def dump_data(self, filename):
        df = pd.DataFrame(self.data)
        df.to_csv(filename)


