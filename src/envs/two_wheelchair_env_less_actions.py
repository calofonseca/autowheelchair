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


class TwoWheelChairEnvLessActions(Env):

    def __init__(self):
        #current data
        self.initial_distance1 = None
        self.initial_distance2 = None
        self.episode = 0
        self.action_n = 0
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
        linear_speed = 0.3
        angular_speed = 1.0471975512 
        self.actions = [(0, 0), 
                        (linear_speed, 0),  
                        (0, angular_speed), 
                        (0, -angular_speed)]
                        
        n_actions = (len(self.actions) , len(self.actions))
                        
                        
        self.action_space = Discrete(np.prod(n_actions))


        self.observation_space = Box(0, 2, shape=(1,(self.split + 7) * 2))
        
        
        while len(self.lidar_sample) != 19  or len(self.lidar_sample2) != 19 :pass
        self.state = np.array(self.lidar_sample + self.lidar_sample2)

    def step(self, action, action2=-1):
        
        if action2 != -1: #Multi agent aaproach
            a1 = action
            a2 = action2
        else:
            a1 = action // 4
            a2 = action % 4
        
        reward=0
        reward1 = 0
        reward2 = 0

        self.change_robot_speed(1, self.actions[a1][0], self.actions[a1][1])
        self.change_robot_speed(2, self.actions[a2][0], self.actions[a2][1])

        self.action_n += 1

        while self.episode == self.action_n: pass

        self.state = np.array(self.lidar_sample + self.lidar_sample2)
        
        done = False
        
        if self.action_n < 4:
            self.collisions = False
            self.finished = False
            self.finished2 = False

        enter_end = False
        exit_end = False

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
            #reward = 800 + ((self.max_episodes - ((self.episode) * 200) / self.max_episodes))
            #Changed for reward based on proximity of target
            done = True
            self.data['end_condition'][-1] = 'finished'

        #TODO    
        elif self.episode > self.max_episodes:
            reward = -(600 + self.forward_reward) 
            print("EPISODES GREATER THAN")
            done = True
            self.data['end_condition'][-1] = 'time out'
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
        reward1 += self.apply_penalty_direction_changes(self.action_history,a1)
        reward2 += self.apply_penalty_direction_changes(self.action_history2,a2)   

        #Penalizing Repetitive Actions #Might disapeer
        reward1 = self.apply_penalty_if_repetitive(self.action_history,a1,reward)
        reward2 = self.apply_penalty_if_repetitive(self.action_history2,a2,reward)


        #Rewarding Maintaining a Specific Formation (TO IMPROVE)
        if max([self.lidar_sample[5], self.lidar_sample[6], self.lidar_sample2[5], self.lidar_sample2[6]]) <= 0.4 and abs(self.lidar_sample[5] - self.lidar_sample2[5]) <= 0.05 and abs(self.lidar_sample[6] - self.lidar_sample2[6]) <= 0.05:
            reward += 10
            self.data['adjacency'][-1].append(1)
            self.adj_steps += 1
        else: self.data['adjacency'][-1].append(0)

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

        # Modify the return statement to accommodate both single and multi-agent scenarios
        if action2 != -1:  # Multi-agent setting
            # Split the state for each agent based on your state formation
            observations = [self.lidar_sample, self.lidar_sample2]  # Separate observations for each agent
            rewards = [reward + reward1, reward + reward2]  # If shared rewards, otherwise calculate individually
            dones = done  # Both agents likely share the same done flag in your scenario
            infos = [{}, {}]  # Additional info if any, per agent
            return observations, rewards, dones, infos
        else:  # Single-agent setting
            # For single-agent, return the entire state and single values for reward and done
            return self.state, reward, done, {}  # Single values as usual
        



    def apply_penalty_if_repetitive(self, action_history, a_current, reward):
        # Ensure there are at least 5 actions to examine
        if len(action_history) >= 5:
            # Look at the last 5 actions, including the current action
            last_five_actions = action_history[-4:] + [a_current]

            # Check if all the last 5 actions are the same and not 0
            if len(set(last_five_actions)) == 1 and (last_five_actions[0] != 0) and last_five_actions[0]!=1 :
                # Apply the penalty
                reward -= (2000 / self.max_episodes) * 5

        # Append the current action to the history for future checks
        action_history.append(a_current)

        # Return the modified reward
        return reward
    
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
        base_reward = 300 / self.max_episodes
        penalty_scale = 2  # Adjust to scale the penalty with distance

        # Helper functions
        def penalty_for_proximity(distance):
            """Calculate penalty based on proximity to the wall, with steeper penalty as distance decreases."""
            return (300 / self.max_episodes) * penalty_scale * (1 / max(distance, 0.01))**2

        # Check terminal state
        end_condition = (chair == 1 and self.end_reached) or (chair == 2 and self.end_reached2)
        if end_condition:
            if action == 0: return base_reward  # Success
            else: return -base_reward  # Wrong action at terminal state

        # Get lidar samples
        lidar = self.lidar_sample if chair == 1 else self.lidar_sample2
    
        # Initialize reward
        reward = 0

        # For forward action, consider specified lidar samples
        if action == 1:
            # Consider samples [14], [13], and [12] for evaluating forward movement
            front_slightly_left = lidar[14]
            front = lidar[13]
            front_slightly_right = lidar[12]
        
            # Use the minimum distance from these samples to determine how close the agent is to an obstacle
            min_distance_forward = min(front_slightly_left, front, front_slightly_right)
            if min_distance_forward < 0.45:  # Close to a wall
                reward = -penalty_for_proximity(min_distance_forward)
            else:
                reward = base_reward

        # Stop action
        elif action == 0:
            reward = -base_reward  # Modify to make less severe if needed

        # Turning actions
        elif action in [2, 3]:
            # Left turning action
            if action == 2:
                # Consider samples [15] to [18] for left side
                left_side_distances = [lidar[i] for i in range(15, 19)]
            # Right turning action
            else:
                # Consider samples [8] to [11] for right side
                right_side_distances = [lidar[i] for i in range(8, 12)]

            # Determine direction and penalize or reward based on proximity and action
            if action == 2:  # Turning left
                min_distance_left = min(left_side_distances)
                if min_distance_left < 0.45:
                    reward = -penalty_for_proximity(min_distance_left)
                else:
                    reward = base_reward
            elif action == 3:  # Turning right
                min_distance_right = min(right_side_distances)
                if min_distance_right < 0.45:
                    reward = -penalty_for_proximity(min_distance_right)
                else:
                    reward = base_reward

        print(f"Reward optimized {reward}")
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
        self.action_n = 0
        while len(self.lidar_sample) != 19 or len(self.lidar_sample2) != 19:pass

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
        self.change_robot_speed(1,0,0)

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

        print(len(data.ranges))
        print("ROBOT 1")
        # Calculate indices for #5 and #6 based on your setup
        start_index = math.ceil(len(data.ranges) / 4) - 1 - (each // 2*3) #18
        end_index = math.ceil(len(data.ranges) / 4) - 1 + (each // 2*3) #26
        print(start_index)
        print(end_index)

        front_distance, back_distance, front_edge_index, back_edge_index = self.find_object_front_and_back(data.ranges, start_index, end_index, 1)
        print(f"Object front side distance: {front_distance}, back side distance: {back_distance}")
        print(f"Detected at indices: front edge - {front_edge_index}, back edge - {back_edge_index}")
        self.lidar_sample2[5] = back_distance
        self.lidar_sample2[6] = front_distance

        self.episode += 0.5

    def sample_lidar2(self,data):
        self.change_robot_speed(2,0,0)

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
        
        print("ROBOT 2")
        # Calculate indices for #5 and #6 based on your setup
        end_index = math.ceil(len(data.ranges) * (3/4)) - 1 + (each // 2*3) #72
        start_index = math.ceil(len(data.ranges) * (3/4)) - 1 - (each // 2*3) #64
        print(start_index)
        print(end_index)

        front_distance, back_distance, front_edge_index, back_edge_index = self.find_object_front_and_back(data.ranges, start_index, end_index, 2)
        print(f"Object front side distance: {front_distance}, back side distance: {back_distance}")
        print(f"Detected at indices: front edge - {front_edge_index}, back edge - {back_edge_index}")
        self.lidar_sample2[5] = back_distance
        self.lidar_sample2[6] = front_distance

        self.episode += 0.5


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
            self.finished = min(values) < self.min_distance


    def check_finished2(self, data):
        values = [x for x in data.ranges if not np.isnan(x)]
        if len(values) > 0:
            self.distance2= min(values)
            self.finished2 = min(values) < self.min_distance
                
    def update_position(self, data):
        rotation = euler_from_quaternion([data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w])[2]
        self.position = (data.pose.pose.position.x, data.pose.pose.position.y, rotation)

    def update_position2(self, data):
        rotation = euler_from_quaternion([data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w])[2]
        self.position2 = (data.pose.pose.position.x, data.pose.pose.position.y, rotation)

    def change_robot_speed(self, robot, linear, angular):
        twist_msg = Twist()
        twist_msg.linear.x = linear
        twist_msg.angular.z = angular

        if(robot == 1):
            self.twist_pub.publish(twist_msg)
        if(robot == 2):
            self.twist_pub2.publish(twist_msg)

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


