import rospy
import math
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose2D
from flatland_msgs.msg import Collisions
from flatland_msgs.srv import MoveModel
from nav_msgs.msg import Odometry 
from gym import Env
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import random
import pandas as pd 
from tf.transformations import euler_from_quaternion


class OneWheelchairEnvWithDist(Env):

    def __init__(self):
        #current data
        self.episode = 0
        self.action_n = 0
        self.max_episodes = 200
        self.front_split = 9
        self.back_split = 3
        self.split = self.front_split + self.back_split
        self.lidar_sample = []
        self.collisions = False
        self.min_distance = 0.25
        self.finished = False
        self.action_history = []
        self.rotation_counter = 0
        self.rotations = 0
        self.consecutive_rotations = 0
        self.forward_reward = 0
        self.map = 0
        self.start_points = []
        self.position = (0,0,0)
        self.target_position = (0,0)
        self.previous_diff = (0,0)
        self.task = 'none'

        self.reset_counters()

        self.reset_data()
        
        #ros topics and services
        rospy.init_node('one_wheelchair_env', anonymous=True)

        self.scan_topic = "/static_laser1"   
        self.twist_topic = "/cmd_vel1"
        self.bumper_topic = "/collisions"
        self.odom_topic = "/odom1"
        self.prox_topic = "/prox_laser1"

        rospy.Subscriber(self.scan_topic, LaserScan, self.sample_lidar, buff_size=10000000, queue_size=1)
        rospy.Subscriber(self.bumper_topic, Collisions, self.check_collisions, buff_size=10000000, queue_size=1)
        rospy.Subscriber(self.prox_topic, LaserScan, self.check_finished, buff_size=10000000, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.update_position, buff_size=10000000, queue_size=1)

        rospy.wait_for_service("/move_model")
        self.move_model_service = rospy.ServiceProxy("/move_model", MoveModel)

        self.twist_pub = rospy.Publisher(self.twist_topic, Twist, queue_size=1)

        #learning env
        linear_speed = 0.3
        angular_speed = 1.0471975512 
        self.actions = [(0, 0), 
                        (linear_speed, 0), 
                        (-linear_speed, 0), 
                        (0, angular_speed), 
                        (0, -angular_speed), 
                        (linear_speed, angular_speed), 
                        (linear_speed, -angular_speed)]
        self.action_space = Discrete(len(self.actions))
        
        self.observation_space = Box(0, 2, shape=(1,self.split + 7))
        while len(self.lidar_sample) == 0:pass
        dist,angle = self.diff_from_target()
        self.state = np.array([min(dist, 2),(angle/math.pi) + 1] + self.lidar_sample)
        self.previous_diff = (dist, angle)
    
    def step(self, action):

        self.change_robot_speed(self.actions[action][0], self.actions[action][1])
        self.action_n += 1

        while self.episode == self.action_n: pass

        dist,angle = self.diff_from_target()
        self.state = np.array([min(dist, 2),(angle/math.pi) + 1] + self.lidar_sample)
        
        done = False
        
        if self.action_n < 4:
            self.collisions = False
            self.finished = False

        if self.collisions:
            reward = -200
            done = True
            self.data['end_condition'][-1] = 'collision'
        elif self.finished:
            reward = 400 + ((self.max_episodes - ((self.episode) * 200) / self.max_episodes))
            done = True
            self.data['end_condition'][-1] = 'finished'
        elif self.episode > self.max_episodes:
            reward = -(300 + self.forward_reward) 
            done = True
            self.data['end_condition'][-1] = 'time out'
        elif action == 1:
            reward = 300 / self.max_episodes
            self.forward_reward += reward
        elif action == 5 or action == 6:
            reward = 300 / (self.max_episodes * 2)
            self.forward_reward += reward
        elif action == 0 or action == 2:
            reward = -(300 / self.max_episodes) * 5
        elif action == 3 or action == 4:
            if self.lidar_sample[0] > 0.1: reward = -(300 / self.max_episodes) * 2
            else:reward = 300 / (self.max_episodes * 2)
        else:
            reward = 0

        if len(self.action_history) > 0:
            last_action = self.action_history[-1]
            if (last_action == 1 and action == 2) or (last_action == 2 and action == 1):
                reward -= (300 / self.max_episodes) * 5
            elif (last_action == 3 and action == 4) or (last_action == 4 and action == 3):
                reward -= (300 / self.max_episodes) * 5

        if action == 3 or action == 5: self.rotation_counter += 1 
        elif action == 4 or action == 6: self.rotation_counter -= 1
        if self.rotation_counter % 30 == 0:
            rot = self.rotation_counter / 30
            if self.rotations != rot:
                direction = rot - self.rotations
                if self.consecutive_rotations != 0:
                    if direction != (self.consecutive_rotations / abs(self.consecutive_rotations)):
                        self.consecutive_rotations = 0
                self.consecutive_rotations += direction
                self.rotations = rot
                reward -= 75 * abs(self.consecutive_rotations)
        
        reward += (self.lidar_sample[2] - 0.5) * (600 / self.max_episodes)
        if self.lidar_sample[2] > 0.5: reward += ((self.lidar_sample[3] - 0.5) + (self.lidar_sample[4] - 0.5)) * (300 / self.max_episodes)

        reward += (self.previous_diff[0]-dist) * 50
        reward += (abs(self.previous_diff[1])-abs(angle)) * 10
        
        reward -= dist * ((self.episode / self.max_episodes))
        reward -= (abs(angle) / math.pi) * 3

        self.previous_diff = (dist, angle)


        self.total_steps += 1
        if action in [1,5,6]:self.forward_steps += 1
        if done:
            self.total_episodes += 1
            if self.finished: self.success_episodes += 1

        info = {}

        self.action_history.append(action)

        self.data['actions'][-1].append(action)
        self.data['rewards'][-1].append(reward)
        self.data['positions'][-1].append(self.position)

        return self.state, reward, done, info
    
    def render(self): pass
    
    def reset(self):
        self.lidar_sample = []

        map = self.start_points[self.map]

        x = map[1][0]
        y = map[1][1]
        theta = random.random() * (math.pi * 2)
        rand_shift = (random.random() * 0.6) - 0.3
        if map[1][2] == 'h': x += rand_shift
        else: y += rand_shift
        self.change_robot_position("robot1", x, y, theta)

        target_x = map[2][0]
        target_y = map[2][1]
        rand_shift = (random.random() * 0.6) - 0.3
        if map[2][2] == 'h': target_x += rand_shift
        else: target_y += rand_shift
        self.change_robot_position("prox", target_x, target_y, 0)
        self.target_position = (target_x, target_y)

        self.collisions = False
        self.finished = False
        self.episode = 0
        self.action_n = 0
        while len(self.lidar_sample) == 0:pass
        dist,angle = self.diff_from_target()
        self.state = np.array([min(dist, 2),(angle/math.pi) + 1] + self.lidar_sample)
        self.previous_diff = (dist, angle)
        self.action_history = []
        self.rotation_counter = 0
        self.rotations = 0
        self.consecutive_rotations = 0
        self.forward_reward = 0

        self.map += 1
        if self.map == len(self.start_points): self.map = 0

        self.data['map'].append(map[0])
        self.data['task'].append(self.task)
        self.data['target'].append((target_x,target_y))
        self.data['actions'].append([])
        self.data['rewards'].append([])
        self.data['positions'].append([(x,y,theta)])
        self.data['end_condition'].append('time out')

        return self.state

    def sample_lidar(self,data):
        self.change_robot_speed(0,0)

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
        
        self.lidar_sample.append((data.ranges[math.ceil(len(data.ranges) / 2) - 1]))
        self.lidar_sample.append((data.ranges[(math.ceil(len(data.ranges) / 2) - 1) - (each // 2)]))
        self.lidar_sample.append((data.ranges[(math.ceil(len(data.ranges) / 2) - 1) - (each // 2)]))
        self.lidar_sample.append((data.ranges[math.ceil(len(data.ranges) / 4) - 1]))
        self.lidar_sample.append((data.ranges[math.ceil(len(data.ranges) * (3/4)) - 1]))

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

        self.episode += 1

    def check_collisions(self, data):
        if self.collisions: return
        if len(data.collisions) > 0:
            self.change_robot_speed(0,0)
            self.collisions = True

    def check_finished(self, data):
        if self.finished: return
        values = [x for x in data.ranges if not np.isnan(x)]
        if len(values) > 0:
            if(min(values) < self.min_distance):
                self.change_robot_speed(0,0)
                self.finished = True

    def update_position(self, data):
        rotation = euler_from_quaternion([data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w])[2]
        self.position = (data.pose.pose.position.x, data.pose.pose.position.y, rotation)

    def change_robot_speed(self, linear, angular):
        twist_msg = Twist()
        twist_msg.linear.x = linear
        twist_msg.angular.z = angular
        self.twist_pub.publish(twist_msg)

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

    def reset_data(self):
        self.data = {
            'map':[],
            'task':[],
            'target':[],
            'actions':[],
            'rewards':[],
            'positions':[],
            'end_condition':[]
        }

    def diff_from_target(self):
        dist = math.sqrt(pow(self.target_position[0] - self.position[0], 2) + pow(self.target_position[1] - self.position[1], 2))
        angle = math.atan2(self.target_position[1] - self.position[1], self.target_position[0] - self.position[0])
        angle = (self.position[2] % (math.pi * 2)) - angle
        if abs(angle) > math.pi:
            if angle > 0: angle = -((math.pi * 2) - angle)
            else: angle = (math.pi * 2) - angle
        return dist, angle
    
    def dump_data(self, filename):
        df = pd.DataFrame(self.data)
        df.to_csv(filename)


