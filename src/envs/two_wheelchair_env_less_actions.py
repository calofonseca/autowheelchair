import rospy
import math
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose2D
from flatland_msgs.msg import Collisions
from flatland_msgs.srv import MoveModel
import matplotlib.pyplot as plt
import numpy as np
from nav_msgs.msg import Odometry 
from gym import Env
from gym.spaces import Discrete, Box, Tuple, MultiDiscrete
import numpy as np
import random
import pandas as pd 
from tf.transformations import euler_from_quaternion
import time
import math
import rospy
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Header, ColorRGBA

class TwoWheelChairEnvLessActions(Env):

    def __init__(self):

        self.previousa1 = 0
        self.previousa2 = 0
        self.action_duration = 0.2

        #current data
        self.distance1 = None
        self.distance2 = None
        self.previous_distance1 = None
        self.previous_distance2 = None

        self.episode = 0
        self.naction1 = False
        self.naction2 = False
        self.max_episodes = 300
        self.front_split = 9
        self.back_split = 3
        self.split = self.front_split + self.back_split

        self.step_number = 0
        
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

        self.points_pub = rospy.Publisher('highlighted_points', Marker, queue_size=10)

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

        # Define observation space for each robot
        self.num_observations = 9  # Number of LiDAR samples + side robot info
        self.min_range = 0  # Minimum range measured by LiDAR
        self.max_range = 2  # Maximum range (set based on your LiDAR specs)

        # Create observation spaces
        self.robot1_observation_space_box = Box(low=self.min_range, high=self.max_range, shape=(self.num_observations,), dtype=np.float32)
        self.robot2_observation_space_box = Box(low=self.min_range, high=self.max_range, shape=(self.num_observations,), dtype=np.float32)
        
        # Assuming these are your maximum and minimum angular velocities
        self.max_angular_speed = 1.5  # Example max angular speed
        self.min_angular_speed = -1.5 # Example min angular speed

        # Assuming these are your maximum and minimum linear velocities
        self.max_linear_speed = 0.1 
        self.min_linear_speed = 0.6 

        # Define the action spaces for each robot
        self.robot1_action_space_box = Box(
            low=np.array([self.min_angular_speed, self.min_linear_speed]), 
            high=np.array([self.max_angular_speed, self.max_linear_speed]), 
            dtype=np.float32
        )
        self.robot2_action_space_box = Box(
            low=np.array([self.min_angular_speed, self.min_linear_speed]), 
            high=np.array([self.max_angular_speed, self.max_linear_speed]), 
            dtype=np.float32
        )

        self.robot1_observation_space = []
        self.robot2_observation_space = []
        self.robot1_action_space = [0,0]
        self.robot2_action_space = [0,0]

        # Wait for observations to fill in (lidar samples + previous actions + calculated features)
        while len(self.robot1_observation_space) != self.num_observations and len(self.robot2_observation_space) != self.num_observations:
            pass
        self.state = [self.robot1_observation_space, self.robot2_observation_space]



        self.robot1_observation_space = []
        self.robot2_observation_space = []
        self.robot1_action_space = 0
        self.robot2_action_space = 0

        #Wait for observations to fill in (lidar samples + previous actions + calculated features)
        while len(self.robot1_observation_space) != self.num_observations  and len(self.robot2_observation_space) != self.num_observations :pass
        self.state = [self.robot1_observation_space, self.robot2_observation_space]


    def change_robot_speed(self, robot, linear=None, angular= None):
        """
        Apply the given action to the robot.
        :param action: The action to apply.
        """
        twist_msg = Twist()
        
        if linear is None:
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
        print()
        print()
        print(f"---------Episode {self.total_episodes} - Step {self.step_number} - (Total Steps {self.total_steps})--------")
        print(f"Applying Actions: Robot 1 {a1}; Robot 2 {a2}")
        #Apply Actions
        #Actions only reflet the changes in angular speed so the sum of the previous and new is made
        self.change_robot_speed(1, a1[0], a1[1])
        self.change_robot_speed(2, a2[0], a2[1])

        time.sleep(self.action_duration)

        # Stop the robots to gather data and calculate rewards
        self.change_robot_speed(1, 0, 0)
        self.change_robot_speed(2, 0, 0)

        #Change variable to indicate that the lidar need to get the latest observations
        self.naction1=False
        self.naction2=False
        while not (self.naction1 and self.naction2):
            # Optionally, include a small delay to prevent the loop from consuming too much CPU
            time.sleep(0.0001)  
       
        reward=0
        reward1 = 0
        reward2 = 0

        self.state = [self.robot1_observation_space, self.robot2_observation_space]

        print(f"New Observations: Robot 1 {self.robot1_observation_space}; Robot 2 {self.robot2_observation_space}")
        
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
            print("EPISODES GREATER THAN")
            done = True
            #self.data['end_condition'][-1] = 'time out'
            print("GREATERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR")
        else:  
            reward = 0

        #if enter_end: reward += 100
        #if exit_end: reward -= 100

        #REWARD CALCULATION    
        reward1 += self.optimized_reward_function(a1, 1)
        reward2 += self.optimized_reward_function(a2, 2)

        print(f"Rewards: Robot 1 {reward + reward1}, Robot 2 {reward + reward2}")

        self.total_steps += 1
        self.step_number += 1
        if done:
            self.total_episodes += 1
            self.step_number = 0
            if self.end_reached and self.end_reached2: self.success_episodes += 1

        info = {}

        self.action_history.append(a1)
        self.action_history2.append(a2)

        self.data['actions'][-1].append((a1, a2))
        self.data['rewards'][-1].append(reward)
        self.data['positions'][-1].append((self.position, self.position2))

        # Split the state for each agent based on your state formation
        rewards = [reward + reward1, reward + reward2]  # If shared rewards, otherwise calculate individually
        dones = done  # Both agents likely share the same done flag in your scenario
        infos = [{}, {}]  # Additional info if any, per agent
        self.episode += 1
        return self.state, rewards, dones, infos      

    def optimized_reward_function(self, action, chair):
        #angles = [0, -10, -25, -45, -65, -89, distance, angle ]  # Define specific angles to sample
        target_distance=0.3
        angle_penalty_factor=2
        distance_penalty_factor=1.5 
        wall_penalty_factor=1
        wall_threshold=0.25

        # Get lidar samples for obstacle detection
        lidar = self.lidar_sample if chair == 1 else self.lidar_sample2
        lidar_readings = lidar[:6]  # First 8 elements are the LiDAR readings
        robot_distance = lidar[7]   # Distance to the other robot
        robot_angle = lidar[8]      # Angle to the other robot
        
        # Calculate angle penalty
        angle_difference = abs(abs(robot_angle) - 90)  # Difference from the target angle
        angle_penalty = angle_penalty_factor * angle_difference 
        
        # Calculate distance penalty
        distance_difference = abs(robot_distance - target_distance)  # Difference from the target distance
        distance_penalty = distance_penalty_factor * distance_difference
        
        # Calculate wall penalty
        wall_penalty = 0
        for distance in lidar_readings:
            if distance < wall_threshold:
                wall_penalty += wall_penalty_factor * (wall_threshold - distance) ** 2  # Exponential penalty
        
        # Total reward is negative sum of penalties
        total_penalty = angle_penalty + distance_penalty + wall_penalty
        reward = -total_penalty

        # Reward or penalize based on progress towards the goal
        current_distance = self.distance1 if chair == 1 else self.distance2
        previous_distance = self.previous_distance1 if chair == 1 else self.previous_distance2
        if previous_distance is not None and current_distance is not None:

            distance_improvement = previous_distance - current_distance  # Positive if closer to the goal

            # Adjust reward or penalty for progress
            if distance_improvement > 0:
                reward += distance_improvement 

        # Update previous distance for next step comparison
        if chair == 1:
            self.previous_distance1 = current_distance
        else:
            self.previous_distance2 = current_distance

        return reward


    def render(self): pass
    
    def reset(self):
        # Stop the robots to gather data and calculate rewards
        self.change_robot_speed(1, 0, 0)
        self.change_robot_speed(2, 0, 0)
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
            theta1 = random.uniform(0.5, 2.5)
            theta2 = random.uniform(0.5, 2.5)
        else: 
            y -= 0.2
            y2 += 0.2
            theta = math.pi

        print("THETAAAAAAAAAAAA")
        print(theta)

        self.change_robot_position("robot1", x, y, theta1)
        self.change_robot_position("robot2", x2, y2, theta2)

        target_x = map[2][0]
        target_y = map[2][1]
        self.change_robot_position("prox", target_x, target_y, 0)

        self.previousa1 = 0
        self.previousa2 = 0
        self.collisions = False
        self.finished = False
        self.finished2 = False
        self.end_reached = False
        self.end_reached2 = False
        self.episode = 0
        self.naction1 = False
        self.naction2 = False
        self.previous_distance1 = None
        self.previous_distance2 = None
        self.distance1 = None
        self.distance2 = None

        self.robot1_observation_space = []
        self.robot2_observation_space = []
        self.robot1_action_space = [0,0]
        self.robot2_action_space = [0,0]

        #Wait for observations to fill in (lidar samples + previous actions + calculated features)
        while not (self.naction1 and self.naction2):
            # Optionally, include a small delay to prevent the loop from consuming too much CPU
            time.sleep(0.0001)  
        self.state = [self.robot1_observation_space,  self.robot2_observation_space]

        self.action_history = []
        self.rotation_counter = 0
        self.rotations = 0
        self.consecutive_rotations = 0

        self.action_history2 = []
        self.rotation_counter2 = 0
        self.rotations2 = 0
        self.consecutive_rotations2 = 0

        self.forward_reward = 0
        self.step_number=0

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

    def get_lidar_samples(self, data, num_ranges, chair):
        angles = [0, -30, -60, -90, 30, 60, 90 ]
        
        sampled_points = []
        
        for angle in angles:
            # Map the angle to the corresponding index in the LIDAR data array
            # LIDAR angle range: -180 to 180 degrees, data.ranges covers this range
            # Convert the angle to the range index
            index = int((angle + 180) / 360 * num_ranges) % num_ranges
            
            # Get the corresponding distance from LIDAR data
            distance = data.ranges[index]
            if distance > self.max_range: distance= self.max_range
            
            # Store the sampled angle and distance
            sampled_points.append((angle, distance))
        
        return sampled_points

    def publish_highlighted_points(self, sampled_points, discontinuity_points, topic_name):
        marker = Marker()
        marker.header.frame_id = "static_laser_link1"
        marker.header.stamp = rospy.Time.now()
        marker.type = marker.POINTS
        marker.action = marker.ADD

        points = []
        colors = []
        
        # Add sampled points
        for angle, distance in sampled_points:
            points.append(Point(np.cos(np.deg2rad(angle)) * distance, np.sin(np.deg2rad(angle)) * distance, 0))
            colors.append(ColorRGBA(1.0, 0.0, 0.0, 1.0))  # Red color for regular points
        
        # Add discontinuity points
        for angle, distance in discontinuity_points:
            points.append(Point(np.cos(np.deg2rad(angle)) * distance, np.sin(np.deg2rad(angle)) * distance, 0))
            colors.append(ColorRGBA(0.0, 1.0, 0.0, 1.0))  # Green color for discontinuity points

        marker.points = points
        marker.colors = colors
        marker.scale.x = 0.15
        marker.scale.y = 0.15

        publisher = rospy.Publisher(topic_name, Marker, queue_size=10)
        publisher.publish(marker)

    def find_discontinuity_and_robot(self, data_ranges, num_ranges, side):
        discontinuity_threshold = 0.6  # Arbitrary factor to indicate significant decrease
        consecutive_count = 3  # Number of consecutive points needed to confirm discontinuity
        
        if side == "right":
            # Check right side (0 to -150 degrees)
            start_index = int((0 + 180) / 360 * num_ranges) % num_ranges
            end_index = int((-150 + 180) / 360 * num_ranges) % num_ranges
        elif side == "left":
            # Check left side (0 to 150 degrees)
            start_index = int((0 + 180) / 360 * num_ranges) % num_ranges
            end_index = int((150 + 180) / 360 * num_ranges) % num_ranges

        if end_index < start_index:
            start_index, end_index = end_index, start_index

        detected_points = []
        discontinuity_points = []
        in_discontinuity = False
        
        for i in range(start_index, end_index):
            if i > 0:
                if not in_discontinuity and data_ranges[i] < data_ranges[i - 1] * discontinuity_threshold:
                    # Start of a new discontinuity
                    discontinuity_start = i
                    in_discontinuity = True
                    discontinuity_points.append((i, data_ranges[i]))
                elif in_discontinuity:
                    # Continue until a significant increase is detected
                    if data_ranges[i] > data_ranges[i - 1] / discontinuity_threshold:
                        # End of the discontinuity
                        in_discontinuity = False
                        break
                    else:
                        discontinuity_points.append((i, data_ranges[i]))
        
        if discontinuity_points:
            angles_distances = [(k / num_ranges * 360 - 180, data_ranges[k]) for k, _ in discontinuity_points]
            min_distance = min(data_ranges[k] for k, _ in discontinuity_points)
            min_index = data_ranges.index(min_distance)
            angle = (min_index / num_ranges) * 360 - 180
            return min_distance, angle, angles_distances
        else:
            return None, None, []

    def sample_lidar(self, data):  # LEFT SIDE ROBOT
        if self.naction1:
            return
        print("Sampling Lidar 1")

        num_ranges = len(data.ranges)
        self.lidar_sample = []

        sampled_points = self.get_lidar_samples(data, num_ranges, 1)
        for angle, distance in sampled_points:
            self.lidar_sample.append(distance)

        robot_distance, robot_angle, discontinuity_points = self.find_discontinuity_and_robot(data.ranges, num_ranges, "right")
        if robot_distance is not None and robot_angle is not None:
            self.lidar_sample.append(robot_distance)
            self.lidar_sample.append(robot_angle)
        else:
            self.lidar_sample.append(-1)
            self.lidar_sample.append(-1)


        self.publish_highlighted_points(sampled_points, discontinuity_points, "left_lidar_markers")

        self.robot1_observation_space = self.lidar_sample
        print(self.robot1_observation_space)

        self.naction1 = True

    def sample_lidar2(self, data):  # RIGHT SIDE ROBOT
        if self.naction2:
            return
        print("Sampling Lidar 2")

        num_ranges = len(data.ranges)
        self.lidar_sample2 = []

        sampled_points = self.get_lidar_samples(data, num_ranges, 2)
        for angle, distance in sampled_points:
            self.lidar_sample2.append(distance)

        robot_distance, robot_angle, discontinuity_points= self.find_discontinuity_and_robot(data.ranges, num_ranges, "left")
        if robot_distance is not None and robot_angle is not None:
            self.lidar_sample2.append(robot_distance)
            self.lidar_sample2.append(robot_angle)
        else:
            self.lidar_sample2.append(-1)
            self.lidar_sample2.append(-1)


        self.publish_highlighted_points(sampled_points, discontinuity_points, "right_lidar_markers")

        self.robot2_observation_space = self.lidar_sample2

        self.naction2 = True

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
        self.adj_steps = 0
        self.step_number = 0

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


