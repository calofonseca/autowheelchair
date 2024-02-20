#!/usr/bin/env python3

import globals
import numpy as np
from envs.one_wheelchair_env import OneWheelchairEnv
from envs.one_wheelchair_env_with_dist import  OneWheelchairEnvWithDist
from envs.two_wheelchair_env import TwoWheelChairEnv
from envs.two_wheelchair_env_large_target import TwoWheelChairEnvLargeTarget
from envs.two_wheelchair_env_less_actions import TwoWheelChairEnvLessActions
from maddpg import MADDPG as RLAgent

if __name__ == '__main__':
    if globals.use_two_wc:
        env = TwoWheelChairEnvLessActions()
        num_wheelchairs = 2
    else:
        env = OneWheelchairEnv()
        num_wheelchairs = 1
    env.reset_robots()
    for map in globals.maps.items():
        if map[1]['usage']:
            env.start_points.append((map[0], map[1]['robot_space'], map[1]['end_space']))


    states = env.observation_space.shape
    actions = env.action_space.n
   
    model = RLAgent(int(num_wheelchairs), int(38/2), int(4), critic_units=[512, 256, 128, 64, 32], 
                    actor_units=[256,128,64,32,16], lr_actor=1e-4, lr_critic=1e-3)

    if globals.load:
        model.load_weights(globals.load_file)

    if globals.train == 1:
        env.task = 'train'
        model.fit(env, total_steps = globals.nsteps)
        if globals.save:
            model.save_weights(globals.save_file)
            print("Weights Saved Successfully")
    elif globals.train == 2:
        env.reset_counters()
        max_acc = 0
        i=1
        while(globals.acc_thresh > ((env.success_episodes / env.total_episodes) if env.total_episodes else 0) or 
                globals.forward_movement_thresh > ((env.forward_steps / env.total_steps) if env.total_steps else 0)):
            env.task = 'train'
            model.learn(steps = globals.nsteps)

            env.reset_counters()
            env.task = 'test'
            print('\n', 'Test number:', i)
            scores = model.test(env, total_episodes=globals.test_episodes)
            acc = env.success_episodes / env.total_episodes
            print('Accurancy:', acc, '/', globals.acc_thresh)
            print('Forward movements', env.forward_steps / env.total_steps, '/', globals.forward_movement_thresh)
            print('Adjacency', env.adj_steps / env.total_steps)
            if globals.save and acc > max_acc:
                max_acc = acc
                #dqn.save_weights(globals.save_file, overwrite=True)
                raise NotImplementedError("Could not save weights")
                if globals.save_data: env.dump_data(globals.data_file)
            i+=1
            

    if globals.test:
        env.reset_counters()
        env.task = 'test'
        scores = model.test(env, total_episodes=globals.test_episodes)
        env.success_episodes / env.total_episodes
        print('Accurancy:', env.success_episodes / env.total_episodes, '/', globals.acc_thresh)
        print('Forward movements', env.forward_steps / env.total_steps, '/', globals.forward_movement_thresh)
        print('Adjacency', env.adj_steps / env.total_steps)

    if globals.save_data:
        env.dump_data(globals.data_file)

    env.reset_robots()
