#!/usr/bin/env python3

import globals
import numpy as np
from tensorflow.keras.optimizers import Adam
from envs.one_wheelchair_env import OneWheelchairEnv
from envs.one_wheelchair_env_with_dist import  OneWheelchairEnvWithDist
from envs.two_wheelchair_env import TwoWheelChairEnv
from envs.two_wheelchair_env_large_target import TwoWheelChairEnvLargeTarget
from envs.two_wheelchair_env_less_actions import TwoWheelChairEnvLessActions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def build_model(states, actions):
    model = Sequential()    
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Flatten())
    model.add(Dense(actions, activation='linear'))
    return model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,  nb_actions=actions, nb_steps_warmup=100, target_model_update=1e-2)
    return dqn

if __name__ == '__main__':
    if globals.use_two_wc:
        env = TwoWheelChairEnvLessActions()
    else:
        env = OneWheelchairEnv()
    env.reset_robots()
    for map in globals.maps.items():
        if map[1]['usage']:
            env.start_points.append((map[0], map[1]['robot_space'], map[1]['end_space']))

    states = env.observation_space.shape
    actions = env.action_space.n
    
    model = build_model(states, actions)
    #model.summary()
    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    if globals.load:
        dqn.load_weights(globals.load_file)

    if globals.train == 1:
        env.task = 'train'
        dqn.fit(env, nb_steps=globals.nsteps, visualize=False, verbose=1)
        if globals.save:
            dqn.save_weights(globals.save_file, overwrite=True)
    elif globals.train == 2:
        env.reset_counters()
        max_acc = 0
        i=1
        while(globals.acc_thresh > ((env.success_episodes / env.total_episodes) if env.total_episodes else 0) or 
                globals.forward_movement_thresh > ((env.forward_steps / env.total_steps) if env.total_steps else 0)):
            env.task = 'train'
            dqn.fit(env, nb_steps=10000, visualize=False, verbose=1)

            env.reset_counters()
            env.task = 'test'
            print('\n', 'Test number:', i)
            scores = dqn.test(env, nb_episodes=globals.test_episodes, visualize=False)
            acc = env.success_episodes / env.total_episodes
            print('Accurancy:', acc, '/', globals.acc_thresh)
            print('Forward movements', env.forward_steps / env.total_steps, '/', globals.forward_movement_thresh)
            print('Adjacency', env.adj_steps / env.total_steps)
            if globals.save and acc > max_acc:
                max_acc = acc
                dqn.save_weights(globals.save_file, overwrite=True)
                if globals.save_data: env.dump_data(globals.data_file)
            i+=1

    if globals.test:
        env.reset_counters()
        env.task = 'test'
        scores = dqn.test(env, nb_episodes=globals.test_episodes, visualize=False)
        env.success_episodes / env.total_episodes
        print('Accurancy:', env.success_episodes / env.total_episodes, '/', globals.acc_thresh)
        print('Forward movements', env.forward_steps / env.total_steps, '/', globals.forward_movement_thresh)
        print('Adjacency', env.adj_steps / env.total_steps)

    if globals.save_data:
        env.dump_data(globals.data_file)

    env.reset_robots()
