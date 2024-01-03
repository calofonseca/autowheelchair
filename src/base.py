import inspect
import logging
import os
from pathlib import Path
import pickle
from typing import Any, List, Mapping, Tuple, Union
from gym import spaces
import time


class Agent():
  
    def __init__(self):
        self.step=0

    def learn(self, env, steps):
        """Train agent.

        Parameters
        ----------
        steps: int, default: 1
            Number of training steps greater :math:`\ge 1`.
        keep_env_history: bool, default: False
            Indicator to store environment state at the end of each episode.
        env_history_directory: Union[str, Path], optional
            Directory to save environment history to.
        deterministic: bool, default: False
            Indicator to take deterministic actions i.e. strictly exploit the learned policy.
        deterministic_finish: bool, default: False
            Indicator to take deterministic actions in the final episode.
        logging_level: int, default: 30
            Logging level where increasing the number silences lower level information.      
        """
        
        rewards_all = []
        individual_runtimes_predict = []
        average_runtime = 0
        kpis_list = []
        observations_ep = []
        for episode in range(episodes):
            deterministic = deterministic or (deterministic_finish and episode >= episodes - 1)
            observations = self.env.reset()
            rewards_ep = []

            while not self.env.done:
                print("\n \n ------TIME STEP------")
                print(f"{episode} - {self.env.time_step}")

                observations_ep.append(observations)
                start_time = time.time()  # Get the current time
                actions = self.predict(observations, deterministic=deterministic)
                end_time = time.time()  # Get the current time again after the function has run

                elapsed_time = end_time - start_time  # Calculate the elapsed time
                individual_runtimes_predict.append(elapsed_time)

                next_observations, rewards, _, _ = self.env.step(actions)
                rewards_ep.append(rewards)


                if not deterministic:
                    self.update(observations, actions, rewards, next_observations, done=self.env.done)
                else:
                    pass

                observations = [o for o in next_observations]

                logging.debug(
                    f'Time step: {self.env.time_step}/{self.env.time_steps - 1},'\
                        f' Episode: {episode}/{episodes - 1},'\
                            f' Actions: {actions},'\
                                f' Rewards: {rewards}'
                )

            # Calculate the average runtime
            average_runtime = sum(individual_runtimes_predict) / len(individual_runtimes_predict)

            #Save kpis
            kpis = self.env.evaluate().pivot(index='cost_function', columns='name', values='value')
            kpis = kpis.dropna(how='all')
            kpis_list.append(kpis)

            rewards_ep = [reward for reward in rewards_ep if isinstance(reward, List)]
            rewards_all.append(rewards_ep) #rewards all is a list, of lists, of lists [[ep1[ts1][ts2]], [ep2[B1][B2]], ....]

            # store episode's env to disk
            if keep_env_history:
                self.__save_env(episode, env_history_directory)
            else:
                pass

        #return rewards_all, average_runtime, kpis_list
        return rewards_all, average_runtime, kpis_list, observations_ep

    def __save_env(self, episode: int, directory: Path):
        """Save current environment state to pickle file."""

        filepath = os.path.join(directory, f'{int(episode)}.pkl')

        with open(filepath, 'wb') as f:
            pickle.dump(self.env, f)


    def predict(self, observations: List[List[float]], deterministic: bool = None) -> List[List[float]]:
        """Provide actions for current time step.

        Return randomly sampled actions from `action_space`.
        
        Parameters
        ----------
        observations: List[List[float]]
            Environment observations
        deterministic: bool, default: False
            Wether to return purely exploitatative deterministic actions.

        Returns
        -------
        actions: List[List[float]]
            Action values
        """
        
        actions = [list(s.sample()) for s in self.action_space]
        self.actions = actions
        self.next_time_step()
        return actions
    
    def __set_logger(self, logging_level: int = None):
        """Set logging level."""

        logging_level = 30 if logging_level is None else logging_level
        assert logging_level >= 0, 'logging_level must be >= 0'
        LOGGER.setLevel(logging_level)

    def update(self, *args, **kwargs):
        """Update replay buffer and networks.
        
        Notes
        -----
        This implementation does nothing but is kept to keep the API for all agents similar during simulation.
        """

        pass

    def next_time_step(self):
        super().next_time_step()

        for i in range(len(self.action_space)):
            self.__actions[i].append([])

    def reset(self):
        super().reset()
        self.__actions = [[[]] for _ in self.action_space]