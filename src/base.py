class BaseClass:
    def __init__(self):
        # Initialize any necessary variables or configurations
        pass

    def fit(self, env, total_steps):
        step_count = 0
        end_reached=False
        while step_count < total_steps:
            print("RESETED")
            obs = env.reset()
            while True:
                action, action2 = self.predict(obs)
                next_obs, reward, done, _ = env.step(action, action2)
                print(f"STEP:{step_count} Reward:{reward}")
                self.update(obs, [action, action2], reward, next_obs, done)
                obs = next_obs
                step_count += 1
                if env.end_reached and env.end_reached2:
                    end_reached=True
                # Update epsilon only every 200 steps
                if step_count % 100 == 0:
                    self.update_epsilon(end_reached)
                if done or step_count >= total_steps:
                    print("BREAKED")
                    break

            if step_count % 100 == 0:  # Just as an example, print every 100 steps
                print(f"Completed Steps: {step_count}/{total_steps}")

    def test(self, env, total_episodes):
        episode_count = 0
        total_reward = 0

        while episode_count < total_episodes:
            obs = env.reset()
            episode_reward = 0

            while True:
                action = self.predict(obs, deterministic=True)
                next_obs, reward, done, _ = env.step(*action)
                obs = next_obs
                episode_reward += reward[0]

                if done:
                    break
        
            total_reward += episode_reward
            episode_count += 1
            print(f"Reward for episode {episode_count}: {episode_reward}")

        print(f"Total Reward after {total_episodes} episodes: {total_reward}")

    def update(self, observations, actions, reward, next_observations, done):
        # Update the model
        raise NotImplementedError("This method should be overridden by the model extending this base class")

    def predict(self, observations, deterministic=False):
        # Predict the action
        raise NotImplementedError("This method should be overridden by the model extending this base class")
    
    def update_epsilon(self):  # Call this method at the end of each episode if you want epsilon to decay
        # Predict the action
        raise NotImplementedError("This method should be overridden by the model extending this base class")