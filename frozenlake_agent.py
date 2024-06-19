import gymnasium as gym
import numpy as np
import math
import time

class FrozenLakeQLearning:
    def __init__(self, num_episodes=500, min_lr=0.1, min_epsilon=0.1, discount=0.99, randomness = 0.0, decay=25):
        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.decay = decay
        self.reward = 0
        self.obs = None
        self.randomness = randomness

        self.env = gym.make('FrozenLake-v1', is_slippery=False, map_name="4x4")



        # Initialize the action-value function to 0
        self.Q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def choose_action(self, state, epsilon):
        x = state % 4
        y = state // 4
        allowed_actions = []
        if x > 0:
            allowed_actions.append(0)
        if x < 3:
            allowed_actions.append(2)
        if y > 0:
            allowed_actions.append(3)
        if y < 3:
            allowed_actions.append(1)

        return np.random.choice(allowed_actions) if (np.random.random() <= epsilon) or np.random.random() <= self.randomness \
              else np.argmax(self.Q_table[state])

    def update_q(self, state_old, action, reward, state_new, alpha):
        self.Q_table[state_old][action] += alpha * (reward + self.discount * np.max(self.Q_table[state_new]) -
                                                    self.Q_table[state_old][action])

    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))

    def get_alpha(self, t):
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))

    def run(self):
        print("Running...")
        total_reward = [0] * self.num_episodes
        for t in range(self.num_episodes):
            current_state = self.env.reset()[0]
            done = False
            i = 0
            while not done and i < 20000:
                epsilon = self.get_epsilon(t)
                action = self.choose_action(current_state, epsilon)
                self.obs, reward, done, _, _ = self.env.step(action)
                new_state = self.obs
                if done and reward == 0:
                    reward = -1  # Negative reward for falling into a hole
                alpha = self.get_alpha(t)
                self.reward = reward
                self.update_q(current_state, action, self.get_reward(), new_state, alpha)
                current_state = new_state
                total_reward[t] += reward
                i += 1
                if done:
                    break
        return total_reward

    def play(self, render=True):
        state = self.env.reset()[0]
        done = False
        while not done:
            if render:
                self.env.render()
                time.sleep(0.5)
            action = np.argmax(self.Q_table[state])
            obs, reward, done, _, _ = self.env.step(action)
            state = obs
        return reward

    def get_feature_expectation(self, gamma=1, num_episodes=1000):
        feature_expectation = np.zeros(self.env.observation_space.n)
        for _ in range(num_episodes):
            done = False
            current_state = self.env.reset()[0]
            feature_expectation[current_state] += 1
            t = 0
            while not done and t < 1000:
                action = self.choose_action(current_state, 0)
                obs, reward, done, _, _ = self.env.step(action)
                current_state = obs
                feature_expectation[current_state] += gamma ** t
                t += 1
        return feature_expectation / num_episodes

    def draw_table(self):
        table = np.zeros((4, 4), dtype=int)
        for i in range(4):
            for j in range(4):
                table[i][j] = np.argmax(self.Q_table[i * 4 + j])
        print(table)

    def get_reward(self):
        return self.reward

if __name__ == "__main__":
    agent = FrozenLakeQLearning(num_episodes=5000)

    # save to a file a image of the environment
    agent.env.reset()
    agent.env.render()
    time.sleep(500)




