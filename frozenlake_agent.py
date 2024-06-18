import gymnasium as gym
import numpy as np
import math
import time
import seaborn as sns
import matplotlib.pyplot as plt

class FrozenLakeQLearning:
    def __init__(self, num_episodes=500, min_lr=0.1, min_epsilon=0.1, discount=1.0, decay=25):
        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.decay = decay
        self.reward = 0
        self.obs = None

        self.env = gym.make('FrozenLake-v1', is_slippery=False)

        # Initialize the action-value function to 0
        self.Q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q_table[state])

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
            while not done:
                epsilon = self.get_epsilon(t)
                action = self.choose_action(current_state, epsilon)
                obs, reward, done, _, _ = self.env.step(action)
                new_state = obs
                if done and reward == 0:
                    reward = -1  # Negative reward for falling into a hole
                if new_state == current_state:
                    reward -= 0.5
                alpha = self.get_alpha(t)
                self.update_q(current_state, action, reward, new_state, alpha)
                current_state = new_state
                total_reward[t] += reward
                if done:
                    break
        self.reward_matrix = self.Q_table
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

    def get_feature_expectation(self, gamma=0.99, num_episodes=100):
        feature_expectation = np.zeros(self.env.observation_space.n)
        for _ in range(num_episodes):
            done = False
            current_state = self.env.reset()[0]
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

    def calculate_average_rewards(self):
        rewards = np.zeros((4, 4))
        for state in range(self.env.observation_space.n):
            state_rewards = []
            for action in range(self.env.action_space.n):
                obs, reward, done, _, _ = self.env.step(action)
                if done and reward == 0:
                    reward = -1  # Negative reward for falling into a hole
                if obs == state:
                    reward -= 0.5
                state_rewards.append(reward)
                self.env.reset()[0]  # Reset the environment after each step
            rewards[state // 4][state % 4] = np.mean(state_rewards)
        return rewards

    def save_heatmap(self, filename="heatmap.png"):
        rewards = self.calculate_average_rewards()
        plt.figure(figsize=(8, 6))
        sns.heatmap(rewards, annot=True, cmap="coolwarm", cbar=True, square=True, linewidths=.5)
        plt.title("Average Reward Heatmap")
        plt.savefig(filename)
        plt.close()

if __name__ == "__main__":
    start = time.time()
    agent = FrozenLakeQLearning(num_episodes=500)
    agent.run()
    print(f"Time to run: {time.time() - start}")
    print(f"Q_table: {agent.Q_table}")
    agent.play(render=False)
    print("Feature Expectation:")
    print(agent.get_feature_expectation())
    agent.draw_table()
    agent.save_heatmap("frozenlake_heatmap.png")

    for _ in range(10):
        reward = agent.play(render=False)
        print("Reward: ", reward, " for run number ", _)
    agent.env.close()

    print(agent.reward_matrix)
