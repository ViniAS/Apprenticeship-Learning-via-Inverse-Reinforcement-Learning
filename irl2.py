import gymnasium as gym
import numpy as np
import cartpole_agent
import matplotlib.pyplot as plt
import time


class IrlAgent(cartpole_agent.CartPoleQLearning):
    def __init__(self, expert_feature_expectation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_feature_expectation = expert_feature_expectation
        self.weights = np.random.rand(4)
        self.feature_expectations = []
        self.feature_expectations_bar = []

    def get_reward(self):
        obs = 1/(1 + np.exp(-self.obs) + 1e-6)
        return np.dot(self.weights, obs)

    def train_weights(self, N = 10):
        self.Q_table = np.zeros(self.buckets + (self.env.action_space.n,))
        self.feature_expectations.append(self.get_feature_expectation())

        self.feature_expectations_bar.append(self.feature_expectations[0])
        self.weights = self.expert_feature_expectation - self.feature_expectations_bar[0]
        games_lengths = []
        for i in range(1, N):
            self.Q_table = np.zeros(self.buckets + (self.env.action_space.n,))
            games_lengths.append(self.run())
            self.feature_expectations.append(self.get_feature_expectation())
            A = self.feature_expectations[i]-self.feature_expectations_bar[i-1]
            B = self.expert_feature_expectation-self.feature_expectations_bar[i-1]
            projection = (np.dot(A, B)/np.dot(A, A))*A
            self.feature_expectations_bar.append(self.feature_expectations_bar[i-1] + projection)
            self.weights = self.expert_feature_expectation - self.feature_expectations_bar[i]
            if np.linalg.norm(self.weights) < 1e-5:
                break

        return games_lengths

if __name__ == '__main__':
    # get time to run
    time_start = time.time()
    expert = cartpole_agent.CartPoleQLearning()
    lengths = expert.run()
    expert_feature_expectation = expert.get_feature_expectation()
    agent = IrlAgent(expert_feature_expectation)
    lengths_irl = agent.train_weights()
    time_end = time.time()
    print("weights: ", agent.weights)
    print("last run: ", lengths_irl[-1])
    plt.plot(lengths, label='Expert')
    plt.plot(lengths_irl[0], label=f'IRL Agent 0')
    plt.plot(lengths_irl[-1], label=f'IRL Agent {len(lengths_irl)-1}')
    plt.title('IRL Agent: time to run: ' + str(time_end - time_start) + 's')
    plt.xlabel('Episode')
    plt.ylabel('Game Length')
    plt.legend()
    plt.savefig('irl2.png')
    agent.play()









