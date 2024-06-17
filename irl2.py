import gymnasium as gym
import numpy as np
import cartpole_agent
import matplotlib.pyplot as plt
import time
import torch
from torch.distributions import Normal
from torch_sgld import SGLD

class IrlAgent(cartpole_agent.CartPoleQLearning):
    def __init__(self, expert_feature_expectation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_feature_expectation = expert_feature_expectation
        self.weights = np.random.rand(4)
        self.feature_expectations = []
        self.feature_expectations_bar = []

    def get_reward(self):
        obs = 1/(1 + np.exp(-self.obs))
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

class IrlAgentBayesian(cartpole_agent.CartPoleQLearning):
    def __init__(self, expert_feature_expectation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_feature_expectation = torch.tensor(expert_feature_expectation, dtype=torch.float32)
        self.weights = np.random.rand(4)
        self.feature_expectations = []
        self.feature_expectations_bar = []

    def get_reward(self):
        obs = 1/(1 + np.exp(-self.obs))
        return np.dot(self.weights, obs)

    def train_weights(self, N = 10, lr = 1e-2, prior: torch.distributions.Distribution = Normal(0, 1)):
        self.Q_table = np.zeros(self.buckets + (self.env.action_space.n,))
        self.feature_expectations.append(self.get_feature_expectation())

        weights = prior.sample((4,))
        weights.requires_grad = True
        self.weights = weights.detach().numpy()
        games_lengths = []

        optimizer = SGLD([weights], lr=lr)

        losses = []

        for i in range(N):
            self.Q_table = np.zeros(self.buckets + (self.env.action_space.n,))
            games_lengths.append(self.run())
            self.feature_expectations.append(self.get_feature_expectation())

            log_prior = prior.log_prob(weights).sum()

            feature_expectation = torch.tensor(self.feature_expectations[i], dtype=torch.float32)

            feature_expectation_difference = self.expert_feature_expectation - feature_expectation

            dot_product = torch.dot(weights, feature_expectation_difference)

            loss = -dot_product + log_prior

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            self.weights = weights.detach().numpy()

        return games_lengths, losses




if __name__ == '__main__':
    # get time to run
    time_start = time.time()
    expert = cartpole_agent.CartPoleQLearning()
    lengths = expert.run()

    expert_feature_expectation = expert.get_feature_expectation()
    agent = IrlAgentBayesian(expert_feature_expectation)

    games_lengths, losses = agent.train_weights(N=12)

    time_end = time.time()

    # plot results in a 3x4 grid
    plt.figure()
    for i in range(12):
        plt.subplot(3, 4, i+1)
        plt.plot(games_lengths[i], label='Agent')
        plt.plot(lengths, label='Expert')
        plt.title('Game Length vs Episode')
        plt.xlabel('Episode')
        plt.ylabel('Game Length')
        plt.legend()



    # save plot
    plt.savefig('irl_cartpole_rewards.png')

    # plot losses

    plt.figure()
    plt.plot(losses)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss vs Episode')
    plt.savefig('irl_cartpole_loss.png')











