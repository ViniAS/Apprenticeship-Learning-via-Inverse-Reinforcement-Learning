import gymnasium as gym
import numpy as np
import frozenlake_agent
import matplotlib.pyplot as plt
import time
import torch
from torch.distributions import Normal
from torch_sgld import SGLD
import cvxpy as cp
import seaborn as sns


class IrlAgentMaxMargin(frozenlake_agent.FrozenLakeQLearning):
    def __init__(self, expert_feature_expectation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_feature_expectation = expert_feature_expectation
        self.weights = np.random.rand(16)
        self.feature_expectations = []
        self.feature_expectations_bar = []
        self.weights_history = []

    def get_reward(self):
        reward = self.weights[self.obs]
        return reward

    def train_weights(self, N=500, lr=1e-2):
        rewards = []
        losses = []
        for i in range(N):
            self.Q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
            rewards.append(self.run())
            self.feature_expectations.append(self.get_feature_expectation())

            weights = cp.Variable(16)
            t = cp.Variable(1)

            constraints = [weights.T @ self.expert_feature_expectation >=
                            weights.T @ feature_expectation + t for feature_expectation in self.feature_expectations]
            constraints += [cp.norm(weights, 2) <= 1]

            objective = cp.Maximize(t)

            problem = cp.Problem(objective, constraints)
            problem.solve()

            self.weights = weights.value
            self.weights_history.append(self.weights)

            losses.append(t.value)
            if t.value and t.value < 1e-5:
                break

        return rewards, losses


class IrlAgentProjection(frozenlake_agent.FrozenLakeQLearning):
    def __init__(self, expert_feature_expectation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_feature_expectation = expert_feature_expectation
        self.weights = np.random.rand(16)
        self.weights_history = []
        self.feature_expectations = []
        self.feature_expectations_bar = []

    def get_reward(self):
        reward = self.weights[self.obs]
        return reward

    def train_weights(self, N=500, lr=1e-2):
        self.Q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.feature_expectations.append(self.get_feature_expectation())

        self.feature_expectations_bar.append(self.feature_expectations[0])
        self.weights = self.expert_feature_expectation - self.feature_expectations_bar[0]

        rewards = []
        losses = []
        for i in range(1,N):
            self.Q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
            rewards.append(self.run())

            self.feature_expectations.append(self.get_feature_expectation())
            A = self.feature_expectations[i] - self.feature_expectations_bar[i-1]
            B = self.expert_feature_expectation - self.feature_expectations_bar[i-1]
            projection = (np.dot(A, B)/np.dot(A, A))*A
            self.feature_expectations_bar.append(self.feature_expectations_bar[i-1] + projection)
            self.weights = self.expert_feature_expectation - self.feature_expectations_bar[i]
            self.weights_history.append(self.weights)
            losses.append(np.linalg.norm(self.weights))
            if np.linalg.norm(self.weights) < 1e-5:
                break

        return rewards, losses



if __name__ == "__main__":
    time_start = time.time()
    expert = frozenlake_agent.FrozenLakeQLearning(num_episodes=1000, randomness = 0.0)
    lengths = expert.run()

    expert.draw_table()

    expert_feature_expectation = expert.get_feature_expectation(num_episodes=10000)

    agent = IrlAgentProjection(expert_feature_expectation, num_episodes=10000, randomness = 0.0)
    agent_max_margin = IrlAgentMaxMargin(expert_feature_expectation, num_episodes=10000, randomness = 0.0)

    rewards_max_margin, losses_max_margin = agent_max_margin.train_weights(N=20)
    rewards, losses = agent.train_weights(N=20)
    time_end = time.time()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].plot(losses_max_margin)
    axs[0].set_ylabel("t")
    axs[0].set_xlabel("Iteration")
    axs[0].set_title("FrozenLake: Max Margin IRL")

    axs[1].plot(losses)
    axs[1].set_ylabel("t")
    axs[1].set_xlabel("Iteration")
    axs[1].set_title("FrozenLake: Projection IRL")

    plt.savefig("img/loss.png")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    sns.heatmap(agent_max_margin.weights.reshape(4,4), annot=True, ax=axs[1])
    sns.heatmap(agent.weights.reshape(4, 4), annot=True, ax=axs[2])

    true_weights = np.zeros((4,4))
    true_weights[1,1] = -1
    true_weights[1,3] = -1
    true_weights[2,3] = -1
    true_weights[3,0] = -1
    true_weights[3,3] = 1

    sns.heatmap(true_weights, annot=True, ax=axs[0])

    axs[0].set_title("FrozenLake: True Weights")
    axs[1].set_title("FrozenLake: Max Margin IRL")
    axs[2].set_title("FrozenLake: Projection IRL")

    plt.savefig("img/weights.png")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    sns.heatmap(agent_max_margin.feature_expectations[-1].reshape(4,4), annot=True, ax=axs[1])
    axs[1].set_title("FrozenLake: Max Margin Feature Expectation")

    sns.heatmap(agent.feature_expectations[-1].reshape(4,4), annot=True, ax=axs[2])
    axs[2].set_title("FrozenLake: Projection Feature Expectation")

    sns.heatmap(expert_feature_expectation.reshape(4,4), annot=True, ax=axs[0])
    axs[0].set_title("FrozenLake: Expert Feature Expectation")

    plt.savefig("img/feature_expectation.png")

    print("Projection weights:")
    for weight in agent.weights_history:
        print(weight.reshape(4,4))
    print("Max Margin weights:")
    for weight in agent_max_margin.weights_history:
        print(weight.reshape(4,4))




    









