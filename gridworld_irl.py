import numpy as np
import math
import matplotlib.pyplot as plt
from time import time
import gridworld
import torch
from torch_sgld import SGLD
from torch.distributions import Laplace


class GridWorldIRL(gridworld.GridWorldQLearning):
    def __init__(self, expert_feature_expectation, env, num_episodes=100):
        super().__init__(env=env, num_episodes=num_episodes)
        self.expert_feature_expectation = expert_feature_expectation
        self.weights = np.random.rand(self.env.num_macrocells ** 2)
        self.feature_expectations = []
        self.feature_expectations_bar = []
        
    def get_agent_reward(self):
        return np.dot(self.weights, self.env.get_feature())
    
    def train_weights(self, N=10):
        self.Q_table = np.zeros((self.env.num_macrocells ** 2, 4))
        self.feature_expectations.append(self.get_feature_expectation())
        
        self.feature_expectations_bar.append(self.feature_expectations[0])
        self.weights = self.expert_feature_expectation - self.feature_expectations_bar[0]
        rewards = []

        losses = []
        
        for i in range(1, N):
            self.Q_table = np.zeros((self.env.num_macrocells ** 2, 4))
            rewards.append(self.run())
            self.feature_expectations.append(self.get_feature_expectation())
            A = self.feature_expectations[i] - self.feature_expectations_bar[i-1]
            B = self.expert_feature_expectation - self.feature_expectations_bar[i-1]
            projection = (np.dot(A, B) / np.dot(A, A)) * A
            self.feature_expectations_bar.append(self.feature_expectations_bar[i-1] + projection)
            self.weights = self.expert_feature_expectation - self.feature_expectations_bar[i]
            print(f"weights: {np.round(self.weights, 3)}")
            losses.append(np.linalg.norm(self.weights))
            if np.linalg.norm(self.weights) < 1e-5:
                break
                
        return rewards, losses


class GridWorldIRLBayesian(gridworld.GridWorldQLearning):
    def __init__(self, expert_feature_expectation, env, num_episodes=100):
        super().__init__(env, num_episodes=num_episodes)
        self.expert_feature_expectation = torch.tensor(expert_feature_expectation, dtype=torch.float32)
        self.weights = np.random.rand(self.env.num_macrocells ** 2)
        self.feature_expectations = []
        self.feature_expectations_bar = []

    def get_agent_reward(self):
        return np.dot(self.weights, self.env.get_feature())

    def train_weights(self, N=10, lr=1e-2, prior: torch.distributions.Distribution = Laplace(0, 1)):
        self.Q_table = np.zeros((self.env.num_macrocells ** 2, 4))

        weights = prior.sample((self.env.num_macrocells ** 2,))
        weights.requires_grad = True
        self.weights = weights.detach().numpy()
        rewards = []

        optimizer = SGLD([weights], lr=lr)

        losses = []

        for i in range(1, N):
            self.Q_table = np.zeros((self.env.num_macrocells ** 2, 4))
            rewards.append(self.run())
            self.feature_expectations.append(self.get_feature_expectation())

            log_prior = prior.log_prob(weights).sum()

            feature_expectations_tensor = torch.tensor(self.feature_expectations[-1], dtype=torch.float32)

            loss = torch.norm(self.expert_feature_expectation - feature_expectations_tensor) - log_prior
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.weights = weights.detach().numpy()
            print(f"weights: {np.round(self.weights, 3)}")
            if np.linalg.norm(self.weights) < 1e-5:
                break

        return rewards, losses




    
if __name__ == "__main__":
    # get time to run
    env = gridworld.GridWorld()
    time_start = time()
    expert = gridworld.GridWorldQLearning(env=env, num_episodes=1000)
    rewards = expert.run()
    expert_feature_expectation = expert.get_feature_expectation(num_episodes=1000)
    agent = GridWorldIRL(expert_feature_expectation, env)
    agent_bayesian = GridWorldIRLBayesian(expert_feature_expectation, env)
    rewards_irl_bayesian, losses_bayesian = agent_bayesian.train_weights()
    rewards_irl, losses = agent.train_weights()
    time_end = time()
    distance_feature_expectations = []

    print("-"*20)
    print(f"weights irl: {np.round(agent.weights, 3)}")
    print(f"weights irl bayesian: {np.round(agent_bayesian.weights, 3)}")
    print(f"weights expert: {np.round(expert.env.rewards, 3)}")

    plt.plot(losses, label='IRL')
    plt.plot(losses_bayesian, label='IRL Bayesian')
    plt.xlabel('Iterations')
    plt.ylabel('Distance')
    plt.title('Distance to expert over iterations. Run time: ' + str(time_end - time_start) + 's')
    plt.legend()
    plt.savefig('irl_gridworld.png')

    # plot expert rewards
    plt.figure()
    plt.plot(rewards, label='Expert')
    plt.plot(rewards_irl[-1], label='Apprentice')
    plt.plot(rewards_irl_bayesian[-1], label='Apprentice Bayesian')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Expert vs Apprentice reward over episodes')
    plt.legend()
    plt.savefig('rewards_irl.png')


