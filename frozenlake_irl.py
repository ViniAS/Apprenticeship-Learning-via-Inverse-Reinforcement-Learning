import gymnasium as gym
import numpy as np
import frozenlake_agent
import matplotlib.pyplot as plt
import time
import torch
from torch.distributions import Normal
from torch_sgld import SGLD

class IrlAgentBayesian(frozenlake_agent.FrozenLakeQLearning):
    def __init__(self, expert_feature_expectation, prior_type='gaussian', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_feature_expectation = torch.tensor(expert_feature_expectation, dtype=torch.float32)
        self.prior_type = prior_type
        self.weights = np.random.rand(16)
        self.feature_expectations = []
        self.feature_expectations_bar = []

    def get_reward(self, state):
        obs = np.zeros(self.env.observation_space.n)
        obs[state] = 1
        return np.dot(self.weights, obs)
    
    def set_prior(self):
        if self.prior_type == 'gaussian':
            return Normal(0, 1)
        elif self.prior_type == 'laplacian':
            return torch.distributions.Laplace(0, 1)
        elif self.prior_type == 'uniform':
            return torch.distributions.Uniform(-1, 1)
        else:
            raise ValueError("Unsupported prior type")

    def train_weights(self, N=500, lr=1e-2):
        self.Q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.feature_expectations.append(self.get_feature_expectation())

        prior = self.set_prior()
        weights = prior.sample((16,))
        weights.requires_grad = True
        self.weights = weights.detach().numpy()
        optimizer = SGLD([weights], lr=lr)
        games_lengths = []
        losses = []

        print("Training weights using MCMC...")
        for i in range(N):
            self.Q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
            games_lengths.append(self.run())
            self.feature_expectations.append(self.get_feature_expectation())
            for j in range(10):
                for feature_expectation in self.feature_expectations:
                    log_prior = prior.log_prob(weights).sum()
                    feature_expectation = torch.tensor(feature_expectation, dtype=torch.float32)
                    feature_expectation_difference = self.expert_feature_expectation - feature_expectation
                    print(feature_expectation_difference.dtype)
                    print(weights.dtype)
                    print(feature_expectation.dtype)

                    dot_product = torch.dot(feature_expectation_difference, weights)

                    loss = -dot_product + log_prior

                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()

                    losses.append(loss.item())

            self.weights = weights.detach().numpy()
        return games_lengths, losses

if __name__ == "__main__":
    print('Running IRL FrozenLake')
    time_start = time.time()
    expert = frozenlake_agent.FrozenLakeQLearning(num_episodes=500)
    lengths = expert.run()
    print(f"Time to run: {time.time() - time_start}")
    print(f"Q_table: {expert.Q_table}")
    expert.draw_table()

    expert_feature_expectation = expert.get_feature_expectation(num_episodes=100)
    prior = 'laplacian'
    agent = IrlAgentBayesian(expert_feature_expectation, prior_type=prior)

    games_lengths, losses = agent.train_weights(N=12)

    time_end = time.time()

    # plot losses
    plt.figure()
    plt.plot(losses)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss vs Episode')
    plt.savefig(f'{prior}_irl_frozenlake_loss.png')
    plt.show()

    expert_reward = expert.reward_matrix
    agent_reward = agent.Q_table

    # Plot expert reward vs agent reward
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(expert_reward, cmap='winter', interpolation='nearest')
    plt.title('Expert Reward')
    plt.colorbar()

    for i in range(expert_reward.shape[0]):
        for j in range(expert_reward.shape[1]):
            plt.text(j, i, f'{expert_reward[i, j]:.2f}', ha='center', va='center', color='black')

    plt.subplot(1, 2, 2)
    plt.imshow(agent_reward, cmap='winter', interpolation='nearest')
    plt.title('Agent Reward')
    plt.colorbar()

    for i in range(agent_reward.shape[0]):
        for j in range(agent_reward.shape[1]):
            plt.text(j, i, f'{agent_reward[i, j]:.2f}', ha='center', va='center', color='black')

    plt.tight_layout()
    plt.savefig(f'{prior}_irl_frozenlake_reward.png')
    plt.show()
