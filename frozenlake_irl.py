import gymnasium as gym
import numpy as np
import frozenlake_agent
import matplotlib.pyplot as plt
import time
import torch
from torch.distributions import Normal
from torch_sgld import SGLD

class IrlAgent(frozenlake_agent.FrozenLakeQLearning):
    def __init__(self, expert_feature_expectation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_feature_expectation = expert_feature_expectation
        self.weights = np.random.rand(16)
        self.feature_expectations = []
        self.feature_expectations_bar = []

    def get_reward(self):
        obs = 1/(1 + np.exp(-self.obs))
        return np.dot(self.weights, obs)
    
    def train_weights(self, N = 10):
        self.Q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.feature_expectations.append(self.get_feature_expectation())
        
        self.feature_expectations_bar.append(self.feature_expectations[0])
        self.weights = self.expert_feature_expectation - self.feature_expectations_bar[0]
        games_lengths = []
        for i in range(1, N):
            self.Q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
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
    
class IrlAgentBayesian(frozenlake_agent.FrozenLakeQLearning):
    def __init__(self, expert_feature_expectation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_feature_expectation = torch.tensor(expert_feature_expectation, dtype=torch.float32)
        self.weights = np.random.rand(16)
        self.feature_expectations = []
        self.feature_expectations_bar = []

    def get_reward(self):
        obs = 1/(1 + np.exp(-self.obs))
        return np.dot(self.weights, obs)
    
    def train_weights(self, N = 10, lr = 1e-2, prior: torch.distributions.Distribution = Normal(0, 1)):
        self.Q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.feature_expectations.append(self.get_feature_expectation())
        
        weights = prior.sample((16,))
        weights.requires_grad = True
        self.weights = weights.detach().numpy()
        optimizer = SGLD([weights], lr=lr)
        games_lengths = []
        losses = []

        print("Training weights...")
        while len(games_lengths) < N or losses[-1] > 1e-5:
            self.Q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
            games_lengths.append(self.run())
            self.feature_expectations.append(self.get_feature_expectation())
            loss = torch.norm(weights - self.expert_feature_expectation)**2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print("loss.item(): ", loss.item())

        return games_lengths, losses
    
if __name__ == "__main__":
    # get time to run
    print('Running IRL FrozenLake')
    time_start = time.time()
    expert = frozenlake_agent.FrozenLakeQLearning(num_episodes=5000)
    lengths = expert.run()
    print(f"Time to run: {time.time() - time_start}")
    print(f"Q_table: {expert.Q_table}")
    expert.draw_table()

    expert_feature_expectation = expert.get_feature_expectation(num_episodes=100)
    agent = IrlAgentBayesian(expert_feature_expectation)

    games_lengths, losses = agent.train_weights(N=500)

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
    plt.savefig('irl_frozenlake_rewards.png')

    # plot losses

    plt.figure()
    plt.plot(losses)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss vs Episode')
    plt.savefig('irl_frozenlake_loss.png')











