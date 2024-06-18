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

class IrlAgentBayesian(frozenlake_agent.FrozenLakeQLearning):
    def __init__(self, expert_feature_expectation, prior_type='gaussian', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_feature_expectation = torch.tensor(expert_feature_expectation, dtype=torch.float32)
        self.prior_type = prior_type
        self.weights = np.random.rand(16)
        self.feature_expectations = []
        self.feature_expectations_bar = []

    def get_reward(self):
        reward = self.weights[self.obs]
        return reward
    
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
            log_prior = prior.log_prob(weights).sum()
            feature_expectation = torch.tensor(self.feature_expectations[-1], dtype=torch.float32)
            feature_diff = feature_expectation - self.expert_feature_expectation
            loss = feature_diff @ weights - log_prior
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print("loss.item(): ", loss.item())

        self.weights = weights.detach().numpy()
        return games_lengths, losses


class IrlAgentMaxMargin(frozenlake_agent.FrozenLakeQLearning):
    def __init__(self, expert_feature_expectation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_feature_expectation = expert_feature_expectation
        self.weights = np.random.rand(16)
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
        print(f"weights {self.weights}")
        rewards = []
        losses = []
        for i in range(1,N):
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
            losses.append(t.value)
            if t.value and t.value < 1e-5:
                break
            print(t.value)

        return rewards, losses


class IrlAgentProjection(frozenlake_agent.FrozenLakeQLearning):
    def __init__(self, expert_feature_expectation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_feature_expectation = expert_feature_expectation
        self.weights = np.random.rand(16)
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
        print(f"weights {self.weights}")
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
            losses.append(np.linalg.norm(self.weights))
            if np.linalg.norm(self.weights) < 1e-5:
                break

        return rewards, losses



if __name__ == "__main__":
    print('Running IRL FrozenLake')
    time_start = time.time()
    expert = frozenlake_agent.FrozenLakeQLearning(num_episodes=5000)
    lengths = expert.run()
    print(f"Time to run: {time.time() - time_start}")
    print(f"Q_table: {expert.Q_table}")
    expert.draw_table()

    expert_feature_expectation = expert.get_feature_expectation(num_episodes=1000)

    agent = IrlAgentProjection(expert_feature_expectation, num_episodes=5000)
    agent_max_margin = IrlAgentMaxMargin(expert_feature_expectation, num_episodes=5000)

    rewards_max_margin, losses_max_margin = agent_max_margin.train_weights(N=5000)
    rewards, losses = agent.train_weights(N=5000)
    time_end = time.time()

    plt.plot(rewards_max_margin[0], label="first run")
    plt.ylabel("reward")
    plt.xlabel("Episode")
    plt.title("FrozenLake: Max Margin IRL")
    plt.legend()

    plt.savefig("frozenlake_max_margin_first.png")

    plt.figure()
    plt.plot(rewards_max_margin[-1], label = f"last run ({len(rewards)}")
    plt.ylabel("reward")
    plt.xlabel("Episode")
    plt.title("FrozenLake: Max Margin IRL")
    plt.legend()

    plt.savefig("frozenlake_max_margin.png")

    plt.figure()
    plt.plot(lengths)
    plt.ylabel("reward")
    plt.xlabel("Episode")
    plt.title("FrozenLake: Expert")

    plt.savefig("frozenlake_expert.png")

    plt.figure()
    plt.plot(rewards[0], label="first run")
    plt.ylabel("reward")
    plt.xlabel("Episode")
    plt.title("FrozenLake: Projection IRL")
    plt.legend()

    plt.savefig("frozenlake_projection_first.png")

    plt.figure()
    plt.plot(rewards[-1], label = f"last run ({len(rewards)}")
    plt.ylabel("reward")
    plt.xlabel("Episode")
    plt.title("FrozenLake: Projection IRL")
    plt.legend()

    plt.savefig("frozenlake_projection.png")

    plt.figure()
    plt.plot(losses_max_margin)
    plt.ylabel("t")
    plt.xlabel("Iteration")
    plt.title("FrozenLake: Max Margin IRL")

    plt.savefig("frozenlake_max_margin_loss.png")

    plt.figure()
    plt.plot(losses)
    plt.ylabel("t")
    plt.xlabel("Iteration")
    plt.title("FrozenLake: Projection IRL")

    plt.savefig("frozenlake_projection_loss.png")

    print(f"Time to run: {time_end - time_start}")

    plt.figure()
    sns.heatmap(agent.weights.reshape(4, 4), annot=True)


    plt.title("FrozenLake: Projection IRL")

    plt.savefig("frozenlake_projection_heatmap.png")

    plt.figure()
    sns.heatmap(agent_max_margin.weights.reshape(4,4), annot=True)
    plt.title("FrozenLake: Max Margin IRL")

    plt.savefig("frozenlake_max_margin_heatmap.png")

    plt.figure()
    sns.heatmap(expert.Q_table, annot=True)
    plt.title("FrozenLake: Expert Q-table")

    plt.savefig("frozenlake_expert_heatmap.png")

    plt.figure()
    sns.heatmap(agent.Q_table, annot=True)
    plt.title("FrozenLake: Projection Q-table")

    plt.savefig("frozenlake_projection_qtable.png")

    plt.figure()
    sns.heatmap(agent_max_margin.Q_table, annot=True)
    plt.title("FrozenLake: Max Margin Q-table")

    plt.savefig("frozenlake_max_margin_qtable.png")



    

    agent.draw_table()

    plt.show()








