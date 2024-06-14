import gymnasium as gym
import numpy as np
import cartpole_agent


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

    def train_weights(self, N = 1000):
        self.feature_expectations.append(self.get_feature_expectation())
        self.feature_expectations_bar.append(self.feature_expectations[0])
        self.weights = self.expert_feature_expectation - self.feature_expectations_bar[0]

        for i in range(2, len(self.feature_expectations)):
            self.Q_table = np.zeros(self.buckets + (self.env.action_space.n,))
            self.run()
            self.feature_expectations.append(self.get_feature_expectation())
            A = self.feature_expectations[i]-self.feature_expectations_bar[i-1]
            B = self.expert_feature_expectation-self.feature_expectations_bar[i-1]
            projection = (np.dot(A, B)/np.dot(A, A))*A
            self.feature_expectations_bar.append(self.feature_expectations_bar[i-1] + projection)
            self.weights = self.expert_feature_expectation - self.feature_expectations_bar[i]
            if np.linalg.norm(self.weights) < 1e-5:
                break

        return self.weights

if __name__ == '__main__':
    expert = cartpole_agent.CartPoleQLearning()
    expert.run()
    expert_feature_expectation = expert.get_feature_expectation()
    agent = IrlAgent(expert_feature_expectation)
    agent.train_weights()
    print(agent.get_reward())
    print(agent.weights)
    agent.play()









