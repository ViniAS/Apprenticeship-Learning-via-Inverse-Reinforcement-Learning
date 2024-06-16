import numpy as np
import math
import matplotlib.pyplot as plt
from time import time
import gridworld


class GridWorldIRL(gridworld.GridWorldQLearning):
    def __init__(self, expert_feature_expectation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_feature_expectation = expert_feature_expectation
        self.weights = np.random.rand(self.num_macrocells ** 2)
        self.feature_expectations = []
        self.feature_expectations_bar = []
        
    def get_reward(self):
        return np.dot(self.weights, self.get_feature())
    
    def train_weights(self, N=20):
        self.Q_table = np.zeros((self.num_macrocells ** 2, 4))
        self.feature_expectations.append(self.get_feature_expectation())
        
        self.feature_expectations_bar.append(self.feature_expectations[0])
        self.weights = self.expert_feature_expectation - self.feature_expectations_bar[0]
        rewards = []
        
        for i in range(1, N):
            self.Q_table = np.zeros((self.num_macrocells ** 2, 4))
            rewards.append(self.run())
            self.feature_expectations.append(self.get_feature_expectation())
            A = self.feature_expectations[i] - self.feature_expectations_bar[i-1]
            B = self.expert_feature_expectation - self.feature_expectations_bar[i-1]
            projection = (np.dot(A, B) / np.dot(A, A)) * A
            self.feature_expectations_bar.append(self.feature_expectations_bar[i-1] + projection)
            self.weights = self.expert_feature_expectation - self.feature_expectations_bar[i]
            if np.linalg.norm(self.weights) < 1e-5:
                break
                
        return rewards
    
if __name__ == "__main__":
    # get time to run
    time_start = time()
    expert = gridworld.GridWorldQLearning()
    rewards = expert.run()
    expert_feature_expectation = expert.get_feature_expectation()
    agent = GridWorldIRL(expert_feature_expectation)
    rewards_irl = agent.train_weights()
    time_end = time()
    distance_feature_expectations = []
    for i in range(len(agent.feature_expectations_bar)):
        distance_feature_expectations.append(np.linalg.norm(agent.feature_expectations_bar[i] -
                                                            expert_feature_expectation))
    print("weights: ", agent.weights)
    print("weights expert: ", expert.rewards)

    plt.plot(distance_feature_expectations, label='Distance to expert')
    plt.savefig('irl_gridworld.png')
