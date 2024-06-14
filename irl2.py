from typing import Iterable

import agent as agents
import numpy
import torch


def get_expert_feature_expectations(expert=agents.QLearningAgent(), env=agents.Game(), N=100):
    expectation = numpy.zeros(expert.num_states)
    for _ in range(N):
        state = env.reset()
        done = False
        while not done:
            action = expert.get_action(state)
            state, reward, done, _ = env.step(action)
            expectation += expert.get_features(state)



class IRL:

    def __init__(self, expert_trajectories, env=agents.Game(), agent=agents.QLearningAgent()):
        self.expert_trajectories = expert_trajectories
        self.env = env
        self.agent = agent

    def get_feature_expectations(self, trajectories):
        feature_expectations = numpy.zeros(self.agent.feature_size)
        for trajectory in trajectories:
            for state, _, _, _ in trajectory:
                feature_expectations += self.agent.get_features(state)
        return feature_expectations / len(trajectories)
