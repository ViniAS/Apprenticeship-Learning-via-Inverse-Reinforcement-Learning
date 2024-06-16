import numpy as np
import math
import matplotlib.pyplot as plt
from time import time


class GridWorld:
    """
    A simple gridworld environment. The agent can move up, down, left, or right, and receives a reward for reaching
    a goal state. The goal states are randomly placed in the gridworld, and the reward is randomly set.
    """
    def __init__(self, size=128, macrocell_size=16, randomness=0.3):
        """
        Initialize the gridworld environment.
        :param size: length of one side of the gridworld
        :param macrocell_size: length of one side of a macrocell
        :param randomness: chance of moving in a random direction
        """
        self.size = size
        self.macrocell_size = macrocell_size
        self.num_macrocells = size // macrocell_size
        self.rewards = np.zeros(self.num_macrocells ** 2)
        self.curr_position = self.reset()
        self.set_reward()
        self.randomness = randomness
        done = False

    def reset(self):
        return 64, 64

    def get_reward(self):

        curr_macrocell = self.get_curr_macrocell()
        return self.rewards[curr_macrocell]

    def get_feature(self):
        """
        Get the feature vector for the current state.
        """
        curr_macrocell = self.get_curr_macrocell()
        feature = np.zeros(self.num_macrocells ** 2)
        feature[curr_macrocell] = 1
        return feature

    def move(self, action):
        """
        Move the agent in the gridworld. Does nothing if move would go out of bounds.
        Has a chance of moving in a random direction, depending on self.randomness.
        :param action: 0: up, 1: right, 2: down, 3: left
        :return: new position, reward, done
        """
        x, y = self.curr_position
        if np.random.rand() < self.randomness:
            action = np.random.randint(4)

        if action == 0 and y > 0:
            y -= 1
        elif action == 1 and x < self.size - 1:
            x += 1
        elif action == 2 and y < self.size - 1:
            y += 1
        elif action == 3 and x > 0:
            x -= 1

        self.curr_position = (x, y)
        reward = self.get_reward()
        done = False
        if reward > 0:
            done = True

        return action, self.curr_position, reward, done

    def get_curr_macrocell(self):
        x = self.curr_position[0] // self.macrocell_size
        y = self.curr_position[1] // self.macrocell_size
        curr_macrocell = self.size // self.macrocell_size * y + x

        return curr_macrocell

    def set_reward(self):
        """
        set the reward for the goal states.
        """
        for i in range(self.num_macrocells ** 2):
            if np.random.rand() < 0.1 and i != self.get_curr_macrocell():
                self.rewards[i] = np.random.rand()

        if np.count_nonzero(self.rewards) < 2:
            self.set_reward()

        self.rewards = self.rewards / np.sum(np.abs(self.rewards))


class GridWorldQLearning():
    def __init__(self, size=128, macrocell_size=16, randomness=0.3, num_episodes=10_000):
        self.env = GridWorld(size, macrocell_size, randomness)
        self.num_episodes = num_episodes
        self.Q_table = np.zeros((self.env.num_macrocells ** 2, 4))
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.1
        self.reward = 0

    def get_epsilon(self, t):
        return max(self.epsilon, min(1, 1.0 - math.log10((t + 1) / 25)))

    def get_alpha(self, t):
        return max(self.alpha, min(1.0, 1.0 - math.log10((t + 1) / 25)))

    def choose_action(self, state, epsilon):
        if np.random.random() <= epsilon:
            return np.random.choice([0, 1, 2, 3])  # action space
        else:
            return np.argmax(self.Q_table[state])

    def update_q(self, state_old, action, reward, state_new, alpha):
        self.Q_table[state_old][action] += alpha * (reward + self.gamma * np.max(self.Q_table[state_new])
                                                    - self.Q_table[state_old][action])

    def get_agent_reward(self):
        return self.reward

    def run(self):
        rewards = []
        for e in range(self.num_episodes):

            self.env.reset()
            current_state = self.env.get_curr_macrocell()
            alpha = self.get_alpha(e)
            epsilon = self.get_epsilon(e)
            done = False

            while not done:
                action = self.choose_action(current_state, epsilon)
                action, obs, self.reward, done = self.env.move(action)

                new_state = self.env.get_curr_macrocell()

                self.update_q(current_state, action, self.get_agent_reward(), new_state, alpha)
                current_state = new_state

            rewards.append(self.get_agent_reward())

        return rewards

    def play(self):
        self.env.reset()
        current_state = self.env.get_curr_macrocell()
        done = False
        while not done:
            action = self.choose_action(current_state, 0)
            action, obs, self.reward, done = self.env.move(action)
            new_state = self.env.get_curr_macrocell()
            current_state = new_state

        return self.get_agent_reward()

    def get_feature_expectation(self, gamma=0.99, num_episodes=1000):
        feature_expectation = np.zeros(self.env.num_macrocells ** 2)
        for _ in range(num_episodes):
            done = False
            self.env.reset()
            current_state = self.env.get_curr_macrocell()
            t = 0
            while not done:
                action = self.choose_action(current_state, 0)
                action, obs, reward, done = self.env.move(action)
                current_state = self.env.get_curr_macrocell()
                feature_expectation[current_state] += gamma ** t
                t += 1
        return feature_expectation / num_episodes


if __name__ == "__main__":
    agent = GridWorldQLearning()
    print(agent.env.rewards)
    time_start = time()
    rewards = agent.run()
    time_end = time()
    plt.plot(rewards)
    # print(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward over time. Run time: ' + str(time_end - time_start) + 's')

    plt.savefig('reward_plot_grid_world.png')



