import gymnasium as gym
import numpy as np
import math

class CartPoleQLearning:
    def __init__(self, buckets=(1, 1, 6, 12), num_episodes=500, min_lr=0.1, min_epsilon=0.1, discount=1.0, decay=25):
        self.buckets = buckets
        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.decay = decay

        self.env = gym.make('CartPole-v1')

        self.Q_table = np.zeros(self.buckets + (self.env.action_space.n,))

    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q_table[state])

    def update_q(self, state_old, action, reward, state_new, alpha):
        self.Q_table[state_old][action] += alpha * (reward + self.discount * np.max(self.Q_table[state_new]) - self.Q_table[state_old][action])

    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))

    def get_alpha(self, t):
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))

    def run(self):
        for e in range(self.num_episodes):
            current_state = self.discretize(self.env.reset()[0])

            alpha = self.get_alpha(e)
            epsilon = self.get_epsilon(e)

            done = False
            while not done:
                action = self.choose_action(current_state, epsilon)
                obs, reward, done, _, _ = self.env.step(action)
                new_state = self.discretize(obs)

                self.update_q(current_state, action, reward, new_state, alpha)
                current_state = new_state

        print('Finished training!')

    def play(self):
        self.env = gym.make('CartPole-v1', render_mode="rgb_array")
        t = 0
        done = False
        current_state = self.discretize(self.env.reset()[0])
        while not done:
            self.env.render()
            t = t+1
            action = self.choose_action(current_state, 0)
            obs, reward, done, _, _ = self.env.step(action)
            new_state = self.discretize(obs)
            current_state = new_state

        return t


if __name__ == '__main__':
    agent = CartPoleQLearning()
    agent.run()
    agent.play()