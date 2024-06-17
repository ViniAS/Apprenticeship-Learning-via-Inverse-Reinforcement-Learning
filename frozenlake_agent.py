import gymnasium as gym
import numpy as np
import math
import time

class FrozenLakeQLearning:
    def __init__(self, num_episodes=500, min_lr=0.1, min_epsilon=0.1, discount=1.0, decay=25, render=True):
        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.decay = decay
        self.reward = 0
        self.obs = None

        if render:
            self.env = gym.make('FrozenLake-v1', render_mode='human')
        else:
            self.env = gym.make('FrozenLake-v1')

        # This is the action-value function being initialized to 0
        self.Q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
    
    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q_table[state])
    
    def update_q(self, state_old, action, reward, state_new, alpha):
        self.Q_table[state_old][action] += alpha * (reward + self.discount * np.max(self.Q_table[state_new]) -
                                                    self.Q_table[state_old][action])
    
    def get_epsilon(self, t):
        return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))
    
    def get_alpha(self, t):
        return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))
    
    def run(self, render=True):
        total_reward = [0] * self.num_episodes
        for t in range(self.num_episodes):
            current_state = self.env.reset()[0]
            state = current_state
            done = False
            while not done:
                epsilon = self.get_epsilon(t)
                action = self.choose_action(state, epsilon)
                obs, reward, done, b, prob = self.env.step(action)
                new_state = obs
                if done and reward == 0:
                    reward = -1  # Negative reward for falling into a hole
                # Penalize if movement hits the edge of the game
                #if new_state == state:
                #    reward -= 0.5
                alpha = self.get_alpha(t)
                self.update_q(state, action, reward, new_state, alpha)
                state = new_state
                total_reward[t] += reward
                if done:
                    break
        return total_reward
    
    def play(self, render=True):
        state = self.env.reset()[0]
        if render:
            self.env.render()
        done = False
        while not done:
            action = np.argmax(self.Q_table[state])
            obs, reward, done, b, prob= self.env.step(action)
            state = obs
            if render:
                self.env.render()
                time.sleep(0.5)
        self.env.close()
        return reward
    
    def draw_table(self):
        table = np.zeros((4, 4), dtype=int)
        for i in range(4):
            for j in range(4):
                table[i][j] = np.argmax(self.Q_table[i * 4 + j])
        print(table)
    

if __name__ == "__main__":
    start = time.time()
    agent = FrozenLakeQLearning(num_episodes=500, render=False)
    total_reward = agent.run()
    reward = agent.play(render=False)
    print(f"Time to run: {time.time() - start}")
    print(f"Q_table: {agent.Q_table}")
    print(f"Reward: {reward}")
    agent.draw_table()

    m = 10
    for i in range(m):
        reward = agent.play(render=False)
        print(f"Reward {i+1}: {reward}")
