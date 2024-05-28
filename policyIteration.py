import numpy as np


def policy_evaluation(policy, rewards, states, actions, gamma=0.99, theta=1e-9):
    V = np.zeros(states)
    while True:
        delta = 0
        for s in range(states):
            v = V[s]
            V[s] = sum([policy[s, a] * (rewards[s, a] + gamma * sum([p * V[s_] for p, s_ in actions[s, a]])) for a in range(actions.shape[1])])
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V


def policy_improvement(rewards, states, actions, gamma=0.99):
    policy = np.ones((states, actions.shape[1])) / actions.shape[1]
    while True:
        V = policy_evaluation(policy, rewards, states, actions, gamma)
        policy_stable = True
        for s in range(states):
            chosen_a = np.argmax(policy[s])
            action_values = np.zeros(actions.shape[1])
            for a in range(actions.shape[1]):
                action_values[a] = sum([p * (rewards[s, a] + gamma * V[s_]) for p, s_ in actions[s, a]])
            best_a = np.argmax(action_values)
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(actions.shape[1])[best_a]
        if policy_stable:
            return policy, V


