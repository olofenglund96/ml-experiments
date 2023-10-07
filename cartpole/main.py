import gymnasium as gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
from algos.rl import QLearning
env = gym.make("CartPole-v1", render_mode="human")
#env = gym.make("CartPole-v1")

observation, info = env.reset(seed=42)

bin_size = 40
alpha = 0.15
gamma = 0.995
epsilon = 0.1

dimensions = [(-4.8,4.8), (-4,4), (-0.418,0.418), (-4,4)]
actions = (0, 1)

ql = QLearning(dimensions, actions, bin_size, alpha, gamma, epsilon)
ix = 0
tot_reward = 0
last_state = 0
state_visits = defaultdict(int)
rewards = []
new_states = []

for _ in tqdm(range(1000)):
    while True:
        observation_bins = ql.bin(observation)
        state_visits[observation_bins] += 1
        
        action = ql.get_action(observation, env.action_space.sample)

        new_obs, reward, terminated, truncated, info = env.step(action)
        tot_reward += reward

        ql.update_q(observation, new_obs, reward, action)
        observation = new_obs

        if terminated or truncated:
            ix += 1
            rewards.append(tot_reward)
            tot_reward = 0
            new_states.append(len(state_visits.keys()) - last_state)
            last_state = len(state_visits.keys())
            observation, info = env.reset()
            break
env.close()

#q.dump("q.dat")

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

x = list(range(ix))
print(x, rewards)
plt.figure()
plt.plot(x, moving_average(rewards, 500), moving_average(new_states, 500))
plt.ylim(bottom=0)
plt.savefig("reward.png")
