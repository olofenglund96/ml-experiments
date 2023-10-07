import numpy as np

class QLearning:
    def __init__(self, dimensions, actions, bin_size, alpha = 0.1, gamma = 0.99, epsilon = 0.2):
        self._bins = self._create_bins(dimensions, bin_size)
        self._q = self._create_q_table(len(dimensions), len(actions), bin_size)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def _create_bins(self, dimensions, bin_size):
        bins = []
        for dim in dimensions:
            bins.append(np.linspace(dim[0], dim[1], bin_size))

        return bins

    def _create_q_table(self, state_space, action_space, bin_size):
        return np.random.uniform(low=-1,high=1,size=([bin_size] * state_space + [action_space]))

    def bin(self, observation):
        indices = []
        print(observation, self._bins)
        for i, obs in enumerate(observation):
            indices.append(np.digitize(obs, self._bins[i]) - 1)

        return tuple(indices)

    def get_action(self, observation, samplefn):
        if np.random.uniform(0,1) < self.epsilon:
            return samplefn()
        
        return np.argmax(self._q[self.bin(observation)])

    def update_q(self, old_observation, new_observation, reward, action):
        old_obs_bins = self.bin(old_observation)
        new_obs_bins = self.bin(new_observation)
        self._q[old_obs_bins + (action,)] = (1 - self.alpha)*self._q[old_obs_bins + (action,)] + self.alpha * (reward + self.gamma * np.max(self._q[new_obs_bins]))