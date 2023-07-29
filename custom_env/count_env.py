import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

class CountEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, 
        max_random=42, random_goal=False, 
        early_termination=False, reward_per_step=False, 
        choices=5):
        # -2^5 -2^4 ... 2^4 2^5
        self.choices = np.array([-(2**i) for i in range(choices, -1, -1)] + [2**i for i in range(0, choices+1, 1)], dtype=int)
        # print(self.choices)
        # self.choices = np.array([-1, 1])
        self.max_random = abs(max_random)
        self.bound = 4*self.max_random
        self.random_goal = random_goal
        self.early_termination = early_termination
        self.reward_per_step = reward_per_step
        self.max_episode_steps = 16*self.bound

        # the current number
        # self.observation_space = spaces.Box(-self.bound, self.bound, shape=(1,), dtype=int)
        # the current number and the goal
        self.observation_space = spaces.Box(-self.bound, self.bound, shape=(2,), dtype=int)
        # the difference between the current number and the goal
        # self.observation_space = spaces.Box(-self.bound, self.bound, shape=(1,), dtype=int)

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(len(self.choices))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.number = 0
        self.goal = max_random
        self.steps = 0

    def _get_obs(self):
        # return np.array([self.number])
        return np.array([self.number, self.goal])
        # return np.array([self.goal - self.number])

    def _get_info(self):
        return {
            "distance": self.goal - self.number,
            "steps": self.steps
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.goal = self.np_random.integers(-self.max_random, self.max_random+1) if self.random_goal else self.goal
        self.number = 0
        self.steps = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            print()
            self._render_frame()

        return observation, info

    def step(self, action):
        self.steps += 1
        num = self.choices[action]
        self.number += num
        truncated = False
        if self.number > self.bound:
            self.number = self.bound
            truncated = True
        elif self.number < -self.bound:
            self.number = -self.bound
            truncated = True

        # An episode is done iff the agent has reached the target
        terminated = self.number == self.goal
        reward = -1 if self.reward_per_step else 0
        if terminated:
            # reward=100 if self.reward_per_step else 1
            reward=1
        elif (self.early_termination and self.steps >= self.max_episode_steps) or truncated:
            # consider out of bounds to not reward going out of bounds as fast as possible
            # reward=-abs(self.goal-self.number) if self.reward_per_step else -1
            reward=-abs(self.goal-self.number) 
            truncated = True
        observation = self._get_obs()
        info = self._get_info()

        # could reward -1 per step
        # could reward diff at end (after k steps => truncate)

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def _render_frame(self):
        if self.render_mode == "human":
            print(f"Number: {self.number}, Goal: {self.goal} Distance: {self.goal - self.number}")

    def close(self):
        pass