import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import itertools


# an environment to learn sorting algorithms using conditional moves
# the only instruction is swap(a,b) which swaps the values at position a and b if a > b
# 3 => 3 instructions
# 5 => 9 instructions
# 6 => 12 instructions
# 10 => 110 instructions, 91

class SortEnv(gym.Env):
    metadata = {"render_modes": ["human", "actions"]}

    def __init__(self, render_mode=None, 
        nums = 3,
        early_termination=True, informed_reward=False, 
        num_tests = -1, max_episode_steps = None,
        choices=5):
        
        # swap any position with any other position without replacement => only (0,1) but not (1,0)
        # conceptual: swap(a,b) if a > b
        self.choices = np.array(list(itertools.combinations(range(nums), 2)))

        # all orderings of 1..n 
        self.tests = np.array(list(itertools.permutations(range(nums)))) if num_tests == -1 else None

        self.nums = nums
        self.max_episode_steps = nums * nums if max_episode_steps is None else max_episode_steps
        self.early_termination = early_termination
        self.informed_reward = informed_reward
        self.num_tests = num_tests
        self.test_count = len(self.tests) if num_tests == -1 else num_tests


        # which instruction to execute
        self.action_space = spaces.Discrete(len(self.choices))

        # (current code), current test states
        self.observation_space = spaces.Box(0, nums, shape=(self.test_count, nums), dtype=int)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.state = None
        self.steps = None


    def _get_obs(self):
        return self.state

    def _get_info(self):
        return {
            "steps": self.steps
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.num_tests == -1:
            self.state = np.copy(self.tests)
        else:
            # num_tests random permutations
            self.state = np.zeros((self.num_tests, self.nums), dtype=int)
            for i in range(self.num_tests):
                self.state[i,:] = self.np_random.permutation(self.nums)
        self.dest = np.sort(self.state, axis=1)
        self.steps = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" or self.render_mode == "actions":
            print()
            self._render_frame()

        return observation, info

    def _apply_action(self, action):
        a,b = self.choices[action]
        # swap a,b if a > b in all tests
        # semantics of the action
        for i in range(len(self.state)):
            if self.state[i,a] > self.state[i,b]:
                self.state[i,a], self.state[i,b] = self.state[i,b], self.state[i,a]
        
    def _action_to_string(self, action):
        a,b = self.choices[action]
        return f"swap({a},{b})"

    def step(self, action):
        self._apply_action(action)
        
        observation = self._get_obs()
        info = self._get_info()

        self.steps += 1

        all_sorted = True
        for i in range(len(self.state)):
            if not np.all(self.state[i,:] == self.dest[i,:]):
                all_sorted = False
                break

        # reward = 0
        # reward = -0.1 # reward faster
        reward = -1 # reward faster
        terminated = False
        truncated = False

        if all_sorted:
            # reward = 1
            reward = 10
            terminated = True
            # maybe use higher reward 
        elif self.early_termination and self.steps >= self.max_episode_steps:
            truncated = True
            reward = -1
            # reward to partial sorting
            if self.informed_reward:
                for i in range(len(self.state)):
                    for j in range(self.nums-1):
                        if self.state[i,j] > self.state[i,j+1]:
                            reward -= 1

        render = self.render_mode == "human" or self.render_mode == "actions"
        if render:
            print(self._action_to_string(action))
        if self.render_mode == "human":
            self._render_frame()
        if render and (truncated or terminated):
            print(f"reward: {reward}")
            print(f"steps: {self.steps}")

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def _render_frame(self):
        if self.render_mode == "human":
            # print the current state
            print(" ".join(",".join(str(x) for x in row) for row in self.state))

    def close(self):
        pass


def extract_actions(actions):
    if isinstance(actions, str):
        actions = actions.split("\n")
        actions = [
            x.split("swap(")[1].split(")")[0].split(",")
            for x in actions
        ]
        actions = [(int(a), int(b)) for a,b in actions]
    return actions

def simulate_actions(actions, nums):
    actions = extract_actions(actions)
    state = np.copy(nums)
    for a,b in actions:
        if state[a] > state[b]:
            state[a], state[b] = state[b], state[a]
    return state

from z3 import *
def verifySort(actions):
    actions = extract_actions(actions)
    count = max(max(a,b) for a,b in actions)+1
    steps = len(actions)
    s = Solver()

    # variables
    orig_nums = [Int(f"num_{i}_0") for i in range(count)]
    nums = orig_nums
    # simulate the swaps
    for i,ab in enumerate(actions):
        a,b = ab
        # if a > b, a_i+1 = b_i, b_i+1 = a_i
        # all other nums stay the same
        new_nums = [Int(f"num_{i}_{j+1}") for j in range(count)]
        condition = nums[a] > nums[b]
        s.add(If(condition, new_nums[a] == nums[b], new_nums[a] == nums[a]))
        s.add(If(condition, new_nums[b] == nums[a], new_nums[b] == nums[b]))
        for j in range(count):
            if j == a or j == b:
                continue
            s.add(new_nums[j] == nums[j])
            # if j == a:
            #     s.add(If(condition, new_nums[j] == nums[b], new_nums[j] == nums[a]))
            # elif j == b:
            #     s.add(If(condition, new_nums[j] == nums[a], new_nums[j] == nums[b]))
            # else:
            #     s.add(new_nums[j] == nums[j])
        nums = new_nums
    
    # check if all nums are sorted
    all_sorted = []
    for i in range(count-1):
        all_sorted.append(nums[i] <= nums[i+1])
    
    s.add(Not(And(all_sorted)))

    # print(s.sexpr())

    if s.check() == sat:
        m = s.model()
        print("Unsorted")
        counter_example = [m.evaluate(orig_nums[i]).as_long() for i in range(count)]
        print(" ".join(str(x) for x in counter_example))
        return (False, counter_example)
    else:
        print(f"All {count} elements sorted in {steps} steps")
        return (True, None)

if __name__ == "__main__":
    # sorter = """
    # swap(0,1)
    # swap(1,2)
    # swap(0,1)
    # """.strip()

    # sorter = """
    # swap(0,1)
    # swap(0,2)
    # swap(1,2)
    # """.strip()

#     sorter = """
# swap(1,2)
# swap(0,3)
# swap(2,3)
# swap(3,4)
# swap(0,1)
# swap(1,2)
# swap(2,3)
# swap(1,2)
# swap(0,1)
#     """.strip()

#     sorter = """
# swap(1,4)
# swap(0,3)
# swap(2,5)
# swap(0,1)
# swap(1,2)
# swap(0,1)
# swap(3,4)
# swap(4,5)
# swap(2,3)
# swap(3,4)
# swap(2,3)
# swap(1,2)
#     """.strip()

    sorter = """
   swap(6,7)
swap(2,3)
swap(4,5)
swap(1,2)
swap(8,9)
swap(6,7)
swap(4,5)
swap(4,7)
swap(1,2)
swap(6,7)
swap(4,6)
swap(5,8)
swap(7,8)
swap(1,9)
swap(1,2)
swap(8,9)
swap(7,8)
swap(5,6)
swap(2,3)
swap(4,5)
swap(3,4)
swap(6,7)
swap(8,9)
swap(7,8)
swap(0,1)
swap(3,5)
swap(1,2)
swap(1,3)
swap(2,3)
swap(5,6)
swap(5,6)
swap(5,6)
swap(8,9)
swap(4,5)
swap(6,7)
swap(1,2)
swap(8,9)
swap(8,9)
swap(6,9)
swap(5,6)
swap(5,6)
swap(2,3)
swap(6,7)
swap(0,1)
swap(2,4)
swap(7,8)
swap(8,9)
swap(7,8)
swap(0,1)
swap(1,3)
swap(3,4)
swap(5,8)
swap(4,5)
swap(6,7)
swap(7,8)
swap(5,6)
swap(3,4)
swap(3,5)
swap(6,7)
swap(2,3)
swap(1,2)
swap(5,6)
swap(2,3)
swap(8,9)
swap(4,5)
swap(1,2)
swap(1,2)
swap(7,8)
swap(1,2)
swap(3,4)
swap(6,9)
swap(7,8)
swap(6,7)
swap(2,3)
swap(2,3)
swap(2,3)
swap(0,1)
swap(5,6)
swap(2,3)
swap(3,4)
swap(3,6)
swap(4,5)
swap(3,5)
swap(6,7)
swap(2,3)
swap(1,2)
swap(4,9)
swap(4,5)
swap(3,6)
swap(2,3)
swap(8,9) 
    """.strip()

    res,example=verifySort(sorter)
    if not res:
        output = simulate_actions(sorter, example)
        print("Output:", output)