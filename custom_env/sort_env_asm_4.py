import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import itertools


# an environment to learn sorting algorithms using conditional moves
# instructions:
# mov A B => copy value from register A to register B
# cmp A B => compare values in register A and B and set flags
# cmovg A B => copy value from register A to register B if greater flag is set
# cmovl A B => copy value from register A to register B if less flag is set

# we have n registers with values to be sorted and n additional swap registers

def apply(instr, a, b, state, nums, swap_count):
    less_flag_idx = nums + swap_count
    greater_flag_idx = less_flag_idx + 1
    if instr == "mov":
        state[b] = state[a]
    elif instr == "cmp":
        state[less_flag_idx] = 1 if state[a] < state[b] else 0
        state[greater_flag_idx] = 1 if state[a] > state[b] else 0
    elif instr == "cmovg":
        if state[greater_flag_idx] == 1:
            state[b] = state[a]
    elif instr == "cmovl":
        if state[less_flag_idx] == 1:
            state[b] = state[a]
    elif instr == "halt":
        pass
    elif instr == "nop":
        pass
    elif instr == "cswapg":
        if flags[greater_flag_idx] == 1:
            val_a = get_register(a)
            val_b = get_register(b)
            set_register(a, val_b)
            set_register(b, val_a)
    elif instr == "swap_gt":
        val_a = get_register(a)
        val_b = get_register(b)
        if val_a > val_b:
            set_register(a, val_b)
            set_register(b, val_a)
    else:
        raise ValueError("Unknown instruction: " + instr)

def state_to_string(state, nums, swap_count):
    total_registers = nums + swap_count
    numbers = ",".join(str(x) for x in state[:nums])
    swap_registers = ",".join(str(x) for x in state[nums:total_registers])
    flags = ""
    if state[total_registers]:
        flags += "<"
    if state[total_registers+1]:
        flags += ">"
                
    return f"{numbers} | {swap_registers} | {flags}"

class SortAsmEnv4(gym.Env):
    metadata = {"render_modes": ["human", "actions"]}

    def __init__(self, render_mode=None, 
        nums = 3, swap_registers = -1,
        early_termination=True, informed_reward=False, 
        num_tests = -1, max_episode_steps = None):

        if swap_registers == -1:
            swap_registers = nums

        self.nums = nums
        self.max_episode_steps = nums * nums if max_episode_steps is None else max_episode_steps
        self.early_termination = early_termination
        self.informed_reward = informed_reward
        self.num_tests = num_tests

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if num_tests == -1:
            # all permutations => all possible inputs (conceptually)
            self.tests = np.array(list(itertools.permutations(range(nums))))
            # print("Generated", len(self.tests), "tests")
            # print(self.tests)
        else:
            self.tests = None
        self.test_count = len(self.tests) if num_tests == -1 else num_tests


        ## State

        # state = sort registers, swap registers; each 0..n-1
        self.swap_register_count = swap_registers
        self.total_registers = nums + swap_registers
        self.flags = 2

        # ~~current code~~, current test states, swap registers, flags
        self.observation_space = spaces.Box(0, nums, shape=(self.test_count, self.total_registers + self.flags), dtype=int)
        # self.observation_space = spaces.Dict({
        #     "registers": spaces.Box(0, nums, shape=(self.nums,), dtype=int),
        #     "swap": spaces.Box(0, nums, shape=(self.nums,), dtype=int),
        #     "flags": spaces.Box(0, 1, shape=(self.flags,), dtype=int),
        # })

        
        ## Actions

        # all moves => binary instructions between all registers
        # self.actions = (
        #     list(
        #         itertools.product(
        #             ["mov", "cmp", "cmovg", "cmovl"],
        #             list(range(self.total_registers)),
        #             list(range(self.total_registers))
        #         )
        #     )
        # ) + [("nop", 0, 0)] + [("halt", 0, 0)]
        self.actions = \
            list(
                [(i,ab[0], ab[1])  for i, ab in
                # *mov* instructions not registered with themselves
                itertools.product(
                    ["mov", "cmovg", "cmovl"],
                    [(a,b) for a,b in itertools.product(range(self.total_registers), repeat=2) if a != b]
                )
                ]
            ) + \
            list(
                # cmp but only a<b
                [("cmp", a,b) for a,b in itertools.combinations(range(self.total_registers), 2)]
            ) + \
            [("nop", 0, 0)] + [("halt", 0, 0)] # halt and nop

        # print("Generated", len(self.actions), "actions", self.actions)
        # exit()


        # which instruction to execute
        self.action_space = spaces.Discrete(len(self.actions))


        ## Prepare Variables

        # keep state bundled together for more efficient algorithms, and test bundling
        self.state = None
        # self.registers = None
        # self.swap = None
        # self.flags = None
        self.steps = None


    def _get_obs(self):
        return self.state
        # return {
        #     "registers": self.registers,
        #     "swap": self.swap,
        #     "flags": self.flags,
        # }

    def _get_info(self):
        return {
            "steps": self.steps
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.num_tests != -1:
            self.tests = np.zeros((self.num_tests, self.nums), dtype=int)
            for i in range(self.num_tests):
                self.tests[i,:] = self.np_random.permutation(self.nums)
            
        self.dest = np.sort(self.tests, axis=1)
        self.steps = 0

        # init state 
        # all tests + swap registers at 0 + flags at 0
        self.state = np.zeros((self.test_count, self.total_registers + self.flags), dtype=int)
        self.state[:,:self.nums] = np.copy(self.tests)
        # self.registers = np.copy(self.tests)
        # self.swap = np.zeros((self.swap_register_count,), dtype=int)
        # self.flags = np.zeros((self.flags,), dtype=int)

        # for i in range(self.test_count):
        #     self.state[i,:self.nums] = self.np_random.permutation(self.nums)

        # print("Reset, starting state:", self.state)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" or self.render_mode == "actions":
            print()
            self._render_frame()

        return observation, info

    def _apply_action(self, action):
        instr,a,b = self.actions[action]
        # reversed for-loops would be faster
        for i in range(len(self.state)):
            apply(instr, a, b, self.state[i], self.nums, self.swap_register_count)
            # less_flag    = self.state[i,self.total_registers]
            # greater_flag = self.state[i,self.total_registers+1]
            # if instr == "mov":
            #     self.state[i,b] = self.state[i,a]
            # elif instr == "cmp":
            #     if self.state[i,a] < self.state[i,b]:
            #         less_flag = 1
            #         greater_flag = 0
            #     elif self.state[i,a] > self.state[i,b]:
            #         less_flag = 0
            #         greater_flag = 1
            #     else:
            #         less_flag = 0
            #         greater_flag = 0
            #     self.state[i,self.total_registers] = less_flag
            #     self.state[i,self.total_registers+1] = greater_flag
            # elif instr == "cmovg" and greater_flag == 1:
            #     self.state[i,b] = self.state[i,a]
            # elif instr == "cmovl" and less_flag == 1:
            #     self.state[i,b] = self.state[i,a]
        
    def _action_to_string(self, action):
        instr,a,b = self.actions[action]
        return f"{instr} ${a} ${b}"

    def step(self, action):
        instr,_,_ = self.actions[action]
        self._apply_action(action)
        self.steps += 1
        
        observation = self._get_obs()
        info = self._get_info()

        render = self.render_mode == "human" or self.render_mode == "actions"
        if render:
            print(self._action_to_string(action))
        if self.render_mode == "human":
            self._render_frame()


        give_reward = False
        all_sorted = True

        info_reward = 0
        # subtract points for duplicate values
        for i in range(len(self.state)):
            for j in range(self.nums):
                for k in range(j+1, self.nums):
                    if self.state[i,j] == self.state[i,k]:
                        info_reward -= 1
                        all_sorted = False
        # subtract points if place not the same across all tests
        for i in range(self.nums):
            unique_values = np.unique(self.state[:,i])
            if len(unique_values) > 1:
                info_reward -= 1*len(unique_values)
                all_sorted = False

        terminated = False
        truncated = False

        if all_sorted:
            info_reward += 1000
            give_reward = True
            terminated = True
            truncated = False
        elif self.early_termination and self.steps >= self.max_episode_steps:
            truncated = True
            give_reward = True
            terminated = False
        elif instr == "halt":
            give_reward = True
            terminated = True
            truncated = False

        reward = info_reward if give_reward else 0 # -1

        if render and (truncated or terminated):
            print("truncate" if truncated else "terminate")
            print(f"reward: {reward}"+(" (informed)" if self.informed_reward else ""))
            print(f"steps: {self.steps}")
            self._render_frame(force=True)

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def _render_frame(self, force=False):
        if self.render_mode == "human" or force:
            # print the current state
            states = []
            for i in range(len(self.state)):
                states.append(state_to_string(self.state[i], self.nums, self.swap_register_count))
                # numbers = ",".join(str(x) for x in self.state[i,:self.nums])
                # swap_registers = ",".join(str(x) for x in self.state[i,self.nums:self.total_registers])
                # flags = ""
                # if self.state[i,self.total_registers]:
                #     flags += "<"
                # if self.state[i,self.total_registers+1]:
                #     flags += ">"
                
                # states.append(f"{numbers} | {swap_registers} | {flags}")
            print("\n".join("  " + x for x in states))

    def close(self):
        pass


if __name__ == "__main__":
    # res,example=verifySort(sorter)
    # if not res:
    #     output = simulate_actions(sorter, example)
    #     print("Output:", output)

    # raise NotImplementedError("The verification code is not implemented yet")

    nums = 3
    swap = 1
    flags = 2
    total_registers = nums + swap
    tests = np.array(list(itertools.permutations(range(nums))))
    test_count = len(tests)
    dest = np.sort(tests, axis=1)
    state = np.zeros((test_count, total_registers + flags), dtype=int)
    state[:,:nums] = np.copy(tests)

    print("Starting state:")
    print("\n".join("  " + state_to_string(s, nums, swap) for s in state))

    # original sort (page 260)
    # P = 0
    # Q = 1
    # R = 2
    # S = 3
    # actions = [
    #     ("mov"  , R, S),
    #     ("cmp"  , P, R),
    #     ("cmovg", P, R), # R = max(A,C)
    #     ("cmovl", P, S), # S = min(A,C)
    #     ("mov"  , S, P), # P = min(A,C)
    #     ("cmp"  , S, Q),
    #     ("cmovg", Q, P), # P = min(A, B, C)
    #     ("cmovg", S, Q), # Q = max(min(A,C), B)
    # ]

    # sort 3 source (https://github.com/deepmind/alphadev/blob/main/sort_functions_test.cc)
    # R = 0
    # A = 1
    # C = 2
    # D = 3
    # actions = [
    #     ("cmp"  , A, C),
    #     ("mov"  , A, D),
    #     ("cmovl", C, D),
    #     ("cmovg", C, A),
    #     ("cmp"  , R, A),
    #     ("mov"  , R, C),
    #     ("cmovl", A, C),
    #     ("cmovl", R, A),
    #     ("cmp"  , C, D),
    #     ("cmovl", D, R),
    #     ("cmovg", D, C),
    # ]
    # results in 2,0,1

    X = 0
    Y = 1
    Z = 2
    S = 3
    actions = [
        # XYZ = ABC
        # Swap
        
        ("mov"  , Y, S), # S = B
        
        ("cmp"  , Y, Z), 
        ("cmovl", Z, S), # S = max(Y,Z) = max(B,C)
        ("cmovg", Z, Y), # Y = min(Y,Z) = min(B,C)
        
        
        ("cmp"  , X, Y), 
        ("mov"  , X, Z), # Z = A
        ("cmovl", Y, Z), # Z = max(X,Y) = max(A,min(B,C))
        ("cmovl", X, Y), # Y = min(X,Y) = min(A,min(B,C)) => global min
        
        ("cmp"  , Z, S), # max(A,min(B,C)) ? max(B,C)
        ("cmovl", S, X), # 
        ("cmovg", S, Z), 
        # Z = min
        # X = middle
        # Y = max
    ]

    for instr, a, b in actions:
        print("Executing", instr, a, b)
        for s in state:
            apply(instr, a, b, s, nums, swap)
        print("\n".join("  " + state_to_string(s, nums, swap) for s in state))
