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

def apply(instr, a, b, registers, swap_registers, flags):
    less_flag_idx = 0
    greater_flag_idx = 1

    def set_register(i,v):
        if i >= len(registers):
            swap_registers[i-len(registers)] = v
        else:
            registers[i] = v
    
    def get_register(i):
        if i >= len(registers):
            return swap_registers[i-len(registers)]
        else:
            return registers[i]
    
    
    if instr == "mov":
        set_register(b, get_register(a))
    elif instr == "cmp":
        flags[less_flag_idx] = 1 if get_register(a) < get_register(b) else 0
        flags[greater_flag_idx] = 1 if get_register(a) > get_register(b) else 0
    elif instr == "cmovg":
        if flags[greater_flag_idx] == 1:
            set_register(b, get_register(a))
    elif instr == "cmovl":
        if flags[less_flag_idx] == 1:
            set_register(b, get_register(a))
    elif instr == "halt":
        pass
    elif instr == "nop":
        pass
    elif instr == "cswapgt":
        if flags[greater_flag_idx] == 1:
            val_a = get_register(a)
            val_b = get_register(b)
            set_register(a, val_b)
            set_register(b, val_a)
    elif instr == "swapgt":
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

class SortAsmEnv5(gym.Env):
    metadata = {"render_modes": ["human", "actions"]}

    def __init__(self, render_mode=None, 
        nums = 3, swap_registers = -1,
        # early_termination=True, informed_reward=False, 
        num_tests = -1, max_episode_steps = None):

        if swap_registers == -1:
            swap_registers = nums

        self.nums = nums
        self.max_episode_steps = nums * nums if max_episode_steps is None else max_episode_steps
        # self.early_termination = early_termination
        # self.informed_reward = informed_reward
        self.num_tests = num_tests

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if num_tests == -1:
            # all permutations => all possible inputs (conceptually)
            self.tests = np.array(list(itertools.permutations(range(nums))))
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

        
        ## Actions
        
        use_cmp = False
        use_cmov = False
        use_mov = False
        use_cswap = False
        use_swapgt = False
        use_halt = False
        use_nop = False
        
        use_cmp = True
        # use_cmov = True
        # use_mov = True
        use_cswap = True
        # use_swapgt = True
        # use_halt = True
        # use_nop = True
        
        self.actions = []
        if use_halt:
            self.actions += [("halt", 0, 0)]
        if use_nop:
            self.actions += [("nop", 0, 0)]
        
        # order independent
        # self.actions += [("cmp", a,b) for a,b in itertools.combinations(range(self.total_registers), 2)]
        self.actions += [
                (i,ab[0], ab[1])  for i, ab in
                itertools.product(
                    (["cmp"] if use_cmp else [])+
                    (["cswapgt"] if use_cswap else []) +
                    (["swapgt"] if use_swapgt else []) +
                    []
                    ,
                    [(a,b) for a,b in itertools.combinations(range(self.total_registers), 2)]
                )
        ]
        
        # order dependent between all "real" registers
        self.actions += [
                (i,ab[0], ab[1])  for i, ab in
                itertools.product(
                    (["mov"] if use_mov else []) +
                    (["cmovg", "cmovl"] if use_cmov else [])+
                    []
                    ,
                    [(a,b) for a,b in itertools.product(range(self.total_registers), repeat=2) if a != b]
                )
        ]
        
        
        
        print("Generated", len(self.actions), "actions", self.actions)
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
            # or sample n permutations
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
            # apply(instr, a, b, self.state[i], self.nums, self.swap_register_count)
            state = self.state[i]
            apply(instr, a, b, state[:self.nums], state[self.nums:self.total_registers], state[self.total_registers:])
        
    def _action_to_string(self, action):
        instr,a,b = self.actions[action]
        return f"{instr} ${a} ${b}"

    def step(self, action):
        self.early_termination = True
        self.informed_reward = False
        
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

        # reward = info_reward if give_reward else 0 # -1
        reward = info_reward if give_reward else -1

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

    raise NotImplementedError("The verification code is not implemented yet")