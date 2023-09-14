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
    elif instr == "swap_cmpg":
        if flags[greater_flag_idx] == 1:
            val_a = get_register(a)
            val_b = get_register(b)
            set_register(a, val_b)
            set_register(b, val_a)
    else:
        raise ValueError("Unknown instruction: " + instr)

def state_to_string(register, swap_register, flags):
    numbers = ",".join(str(x) for x in register)
    swap_registers = ",".join(str(x) for x in swap_register)
    flag_str = ""
    if flags[0]:
        flag_str += "<"
    if flags[1]:
        flag_str += ">"
    
    return f"{numbers} | {swap_registers} | {flag_str}"

class SortAsmEnv3(gym.Env):
    metadata = {"render_modes": ["human", "actions"]}

    def __init__(self, render_mode=None, 
        nums = 3, swap_registers = -1,
        early_termination=True, informed_reward=False, 
        max_episode_steps = None, secret_case_count=-1):

        if swap_registers == -1:
            swap_registers = nums

        self.nums = nums
        self.max_episode_steps = nums * nums if max_episode_steps is None else max_episode_steps
        self.early_termination = early_termination
        self.informed_reward = informed_reward
        self.secret_case_count = secret_case_count

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        ## State

        # state = sort registers, swap registers; each 0..n-1
        self.swap_register_count = swap_registers
        self.total_registers = nums + swap_registers
        self.flag_count = 2

        # ~~current code~~, current test states, swap registers, flags
        self.observation_space = spaces.Dict({
            "registers": spaces.Box(0, nums, shape=(self.nums,), dtype=int),
            "swap": spaces.Box(0, nums, shape=(self.swap_register_count,), dtype=int),
            "flags": spaces.Box(0, 1, shape=(self.flag_count,), dtype=int),
        })

        
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
            list(
                # cmp but only a<b
                [("swap_cmpg", a,b) for a,b in itertools.combinations(range(self.total_registers), 2)]
            ) + \
            [("nop", 0, 0)] + [("halt", 0, 0)] # halt and nop

        # print("Generated", len(self.actions), "actions", self.actions)
        # exit()


        # which instruction to execute
        self.action_space = spaces.Discrete(len(self.actions))


        ## Prepare Variables

        # keep state bundled together for more efficient algorithms, and test bundling
        self.registers = None
        self.swap = None
        self.flags = None
        self.steps = None


    def _get_obs(self):
        return {
            "registers": self.registers,
            "swap": self.swap,
            "flags": self.flags,
        }

    def _get_info(self):
        return {
            "steps": self.steps
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        test = self.np_random.permutation(self.nums)
        self.steps = 0

        if self.secret_case_count == -1:
            self.secret_cases = np.array(list(itertools.permutations(range(self.nums))))
        else:
            self.secret_cases = np.zeros((self.secret_case_count, self.nums), dtype=int)
            for i in range(self.secret_case_count):
                self.secret_cases[i] = self.np_random.permutation(self.nums)

        # init state 
        # all tests + swap registers at 0 + flags at 0
        self.registers = np.copy(test)
        self.swap = np.zeros((self.swap_register_count,), dtype=int)
        self.flags = np.zeros((self.flag_count,), dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human" or self.render_mode == "actions":
            print()
            self._render_frame()

        return observation, info

    def _apply_action(self, action):
        instr,a,b = self.actions[action]

        apply(instr, a, b, self.registers, self.swap, self.flags)
        
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

        # TODO: eval on secret cases
        # the direct test case is not enough as it does not generalize
        # => just use a giant network to sort three numbers

        info_reward = 0
        # subtract points for duplicate values in registers
        for i in range(self.nums):
            for j in range(i+1, self.nums):
                if self.registers[i] == self.registers[j]:
                    info_reward -= 1
                    all_sorted = False

        # check if sorted
        for i in range(self.nums-1):
            if self.registers[i] > self.registers[i+1]:
                all_sorted = False

        # subtract points for out of order registers
        for i in range(self.nums-1):
            for j in range(i+1, self.nums):
                if self.registers[i] > self.registers[j]:
                    info_reward -= 1

        terminated = False
        truncated = False

        if all_sorted:
            info_reward += 100
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

        info_reward -= self.steps/10

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
            s = state_to_string(self.registers, self.swap, self.flags)
            print("  " + s)

    def close(self):
        pass


if __name__ == "__main__":
    raise NotImplementedError("This is (currently) not a standalone script.")