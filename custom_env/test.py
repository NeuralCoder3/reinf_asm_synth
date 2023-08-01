import gymnasium as gym

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.policies import CnnPolicy
import torch

from gymnasium.wrappers import TimeLimit

from grid_env import GridWorldEnv
from count_env import CountEnv
from sort_env import SortEnv
from sort_env_asm import SortAsmEnv

# model_name = "ppo-GridWorldEnv"
# max_steps = 100
# envGenerator = lambda **args: TimeLimit(GridWorldEnv(size=5), max_episode_steps=max_steps, **args)
# training_episodes = 100000

# model_name = "ppo-CountEnv"
# successfully learns binary code for 42 (try other choice_bounds for pos,neg combination)
# envGenerator = lambda **args: CountEnv(early_termination=True, reward_per_step=True, max_random=42, **args)
# only works for positive goals (works better with diff as observation)
# envGenerator = lambda **args: CountEnv(early_termination=True, reward_per_step=True, max_random=64, random_goal=True, **args)
# needs early termination
# envGenerator = lambda **args: CountEnv(early_termination=True, reward_per_step=False, max_random=64, random_goal=True, **args)
#
# training_episodes = 100000


# model_name = "ppo-SortEnv3"
# envGenerator = lambda **args: SortEnv(informed_reward=True,**args)
# training_episodes = 10000
# # trains for five nodes
# model_name = "ppo-SortEnv5"
# envGenerator = lambda **args: SortEnv(informed_reward=True, nums=5,**args)
# training_episodes = 100000
# # only train using 10 permutations
# model_name = "ppo-SortEnv5"
# envGenerator = lambda **args: SortEnv(informed_reward=True, nums=5, num_tests=10,**args)
# training_episodes = 200000
# # try with 6 elements and 40 test cases
# model_name = "ppo-SortEnv6-sample"
# envGenerator = lambda **args: SortEnv(informed_reward=True, nums=6, num_tests=40,**args)
# training_episodes = 200000
# model_name = "ppo-SortEnv10-sample"
# envGenerator = lambda **args: SortEnv(informed_reward=True, nums=10, num_tests=100, max_episode_steps=10**3,**args)
# training_episodes = 1000000

model_name = "ppo-SortAsmEnv3"
# envGenerator = lambda **args: SortAsmEnv(informed_reward=True, nums=3, max_episode_steps=3**3,**args)
envGenerator = lambda **args: SortAsmEnv(informed_reward=True, nums=3, max_episode_steps=100,**args)
# envGenerator = lambda **args: SortAsmEnv(informed_reward=True, extra_registers=1, nums=3, max_episode_steps=50, num_tests=100,**args)
training_episodes = 1000000

train = True
# train = False
continueModel = False
# continueModel = True

# evalMode = "human"
evalCount = 10
evalMode = "actions"
# evalCount = 1
# deterministicEval = True
deterministicEval = False

evalModel = lambda **args: Monitor(envGenerator(render_mode=evalMode, **args))
env = envGenerator()
observation, info = env.reset(seed=42)

# https://spinningup.openai.com/en/latest/algorithms/ppo.html
# model = PPO(
#     policy="MlpPolicy",
#     # policy="MultiInputPolicy",
#     env=env,
#     n_steps=1024,
#     batch_size=64,
#     n_epochs=4,
#     gamma=0.999,
#     gae_lambda=0.98,
#     ent_coef=0.01,
#     verbose=1,
# )


# https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
# default
# policy_kwargs = dict(activation_fn=torch.nn.Tanh,
#                      net_arch=dict(pi=[64, 64], vf=[64, 64]))
# same as net_arch=[64, 64]
# pi = policy network
# vf = value function network
# policy_kwargs = dict(activation_fn=torch.nn.ReLU,
#                      net_arch=dict(pi=[128, 64], vf=[128, 64]))
policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[64, 64, 64])

model = PPO(
    # policy="CnnPolicy",
    policy="MlpPolicy",
    env=env,
    verbose=1,
    policy_kwargs=policy_kwargs,
)
print(model.policy)
# exit(1)

# model = A2C(
#     CnnPolicy,
#     env,
#     verbose=1
# )


if continueModel or not train:
    try:
        model = PPO.load(model_name, env=env)
        print("Loaded model from", model_name)
    except:
        print("Could not load model from", model_name)
        if not train:
            print("Exiting")
            exit(1)

if train:
    model.learn(total_timesteps=training_episodes)
    print("Finished learning")
    print("Saving model to", model_name)
    model.save(model_name)


print("Evaluating model")
eval_env = evalModel()
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=evalCount, deterministic=deterministicEval)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")