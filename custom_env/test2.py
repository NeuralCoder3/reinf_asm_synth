import gymnasium as gym

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.policies import CnnPolicy
import torch

from gymnasium.wrappers import TimeLimit

import optuna
from absl import flags
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement

import os
import pickle as pkl
import random
import sys
import time
from pprint import pprint

from optuna_utils.sample_params.ppo import sample_ppo_params
from optuna_utils.trial_eval_callback import TrialEvalCallback

from grid_env import GridWorldEnv
from count_env import CountEnv
from sort_env import SortEnv
from sort_env_asm import SortAsmEnv

study_path = "studies/sort_asm"

def objective(trial: optuna.Trial) -> float:

    time.sleep(random.random() * 16)

    # step_mul = trial.suggest_categorical("step_mul", [4, 8, 16, 32, 64])
    # env_kwargs = {"step_mul": step_mul}
    env_kwargs = {}

    sampled_hyperparams = sample_ppo_params(trial)

    path = f"{study_path}/trial_{str(trial.number)}"
    os.makedirs(path, exist_ok=True)

    env = SortAsmEnv(informed_reward=True, nums=3, max_episode_steps=50, swap_registers=1, **env_kwargs)
    env = Monitor(env)
    model = PPO("MlpPolicy", env=env, seed=None, verbose=0, tensorboard_log=path, **sampled_hyperparams)

    stop_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=30, min_evals=50, verbose=1)
    eval_callback = TrialEvalCallback(
        env, trial, best_model_save_path=path, log_path=path,
        n_eval_episodes=5, eval_freq=10000, deterministic=False, callback_after_eval=stop_callback
    )

    params = env_kwargs | sampled_hyperparams
    with open(f"{path}/params.txt", "w") as f:
        f.write(str(params))

    try:
        model.learn(10000000, callback=eval_callback)
        env.close()
    except (AssertionError, ValueError) as e:
        env.close()
        print(e)
        print("============")
        print("Sampled params:")
        pprint(params)
        raise optuna.exceptions.TrialPruned()

    is_pruned = eval_callback.is_pruned
    reward = eval_callback.best_mean_reward

    del model.env
    del model

    if is_pruned:
        raise optuna.exceptions.TrialPruned()

    return reward


if __name__ == "__main__":

    sampler = TPESampler(n_startup_trials=10, multivariate=True)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=10)

    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
        direction="maximize",
    )

    try:
        study.optimize(objective, n_jobs=4, n_trials=128)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    trial = study.best_trial
    print(f"Best trial: {trial.number}")
    print("Value: ", trial.value)

    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    study.trials_dataframe().to_csv(f"{study_path}/report.csv")

    with open(f"{study_path}/study.pkl", "wb+") as f:
        pkl.dump(study, f)

    try:
        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)
        fig3 = plot_parallel_coordinate(study)

        fig1.show()
        fig2.show()
        fig3.show()

    except (ValueError, ImportError, RuntimeError) as e:
        print("Error during plotting")
        print(e)





