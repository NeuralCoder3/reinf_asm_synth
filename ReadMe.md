We use reinforcement learning (RL) to synthesize algorithms.
In this repository, we focus on general RL-techniques and focus on low-level algorithms.

For the current experiments, we use [Gymnasium](https://gymnasium.farama.org/index.html) 
as a framework to reduce the boilerplate of reinforcement learning.

The ideas are inspired by
* [AlphaTensor](https://www.deepmind.com/blog/discovering-novel-algorithms-with-alphatensor)
* [AlphaDev](https://github.com/deepmind/alphadev)

Both methods are based on AlphaZero and use MCTS with networks (including transformer networks)
for reinforcement learning.

There are at least for AlphaTensor some reimplementations:
* [AlphaStrassen](https://github.com/mishgon/alphastrassen)
* [Nebuly OpenAlphaTensor](https://github.com/nebuly-ai/nebuly/tree/main/optimization/open_alpha_tensor)

In contrast, we first try generalized reinforcement learning approaches where the specialization is completely
localized to the environment.

The first experiments use [PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) as an agent (code based on [hugging face](https://huggingface.co/learn/deep-rl-course/unit1/hands-on?fw=pt)).

## Environments

### Moon

Given gymnasium example.
In this environment, a spacecraft has to land safely on a landing region.
The agent can control left/right/up thrusters.

### Grid

In this simple environment (based on the [gymnasium example](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#sphx-glr-tutorials-gymnasium-basics-environment-creation-py)), the agent is on a position in a grid and has to manoeuvre to a target.
The agent can move to any adjacent tile.

[TODO]

## Results

All trainings took less than 30min.

[TODO]