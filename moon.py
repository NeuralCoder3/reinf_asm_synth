import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)

for i in range(10000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)
   if i % 100 == 0:
      print("Reward:",reward)

   if terminated or truncated:
      observation, info = env.reset()
      print("Reset")


env.close()
