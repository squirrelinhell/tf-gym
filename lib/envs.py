
import gym.envs

gym.envs.register(
    id='BipedalWalkerShort-v2',
    entry_point='gym.envs.box2d:BipedalWalker',
    max_episode_steps=300,
    reward_threshold=300,
)
