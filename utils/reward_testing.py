import os
import wandb
import numpy
from typing import Any

import torch.jit
from torch.nn import Linear, Sequential, ReLU

from redis import Redis

from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym.utils.gamestates import PlayerData, GameState
from training.rewards import anneal_rewards_fn, MyRewardFunction
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction

from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from training.agent import DiscretePolicy
from training.my_ppo import PPO
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator
from rocket_learn.utils.util import ExpandAdvancedObs, SplitLayer

from training.Constants import *
from utils.misc import count_parameters


if __name__ == "__main__":
    split = (3, 3, 2, 2, 2)
    total_output = sum(split)

    # TOTAL SIZE OF THE INPUT DATA
    state_dim = 107

    critic = Sequential(
        Linear(state_dim, 256),
        ReLU(),
        Linear(256, 256),
        ReLU(),
        Linear(256, 1)
    )

    actor = DiscretePolicy(Sequential(
        Linear(state_dim, 256),
        ReLU(),
        Linear(256, 256),
        ReLU(),
        Linear(256, total_output),
        SplitLayer(splits=split)
    ), split)

    optim = torch.optim.Adam([
        {"params": actor.parameters()},
        {"params": critic.parameters()}
    ])

    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)

    params = count_parameters(agent)
    print(f"Number of parameters is {params}")


# def anneal_rewards_fn():
#
#     max_steps = 100_000_000
#     # when annealing, change the weights between 1 and 2, 2 is new
#     reward1 = MyOldRewardFunction(
#         team_spirit=0,
#         goal_w=10,
#         shot_w=5,
#         save_w=5,
#         demo_w=0,
#         above_w=0,
#         got_demoed_w=0,
#         behind_ball_w=0,
#         save_boost_w=0.03,
#         concede_w=0,
#         velocity_w=0,
#         velocity_pb_w=0,
#         velocity_bg_w=0.5,
#         ball_touch_w=4,
#     )
#     reward2 = MyRewardFunction(
#         team_spirit=0,
#         goal_w=10,
#         shot_w=5,
#         save_w=5,
#         demo_w=0,
#         above_w=0,
#         got_demoed_w=0,
#         behind_ball_w=0,
#         save_boost_w=0.03,
#         concede_w=0,
#         velocity_w=0,
#         velocity_pb_w=0,
#         velocity_bg_w=0.5,
#         ball_touch_w=1,
#     )
#
#     alternating_rewards_steps = [reward1, max_steps, reward2]
#
#     return AnnealRewards(*alternating_rewards_steps, mode=AnnealRewards.STEP)
#
#
# env = rlgym.make(
#         reward_fn=DoubleTapReward(),
#         game_speed=1,
#         state_setter=AugmentSetter(WallDribble(),
#                                    shuffle_within_teams=True,
#                                    swap_front_back=False,
#                                    swap_left_right=False,
#                                    swap_teams=False,
#                                    ),
#         terminal_conditions=[BallTouchGroundCondition()],
#         self_play=True,
#         )
# try:
#     while True:
#         env.reset()
#         done = False
#         while not done:
#             # Here we sample a random action. If you have an agent, you would get an action from it here.
#             # action = env.action_space.sample()
#             action = [1, 0, 0, 0, 0, 0, 0, 0] * 2
#
#             next_obs, reward, done, gameinfo = env.step(action)
#
#             if any(reward) > 0:
#                 print(reward)
#                 pass
#
#             obs = next_obs
#
# finally:
#     env.close()
