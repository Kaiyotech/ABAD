from typing import Any
import numpy
import sys
import os
import torch
import ctypes

from redis import Redis

from rlgym.envs import Match
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym.utils.reward_functions.default_reward import DefaultReward
from utils.mybots_statesets import WallDribble, GroundAirDribble
from rocket_learn.utils.util import ExpandAdvancedObs
from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter
from rlgym.utils.action_parsers.discrete_act import DiscreteAction

from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutWorker

from training.Constants import *
from training.rewards import anneal_rewards_fn, MyRewardFunction
from utils.mybots_terminals import *
from rlgym_tools.extra_state_setters.augment_setter import AugmentSetter


# ROCKET-LEARN ALWAYS EXPECTS A BATCH DIMENSION IN THE BUILT OBSERVATION
class ExpandAdvancedObs(AdvancedObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: numpy.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)


if __name__ == "__main__":
    ctypes.windll.kernel32.SetConsoleTitleW("RLearnWorkerABAD")
    torch.set_num_threads(1)
    streamer_mode = False
    game_speed = 100
    if len(sys.argv) > 1:
        if sys.argv[1] == 'STREAMER':
            streamer_mode = True
            game_speed = 1
    match = Match(
        game_speed=game_speed,
        self_play=True,
        team_size=1,
        state_setter=WeightedSampleSetter((
                        AugmentSetter(
                            GroundAirDribble(),
                            shuffle_within_teams=True,
                            swap_front_back=False,
                            swap_left_right=False,
                            ),
                        AugmentSetter(
                            WallDribble(),
                            shuffle_within_teams=True,
                            swap_front_back=False,
                            swap_left_right=False,
                            ),
                        ),
                        (
                        0.5,
                        0.5,
                        ),
                    ),
        obs_builder=ExpandAdvancedObs(),
        action_parser=KBMAction(),
        terminal_conditions=[TimeoutCondition(round(20 // T_STEP)),
                             GoalScoredCondition(), BallTouchGroundCondition(round(3 // T_STEP))],
        reward_function=anneal_rewards_fn()
    )

    r = Redis(host="127.0.0.1", username="user1", password=os.environ["redis_user1_key"])
    RedisRolloutWorker(r,
                       "Kaiyotech",
                       match,
                       past_version_prob=0.2,
                       streamer_mode=streamer_mode,
                       send_gamestates=False,
                       evaluation_prob=0.0,
                       ).run()
