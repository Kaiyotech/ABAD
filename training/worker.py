import sys
import os
import torch
import ctypes

from redis import Redis

from rlgym.envs import Match
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from utils.mybots_statesets import WallDribble, GroundAirDribble
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter
from rlgym_tools.extra_state_setters.wall_state import WallPracticeState
from rlgym_tools.extra_state_setters.symmetric_setter import KickoffLikeSetter
from rlgym_tools.extra_state_setters.goalie_state import GoaliePracticeState
from rlgym_tools.extra_state_setters.hoops_setter import HoopsLikeSetter
from rlgym.utils.state_setters.default_state import DefaultState
from utils.mybots_obs import ExpandAdvancedPaddedStackObs

from utils.nectoparser import NectoAction

from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutWorker

from training.Constants import *
from training.rewards import anneal_rewards_fn
from rlgym_tools.extra_state_setters.augment_setter import AugmentSetter


if __name__ == "__main__":
    ctypes.windll.kernel32.SetConsoleTitleW("RLearnWorkerABAD")
    torch.set_num_threads(1)
    streamer_mode = False
    game_speed = 100
    team_size = 1
    if len(sys.argv) > 1:
        team_size = int(sys.argv[1])
    if len(sys.argv) > 2:
        if sys.argv[2] == 'STREAMER':
            streamer_mode = True
            game_speed = 1
    match = Match(
        game_speed=game_speed,
        self_play=True,
        team_size=team_size,
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
                        AugmentSetter(
                            KickoffLikeSetter(
                                cars_on_ground=True,
                                ball_on_ground=True,
                                )
                            ),
                        AugmentSetter(
                            KickoffLikeSetter(
                                cars_on_ground=False,
                                ball_on_ground=False,
                            )
                        ),
                        AugmentSetter(
                            WallPracticeState()
                            ),
                        # AugmentSetter(
                        #     GoaliePracticeState(
                        #         allow_enemy_interference=True,
                        #         aerial_only=False,
                        #         first_defender_in_goal=True,
                        #         reset_to_max_boost=False,
                        #         )
                        #     ),
                        AugmentSetter(
                            HoopsLikeSetter()
                        ),
                        DefaultState()  # this is kickoff normal
                        ),
                        (
                        0.1,  # groundair
                        0.1,  # wallair
                        0.45,  # kickofflike ground
                        0.05,  # kickofflike air
                        0.075,  # wall
                        # 0.10,  # goalie
                        0.075,  # hoops
                        0.15,  # default kickoff
                        ),
                    ),
        obs_builder=ExpandAdvancedPaddedStackObs(stack_size=5, team_size=3),
        action_parser=NectoAction(),
        terminal_conditions=[TimeoutCondition(round(60 // T_STEP)),
                             GoalScoredCondition(),
                             ],
        reward_function=anneal_rewards_fn()
    )

    r = Redis(host="127.0.0.1", username="user1", password=os.environ["redis_user1_key"])
    RedisRolloutWorker(r,
                       "Kaiyotech",
                       match,
                       past_version_prob=0.1,
                       streamer_mode=streamer_mode,
                       send_gamestates=False,
                       evaluation_prob=0,
                       ).run()
