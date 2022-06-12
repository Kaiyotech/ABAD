import sys
import os
import torch

from redis import Redis

from rlgym.envs import Match
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym_tools.extra_state_setters.goalie_state import GoaliePracticeState

from utils.mybots_statesets import WallDribble, GroundAirDribble, BallFrontGoalState
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter
from rlgym_tools.extra_state_setters.wall_state import WallPracticeState
from rlgym_tools.extra_state_setters.symmetric_setter import KickoffLikeSetter
from rlgym_tools.extra_state_setters.hoops_setter import HoopsLikeSetter
from rlgym.utils.state_setters.default_state import DefaultState
from utils.mybots_obs import ExpandAdvancedPaddedObs

from utils.nectoparser import NectoAction

from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutWorker
from pretrained_agents.necto.necto_v1 import NectoV1

from training.Constants import *
from training.rewards import CoyoteReward
from rlgym_tools.extra_state_setters.augment_setter import AugmentSetter


if __name__ == "__main__":
    # ctypes.windll.kernel32.SetConsoleTitleW("RLearnWorkerABAD")
    torch.set_num_threads(1)
    streamer_mode = False
    game_speed = 100
    team_size = 1
    host = "127.0.0.1"
    past_version_prob = 0.2  # 0.2
    evaluation_prob = 0.02  # 0.01
    name = "Default"
    if len(sys.argv) > 1:
        team_size = int(sys.argv[1])
    if len(sys.argv) > 2:
        host = sys.argv[2]
    if len(sys.argv) > 3:
        name = sys.argv[3]
    if len(sys.argv) > 4:
        if sys.argv[4] == 'STREAMER':
            streamer_mode = True
            past_version_prob = 0
            evaluation_prob = 0
            game_speed = 1
    name = name+"-"+str(team_size)+"s"
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
                        AugmentSetter(
                            GoaliePracticeState(
                                allow_enemy_interference=True,
                                aerial_only=False,
                                first_defender_in_goal=True,
                                reset_to_max_boost=False,
                                )
                            ),
                        AugmentSetter(
                            HoopsLikeSetter()
                        ),
                        DefaultState(),  # this is kickoff normal
                        AugmentSetter(
                            BallFrontGoalState(),
                        ),
                        ),
                        (
                        0,  # groundair make this 0
                        0.05,  # wallair
                        0.05,  # kickofflike ground
                        0.15,  # kickofflike air
                        0.30,  # wall make this 0.075
                        0.10,  # goalie
                        0.20,  # hoops
                        0.15,  # default kickoff
                        0,  # ball front goal
                        ),
                    ),
        obs_builder=ExpandAdvancedPaddedObs(),
        action_parser=NectoAction(),
        terminal_conditions=[TimeoutCondition(round(300 // T_STEP)),
                             GoalScoredCondition(),
                             ],
        reward_function=CoyoteReward(),
    )

    r = Redis(host=host, username="user1", password=os.environ["redis_user1_key"])

    model_name = "necto-model-10Y.pt"
    nectov1 = NectoV1(model_string=model_name, n_players=team_size * 2)
    pretrained_agents = {nectov1: .15}

    RedisRolloutWorker(r,
                       name,
                       match,
                       past_version_prob=past_version_prob,
                       streamer_mode=streamer_mode,
                       send_gamestates=False,
                       evaluation_prob=evaluation_prob,
                       sigma_target=2,
                       # pretrained_agents=pretrained_agents
                       ).run()
