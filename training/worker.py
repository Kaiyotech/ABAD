import sys
import os
import torch
import ctypes

from redis import Redis

from rlgym.envs import Match
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from utils.mybots_terminals import BallTouchGroundCondition
from utils.mybots_statesets import WallDribble, GroundAirDribble, BallFrontGoalState
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter
from rlgym_tools.extra_state_setters.wall_state import WallPracticeState
from rlgym_tools.extra_state_setters.symmetric_setter import KickoffLikeSetter
from rlgym_tools.extra_state_setters.goalie_state import GoaliePracticeState
from rlgym_tools.extra_state_setters.hoops_setter import HoopsLikeSetter
from rlgym.utils.state_setters.default_state import DefaultState
from utils.mybots_obs import ExpandAdvancedPaddedObs

from utils.mybots_parser import DribbleAction

from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutWorker

from training.Constants import *
from training.rewards import anneal_rewards_fn, MyRewardFunction
from rlgym_tools.extra_state_setters.augment_setter import AugmentSetter


if __name__ == "__main__":
    # ctypes.windll.kernel32.SetConsoleTitleW("RLearnWorkerABAD")
    torch.set_num_threads(1)
    streamer_mode = False
    game_speed = 100
    team_size = 1
    host = "127.0.0.1"
    past_version_prob = 0  # 0.2
    evaluation_prob = 0  # 0.01
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
        state_setter=AugmentSetter(
                        WallDribble(),
                        shuffle_within_teams=True,
                        swap_front_back=False,
                        swap_left_right=False,
                        ),
        obs_builder=ExpandAdvancedPaddedObs(),
        action_parser=DribbleAction(),
        terminal_conditions=[TimeoutCondition(round(300 // T_STEP)),
                             GoalScoredCondition(),
                             BallTouchGroundCondition(),
                             ],
        reward_function=MyRewardFunction(
            team_spirit=0,
            goal_w=0,
            aerial_goal_w=8,
            double_tap_goal_w=5,
            shot_w=1,
            save_w=2,
            demo_w=1,
            above_w=0,
            got_demoed_w=-1,
            behind_ball_w=0,
            save_boost_w=0,
            concede_w=-6.5,
            velocity_w=0,
            velocity_pb_w=0,
            velocity_bg_w=0.025,
            aerial_ball_touch_w=0.5,
            kickoff_w=0,
            ball_touch_w=0,
            touch_grass_w=0,
            ceiling_touch_w=0,
            dist_ball_w=0,
            height_w=0.1,
            final_w=-3,
        )
    )

    r = Redis(host=host,
              username="user1",
              password=os.environ["redis_user1_key"],
              health_check_interval=30,
              retry_on_timeout=True,
              socket_keepalive=True,
              socket_timeout=120,
              socket_connect_timeout=30,
              )
    RedisRolloutWorker(r,
                       name,
                       match,
                       past_version_prob=past_version_prob,
                       streamer_mode=streamer_mode,
                       send_gamestates=False,
                       evaluation_prob=evaluation_prob,
                       sigma_target=2,
                       ).run()
