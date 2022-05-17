import sys
import os
import torch
import ctypes

from redis import Redis

from rlgym.envs import Match
from rlgym.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition, NoTouchTimeoutCondition
from utils.mybots_terminals import BallTouchGroundCondition, EagleTerminalCondition
from utils.mybots_statesets import WallDribble, GroundAirDribble, BallFrontGoalState, EagleState
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter
from rlgym_tools.extra_state_setters.wall_state import WallPracticeState
from rlgym_tools.extra_state_setters.symmetric_setter import KickoffLikeSetter
from rlgym_tools.extra_state_setters.goalie_state import GoaliePracticeState
from rlgym_tools.extra_state_setters.hoops_setter import HoopsLikeSetter
from rlgym.utils.state_setters.default_state import DefaultState
from utils.mybots_obs import ExpandAdvancedObs

from utils.mybots_parser import DribbleAction

from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutWorker

from training.Constants import *
from training.rewards import anneal_rewards_fn, MyRewardFunction, EagleReward
from rlgym_tools.extra_state_setters.augment_setter import AugmentSetter
from rlgym.utils.reward_functions.combined_reward import CombinedReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward
from utils.mybots_rewards import DoubleTapReward


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
        self_play=False,
        team_size=1,
        state_setter=EagleState(),
        obs_builder=ExpandAdvancedObs(),
        action_parser=DribbleAction(),
        terminal_conditions=[BallTouchGroundCondition(), GoalScoredCondition()],
        reward_function=CombinedReward(
                    (
                        EagleReward(),
                        VelocityBallToGoalReward(),
                        EventReward(goal=10),
                        DoubleTapReward(),
                    ),
                    (
                        0.1,
                        0.2,
                        1,
                        20,
                    ),
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
