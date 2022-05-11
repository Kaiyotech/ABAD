import numpy as np

from rlgym.utils import math
from rlgym.utils.common_values import BLUE_TEAM, BLUE_GOAL_BACK, ORANGE_GOAL_BACK, ORANGE_TEAM, BALL_MAX_SPEED, \
    CAR_MAX_SPEED, BALL_RADIUS, GOAL_HEIGHT, CEILING_Z
from rlgym.utils.gamestates import GameState, PlayerData

from rlgym.utils.math import cosine_similarity

from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions.common_rewards.misc_rewards import *
from rlgym.utils.reward_functions.common_rewards.conditional_rewards import *
from rlgym.utils.reward_functions import RewardFunction, CombinedReward
from rlgym_tools.extra_rewards.distribute_rewards import DistributeRewards


class AerialGoalReward(RewardFunction):

    def __init__(self):
        self.last_touch_height = 0
        self.initial_state = None
        self.last_goal = None

    def reset(self, initial_state: GameState):
        self.last_touch_height = 0
        self.initial_state = initial_state

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM:
            self.last_goal = self.initial_state.blue_score
            now_score = state.blue_score
        else:
            self.last_goal = self.initial_state.orange_score
            now_score = state.orange_score

        if state.last_touch == player.car_id:
            self.last_touch_height = player.car_data.position[2]

        if self.last_touch_height > GOAL_HEIGHT and now_score > self.last_goal:
            return 1

        else:
            return 0


class DoubleTapReward(RewardFunction):

    def __init__(self):
        self.backboard_bounce = False
        self.floor_bounce = False
        self.last_goal = 0
        self.last_ball_vel_y = 0

    def reset(self, initial_state: GameState):
        self.last_goal = initial_state.blue_score
        self.backboard_bounce = False
        self.floor_bounce = False
        self.last_ball_vel_y = initial_state.ball.linear_velocity[1]

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state.ball.position[2] < BALL_RADIUS * 2 and state.last_touch != player.car_id:
            self.floor_bounce = True

        elif 0.55 * self.last_ball_vel_y < state.ball.linear_velocity[1] > 0.65 * self.last_ball_vel_y and\
                state.ball.position[1] > 4900 and state.ball.position[2] > 500:
            self.backboard_bounce = True

        if state.blue_score > self.last_goal and self.backboard_bounce and not self.floor_bounce:
            return 1

        else:
            return 0


class AerialRewardPerTouch(RewardFunction):
    """
        Rewards consecutive touches in the air
        :param exp_base: exp_base^n where n is consecutive touches
        :param max_touches_reward: maximum reward to give regardless of consecutive touches
        """

    def __init__(self, exp_base=1.06, max_touches_reward=20):
        self.exp_base = exp_base
        self.max_touches_reward = max_touches_reward
        self.num_touches = 0

    def reset(self, initial_state: GameState):
        self.num_touches = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state.last_touch == player.car_id and state.ball.position[2] > 300:
            self.num_touches += 1
            reward = self.exp_base ** self.num_touches
            return min(reward, self.max_touches_reward) / self.max_touches_reward  # normalize to 1

        elif state.ball.position[2] <= 300:
            self.num_touches = 0
            return 0

        else:
            return 0


class IncreaseRewardPerTouch(RewardFunction):
    """
    Rewards consecutive touches
    :param exp_base: exp_base^n where n is consecutive touches
    :param max_touches_reward: maximum reward to give regardless of consecutive touches
    """

    def __init__(self, exp_base=1.06, max_touches_reward=20):
        self.exp_base = exp_base
        self.max_touches_reward = max_touches_reward
        self.num_touches = 0

    def reset(self, initial_state: GameState):
        self.num_touches = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state.last_touch == player.car_id:
            self.num_touches += 1
            reward = self.exp_base ** self.num_touches
            return min(reward, self.max_touches_reward) / self.max_touches_reward  # normalize

        else:
            self.num_touches = 0
            return 0


class AboveCrossbar(RewardFunction):
    # reward function if above the crossbar and near the ball
    # or aimed towards
    # ball, to reward good aerials pretty aggressively
    def __init__(self, defense=1., offense=1.):
        self.defense = defense
        self.offense = offense

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball_pos = state.ball.position
        pos = player.car_data.position
        # not above crossbar
        car_z = pos[2]
        if car_z < GOAL_HEIGHT:
            return 0

        # from player_ball_rewards rlgym
        dist = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
        close = np.exp(-0.5 * dist / CAR_MAX_SPEED)  # Inspired by https://arxiv.org/abs/2105.12196

        # Align player->ball and player->net vectors
        alignment = 0.5 * (cosine_similarity(ball_pos - pos, ORANGE_GOAL_BACK - pos)
                           - cosine_similarity(ball_pos - pos, BLUE_GOAL_BACK - pos))
        if player.team_num == ORANGE_TEAM:
            alignment *= -1

        return close + alignment


class OnWall(RewardFunction):

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        on_ground = player.on_ground
        absz = abs(player.car_data.up()[2])
        reward = 0
        pos = player.car_data.position
        car_x = pos[0]
        car_y = pos[1]
        car_z = pos[2]
        if on_ground and absz < 0.2:
            reward = 1
        elif on_ground and absz < 0.9:
            reward = 0.2
        return reward


class Demoed(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.is_demoed:
            return 1
        else:
            return 0


# class GetBoostReward(RewardFunction):
#     def reset(self, initial_state: GameState):  # TODO fix this to be better
#         self.last =
#
#     def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
#         # return 1 reward for picking up full pad
#         self.
#         return np.sqrt(player.boost_amount)
#
#     boost_diff = np.sqrt(player.boost_amount) - np.sqrt(last.boost_amount)
#     if boost_diff >= 0:
#         player_rewards[i] += self.boost_gain_w * boost_diff
#     else:
#         player_rewards[i] += self.boost_lose_w * boost_diff


class TouchGrass(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.on_ground:
            return 1
        else:
            return 0


class BallCloseCeilingReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state.ball.position[2] < (2 * BALL_RADIUS):
            return 1
        else:
            return 0


class HeightReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # ideal height is around 1500, make 1500 = 1 reward, CEILING_Z should be -1, floor should be 0
        if state.last_touch == player.car_id:
            z = state.ball.position[2]
            reward = (-1.2254e-10 * z ** 3) - (1.07493e-6 * z ** 2) + (0.00222 * z)
            return reward

        else:
            return 0


class FinalReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.offense = False
        self.checked = False

    def reset(self, initial_state: GameState):
        self.offense = False
        self.checked = False

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if not self.checked:
            if state.last_touch == player.car_id:
                self.offense = True
                self.checked = True
            elif state.last_touch != -1:
                self.checked = True
        return 0

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if self.checked and self.offense:
            return -1
        else:
            return 0
