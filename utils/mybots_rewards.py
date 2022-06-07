import numpy as np

from rlgym.utils import math
from rlgym.utils.common_values import BLUE_TEAM, BLUE_GOAL_BACK, ORANGE_GOAL_BACK, ORANGE_TEAM, BALL_MAX_SPEED, \
    CAR_MAX_SPEED, BALL_RADIUS, GOAL_HEIGHT
from rlgym.utils.gamestates import GameState, PlayerData

from rlgym.utils.math import cosine_similarity

from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import VelocityPlayerToBallReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions.common_rewards.misc_rewards import *
from rlgym.utils.reward_functions.common_rewards.conditional_rewards import *
from rlgym.utils.reward_functions import RewardFunction, CombinedReward
from rlgym_tools.extra_rewards.distribute_rewards import DistributeRewards

from numpy.linalg import norm


class AerialGoalReward(RewardFunction):

    def __init__(self):
        raise NotImplemented  # this only works for 1v0 currently, may have bugs?
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
        raise NotImplemented  # this only works for 1v0 currently, may have bugs?
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
        raise NotImplemented  # this only works for 1v0 currently
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
        raise NotImplemented  # this only works for 1v0 currently
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
        raise NotImplemented
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


class AccelBallReward(RewardFunction):

    def __init__(self):
        self.last_vel = None

    def reset(self, initial_state: GameState):
        self.last_vel = initial_state.ball.linear_velocity

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched:
            curr_vel = state.ball.linear_velocity
            reward = norm(curr_vel - self.last_vel) / CAR_MAX_SPEED
            self.last_vel = curr_vel
            return reward
        else:
            return 0


class AccelCarReward(RewardFunction):

    def __init__(self):
        self.last_vel = None

    def reset(self, initial_state: GameState):
        self.last_vel = initial_state

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        i = player.car_id
        curr_vel = player.car_data.linear_velocity
        if self.last_vel is None:
            self.last_vel = curr_vel
            return 0
        reward = norm(curr_vel - self.last_vel) / CAR_MAX_SPEED
        self.last_vel = curr_vel
        return reward


class BoostReward(RewardFunction):

    def __init__(self):
        self.last_boost = None

    def reset(self, initial_state: GameState):
        self.last_boost = None  # this isn't exactly right, but I'm not sure what to do better

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        curr_boost = player.boost_amount
        if self.last_boost is None:
            self.last_boost = curr_boost
            return 0
        reward = np.sqrt(curr_boost) - np.sqrt(self.last_boost)
        self.last_boost = curr_boost
        return reward


class CoyoteReward(RewardFunction):

    def __init__(
        self,
        goal_w=5,
        concede_w=-5,
        velocity_pb_w=0,
        velocity_bg_w=0.005,  # 0.005,
        kickoff_w=0.025,
        ball_touch_w=0.02,  # 0.01,
        touch_grass_w=-0.001,
        acel_car_w=0.025,  # 0.01,
        acel_ball_w=0.035,  # 0.01,
        boost_gain_w=0.02,  # 0.01,
        boost_spend_w=-0.025,  # -0.01,
    ):
        self.goal_w = goal_w
        self.concede_w = concede_w
        self.velocity_pb_w = velocity_pb_w
        self.velocity_bg_w = velocity_bg_w
        self.ball_touch_w = ball_touch_w
        self.kickoff_w = kickoff_w
        self.touch_grass_w = touch_grass_w
        self.acel_car_w = acel_car_w
        self.acel_ball_w = acel_ball_w
        self.boost_gain_w = boost_gain_w
        self.boost_spend_w = boost_spend_w
        self.been_touched = False
        self.last_touched = None
        self.rewards = None
        self.current_state = None
        self.last_state = None
        self.n = 0

    def _calculate_rewards(self, state: GameState):
        # Calculate rewards, positive for blue, negative for orange
        player_rewards = np.zeros(len(state.players))
        # ball_height = state.ball.position[2]

        for i, player in enumerate(state.players):
            last = self.last_state.players[i]

            if player.ball_touched:
                self.been_touched = True  # someone has touched the ball
                self.last_touched = i
                # ball touch
                player_rewards[i] += self.ball_touch_w

                # vel bg
                if player.team_num == BLUE_TEAM:
                    objective = np.array(ORANGE_GOAL_BACK)
                else:
                    objective = np.array(BLUE_GOAL_BACK)
                vel = state.ball.linear_velocity
                pos_diff = objective - state.ball.position
                norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
                norm_vel = vel / BALL_MAX_SPEED
                player_rewards[i] += self.velocity_bg_w * float(np.dot(norm_pos_diff, norm_vel))

                # acel_ball
                curr_ball_vel = self.current_state.ball.linear_velocity
                last_ball_vel = self.last_state.ball.linear_velocity
                player_rewards[i] += self.acel_ball_w * (norm(curr_ball_vel - last_ball_vel) / CAR_MAX_SPEED)

            # boost
            boost_diff = np.sqrt(player.boost_amount) - np.sqrt(last.boost_amount)
            if boost_diff >= 0:
                player_rewards[i] += self.boost_gain_w * boost_diff
            else:
                player_rewards[i] += self.boost_spend_w * boost_diff

            # touch_grass
            player_rewards[i] += self.touch_grass_w * player.on_ground

            # vel pb
            vel = player.car_data.linear_velocity
            pos_diff = state.ball.position - player.car_data.position
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            norm_vel = vel / CAR_MAX_SPEED
            speed_rew = float(np.dot(norm_pos_diff, norm_vel))
            player_rewards[i] += self.velocity_pb_w * speed_rew

            # kickoff extra reward for speed to ball
            if state.ball.position[0] == 0 and state.ball.position[1] == 0:
                player_rewards[i] += self.kickoff_w * speed_rew

            # acel_car
            curr_car_vel = player.car_data.linear_velocity
            last_car_vel = last.car_data.linear_velocity
            player_rewards[i] += self.acel_car_w * (norm(curr_car_vel - last_car_vel) / CAR_MAX_SPEED)

        mid = len(player_rewards) // 2

        # Handle goals with no scorer for critic consistency,
        # random state could send ball straight into goal
        # TODO fix this for more than 1v1 to be only award the scorer?
        if self.been_touched:
            d_blue = state.blue_score - self.last_state.blue_score
            d_orange = state.orange_score - self.last_state.orange_score
            if d_blue > 0:
                goal_speed = norm(self.last_state.ball.linear_velocity)
                player_rewards[:mid] = (self.goal_w * goal_speed / (CAR_MAX_SPEED * 1.25))
                player_rewards[mid:] = self.concede_w
            if d_orange > 0:
                goal_speed = norm(self.last_state.ball.linear_velocity)
                player_rewards[mid:] = (self.goal_w * goal_speed / (CAR_MAX_SPEED * 1.25))
                player_rewards[:mid] = self.concede_w

        self.last_state = state
        self.rewards = player_rewards

    def reset(self, initial_state: GameState):
        self.n = 0
        self.last_state = None
        self.rewards = None
        self.current_state = initial_state
        self.been_touched = False

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state != self.current_state:
            self.last_state = self.current_state
            self.current_state = state
            self._calculate_rewards(state)
            self.n = 0
        rew = self.rewards[self.n]
        self.n += 1
        return float(rew)
