from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym_tools.extra_obs.advanced_stacker import AdvancedStacker
import numpy
from rlgym.utils.gamestates import PlayerData, GameState, PhysicsObject
from rlgym.utils.obs_builders import ObsBuilder
from typing import Any, List
from rlgym.utils import common_values
import numpy as np
import math
from collections import deque


class ExpandAdvancedObs(AdvancedObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: numpy.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)


class ExpandAdvancedStackObs(AdvancedStacker):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: numpy.ndarray) -> Any:
        obs = super(ExpandAdvancedStackObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)


def _add_dummy(obs: List):
    obs.extend([
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        [0, 0, 0, 0]])
    obs.extend([np.zeros(3), np.zeros(3)])


class AdvancedActionStackingPaddingObs(ObsBuilder):
    """adds a stack of previous actions to the obs as well as 0 padding to accommodate differing numbers of agents"""
    def __init__(self, team_size=3, stack_size=5, expanding=False):
        super().__init__()
        self.team_size = team_size
        self.POS_STD = 2300
        self.ANG_STD = math.pi
        self.expanding = expanding
        self.default_action = [0, 0, 0, 0, 0, 0, 0, 0]
        self.stack_size = stack_size
        self.action_stack = [deque([], maxlen=self.stack_size) for _ in range(66)]
        for i in range(len(self.action_stack)):
            self.blank_stack(i)

    def blank_stack(self, index: int) -> None:
        for _ in range(self.stack_size):
            self.action_stack[index].appendleft(self.default_action)

    def add_action_to_stack(self, new_action: np.ndarray, index: int):
        self.action_stack[index].appendleft(new_action)

    def reset(self, initial_state: GameState):
        for p in initial_state.players:
            self.blank_stack(p.car_id)

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        self.add_action_to_stack(previous_action, player.car_id)
        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pads
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pads

        obs = [ball.position / self.POS_STD,
               ball.linear_velocity / self.POS_STD,
               ball.angular_velocity / self.ANG_STD,
               previous_action,
               pads]

        obs.extend(list(self.action_stack[player.car_id]))

        player_car = self._add_player_to_obs(obs, player, ball, inverted)

        allies = []
        enemies = []
        ally_count = 0
        enemy_count = 0
        for other in state.players:
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies
                ally_count += 1
                if ally_count > self.team_size - 1:
                    continue
            else:
                team_obs = enemies
                enemy_count += 1
                if enemy_count > self.team_size:
                    continue
            other_car = self._add_player_to_obs(team_obs, other, ball, inverted)

            # Extra info
            team_obs.extend([
                (other_car.position - player_car.position) / self.POS_STD,
                (other_car.linear_velocity - player_car.linear_velocity) / self.POS_STD
            ])

        while ally_count < self.team_size - 1:
            _add_dummy(allies)
            ally_count += 1

        while enemy_count < self.team_size:
            _add_dummy(enemies)
            enemy_count += 1

        obs.extend(allies)
        obs.extend(enemies)
        if self.expanding:
            return np.expand_dims(np.concatenate(obs), 0)
        return np.concatenate(obs)

    def _add_player_to_obs(self, obs: List, player: PlayerData, ball: PhysicsObject, inverted: bool):
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        rel_pos = ball.position - player_car.position
        rel_vel = ball.linear_velocity - player_car.linear_velocity

        obs.extend([
            rel_pos / self.POS_STD,
            rel_vel / self.POS_STD,
            player_car.position / self.POS_STD,
            player_car.forward(),
            player_car.up(),
            player_car.linear_velocity / self.POS_STD,
            player_car.angular_velocity / self.ANG_STD,
            [player.boost_amount,
             int(player.on_ground),
             int(player.has_flip),
             int(player.is_demoed)]])

        return player_car


class AdvancedObsPadder(ObsBuilder):
    """adds 0 padding to accommodate differing numbers of agents"""

    def __init__(self, team_size=3, expanding=False):
        super().__init__()
        self.team_size = team_size
        self.POS_STD = 2300
        self.ANG_STD = math.pi
        self.expanding = expanding

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:

        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pads
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pads

        obs = [ball.position / self.POS_STD,
               ball.linear_velocity / self.POS_STD,
               ball.angular_velocity / self.ANG_STD,
               previous_action,
               pads]

        player_car = self._add_player_to_obs(obs, player, ball, inverted)

        allies = []
        enemies = []
        ally_count = 0
        enemy_count = 0
        for other in state.players:
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies
                ally_count += 1
                if ally_count > self.team_size - 1:
                    continue
            else:
                team_obs = enemies
                enemy_count += 1
                if enemy_count > self.team_size:
                    continue
            other_car = self._add_player_to_obs(team_obs, other, ball, inverted)

            # Extra info
            team_obs.extend([
                (other_car.position - player_car.position) / self.POS_STD,
                (other_car.linear_velocity - player_car.linear_velocity) / self.POS_STD
            ])

        while ally_count < self.team_size - 1:
            self._add_dummy(allies)
            ally_count += 1

        while enemy_count < self.team_size:
            self._add_dummy(enemies)
            enemy_count += 1

        obs.extend(allies)
        obs.extend(enemies)
        if self.expanding:
            return np.expand_dims(np.concatenate(obs), 0)
        return np.concatenate(obs)

    def _add_dummy(self, obs: List):
        obs.extend([
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            [0, 0, 0, 0]])
        obs.extend([np.zeros(3), np.zeros(3)])

    def _add_player_to_obs(self, obs: List, player: PlayerData, ball: PhysicsObject, inverted: bool):
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        rel_pos = ball.position - player_car.position
        rel_vel = ball.linear_velocity - player_car.linear_velocity

        obs.extend([
            rel_pos / self.POS_STD,
            rel_vel / self.POS_STD,
            player_car.position / self.POS_STD,
            player_car.forward(),
            player_car.up(),
            player_car.linear_velocity / self.POS_STD,
            player_car.angular_velocity / self.ANG_STD,
            [player.boost_amount,
             int(player.on_ground),
             int(player.has_flip),
             int(player.is_demoed)]])

        return player_car


class ExpandAdvancedPaddedStackObs(AdvancedActionStackingPaddingObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: numpy.ndarray) -> Any:
        obs = super(ExpandAdvancedPaddedStackObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)


if __name__ == "__main__":
    pass
