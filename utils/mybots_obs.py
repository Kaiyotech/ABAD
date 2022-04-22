from rlgym.utils.obs_builders.advanced_obs import AdvancedObs
from rlgym_tools.extra_obs.advanced_stacker import AdvancedStacker
from rlgym.utils.gamestates import PlayerData, GameState
import numpy
from typing import Any


class ExpandAdvancedObs(AdvancedObs):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: numpy.ndarray) -> Any:
        obs = super(ExpandAdvancedObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)


class ExpandAdvancedStackObs(AdvancedStacker):
    def build_obs(self, player: PlayerData, state: GameState, previous_action: numpy.ndarray) -> Any:
        obs = super(ExpandAdvancedStackObs, self).build_obs(player, state, previous_action)
        return numpy.expand_dims(obs, 0)
