import numpy as np
import torch
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlgym_compat import GameState

from mybots_obs import ExpandAdvancedPaddedObs
from nectoparser import NectoAction

from agent import Agent


class Coyote(BaseAgent):
    def __init__(self, name, team, index, beta=1):
        super().__init__(name, team, index)

        self.obs_builder = None
        self.agent = Agent()
        self.tick_skip = 8

        self.game_state: GameState = None
        self.controls = None
        self.action = None
        self.update_action = True
        self.ticks = 0
        self.prev_time = 0

    def initialize_agent(self):
        # Initialize the rlgym GameState object now that the game is active and the info is available
        field_info = self.get_field_info()
        self.obs_builder = ExpandAdvancedPaddedObs()
        self.game_state = GameState(field_info)
        self.ticks = self.tick_skip  # So we take an action the first tick
        self.prev_time = 0
        self.controls = SimpleControllerState()
        self.action = np.zeros(8)
        self.update_action = True

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:

        cur_time = packet.game_info.seconds_elapsed
        delta = cur_time - self.prev_time
        self.prev_time = cur_time

        ticks_elapsed = round(delta * 120)
        self.ticks += ticks_elapsed
        self.game_state.decode(packet, ticks_elapsed)

        if self.update_action:
            self.update_action = False

            player = self.game_state.players[self.index]
            # teammates = [p for p in self.game_state.players if p.team_num == self.team and p != player]
            # opponents = [p for p in self.game_state.players if p.team_num != self.team]
            #
            # self.game_state.players = [player] + teammates + opponents

            obs = self.obs_builder.build_obs(player, self.game_state, self.action)
            self.action = self.agent.act(obs)

        if self.ticks >= self.tick_skip - 1:
            self.update_controls(self.action)

        if self.ticks >= self.tick_skip:
            self.update_action = True
            self.ticks = 0

        return self.controls

    def update_controls(self, action):
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = 0 if action[5] > 0 else action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0
        self.controls.boost = action[6] > 0
        self.controls.handbrake = action[7] > 0
