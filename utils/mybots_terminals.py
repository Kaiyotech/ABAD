from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.gamestates import GameState
from rlgym.utils.common_values import BALL_RADIUS


class BallTouchGroundCondition(TerminalCondition):
    """
    A condition that will terminate an episode after ball touches ground
    """

    def __init__(self, min_steps=30):
        super().__init__()
        self.min_steps = min_steps
        self.steps = 0
        self.counter = 0

    def reset(self, initial_state: GameState):
        self.steps = 0
        self.counter = 0

    def is_terminal(self, current_state: GameState) -> bool:
        """
        return True if ball is touching the ground and it has been minimum number of steps
        """
        self.steps += 1
        if self.counter > 0:
            self.counter += 1  # continue counting if started, regardless of ball position
        if self.steps > self.min_steps and current_state.ball.position[2] < (2 * BALL_RADIUS):
            self.counter += 1

        if self.steps > self.min_steps and current_state.ball.position[2] < (2 * BALL_RADIUS)\
                and self.counter > 4:  # give 4 extra ticks to account for balls about to score but low
            return True
        else:
            return False
