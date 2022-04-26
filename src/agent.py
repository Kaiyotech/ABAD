import os

import numpy as np
import torch
import torch.nn.functional as F
from nectoparser import NectoAction


class Agent:
    def __init__(self):
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        file = os.path.join(cur_dir, "CoyoteBot_model.pt")
        print(f"attempting to open {file}")
        self.actor = torch.jit.load(os.path.join(cur_dir, "CoyoteBot_model.pt"))
        torch.set_num_threads(1)
        self.parser = NectoAction()

    def act(self, state):
        gamestate = state
        state = tuple(torch.from_numpy(s).float() for s in state)
        with torch.no_grad():
            out, weights = self.actor(state)

        max_shape = max(o.shape[-1] for o in out)
        logits = torch.stack(
            [
                l
                if l.shape[-1] == max_shape
                else F.pad(l, pad=(0, max_shape - l.shape[-1]), value=float("-inf"))
                for l in out
            ]
        ).swapdims(0, 1).squeeze()

        actions = np.argmax(logits, axis=-1)
        x = self.parser.parse_actions(actions[0], gamestate)

        return x[0]
