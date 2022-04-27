import torch
import os

from rocket_learn.agent.discrete_policy import DiscretePolicy
from torch.nn import Linear, Sequential, ReLU
from rocket_learn.utils.util import SplitLayer


# TODO add your network here
def get_actor(_split, _state_dim):
    return DiscretePolicy(Sequential(
        Linear(_state_dim, 512),
        ReLU(),
        Linear(512, 512),
        ReLU(),
        Linear(512, sum(_split)),
        SplitLayer(splits=_split)
    ), _split)


# TODO set your split and obs length
split = (90,)

# TOTAL SIZE OF THE INPUT DATA
state_dim = 231 + (8*5)

actor = get_actor(split, state_dim)

# PPO REQUIRES AN ACTOR/CRITIC AGENT

cur_dir = os.path.dirname(os.path.realpath(__file__))
checkpoint = torch.load(os.path.join(cur_dir, "checkpoint.pt"))
actor.load_state_dict(checkpoint['actor_state_dict'])
actor.eval()
torch.jit.save(torch.jit.script(actor), 'jit.pt')

exit(0)