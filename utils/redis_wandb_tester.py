import os
import time

import wandb

from redis import Redis

from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator, _unserialize, QUALITIES
from trueskill import Rating


import numpy as np
import plotly.graph_objs as go
from scipy import signal


def _plot_ratings(ratings, logger_):
    if len(ratings) <= 0:
        return
    mus = np.array([r.mu for r in ratings])
    mus = mus - mus[0]
    sigmas = np.array([r.sigma for r in ratings])
    # sigmas[1:] = (sigmas[1:] ** 2 + sigmas[0] ** 2) ** 0.5

    x = np.arange(len(mus))
    y = mus
    y_upper = mus + 2 * sigmas  # 95% confidence
    y_lower = mus - 2 * sigmas

    fig = go.Figure([
        go.Scatter(
            x=x,
            y=y,
            line=dict(color='rgb(0,100,80)'),
            mode='lines',
            name="mu",
            showlegend=False
        ),
        go.Scatter(
            x=np.concatenate((x, x[::-1])),  # x, then x reversed
            y=np.concatenate((y_upper, y_lower[::-1])),  # upper, then lower reversed
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name="sigma",
            showlegend=False
        ),
    ])

    fig.update_layout(title="Rating", xaxis_title="Iteration", yaxis_title="TrueSkill")

    fig_smooth = go.Figure([
        go.Scatter(
            x=x,
            y=signal.medfilt(y, 21),
            line=dict(color='rgb(175, 79, 219)'),
            mode='lines',
            name='Smoothed',
            showlegend=False,
        )
    ])

    fig_smooth.update_layout(title="Smoothed Rating", xaxis_title="Iteration", yaxis_title="TrueSkill")

    logger_.log({
        "qualities": fig,
        "qualities_smooth": fig_smooth,
    }, commit=False)


if __name__ == "__main__":
    run_id = "Test"
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(dir="wandb_store",
                        name="test_smoothing",
                        project="testing",
                        entity="kaiyotech",
                        id=run_id,
                        )

    redis = Redis(username="user1", password=os.environ["redis_user1_key"])  # host="192.168.0.201",
    _plot_ratings([Rating(*_unserialize(v)) for v in redis.lrange(QUALITIES, 0, -1)], logger)


