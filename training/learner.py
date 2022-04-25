import os
import wandb

import torch.jit
from torch.nn import Linear, Sequential, ReLU

from redis import Redis

from training.rewards import anneal_rewards_fn, MyRewardFunction
from utils.nectoparser import NectoAction

from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from training.agent import DiscretePolicy
from training.my_ppo import PPO
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator
from rocket_learn.utils.util import SplitLayer
from utils.mybots_obs import ExpandAdvancedPaddedStackObs

from training.Constants import *
from utils.misc import count_parameters


if __name__ == "__main__":
    config = dict(
        gamma=1 - (T_STEP / TIME_HORIZON),
        gae_lambda=0.95,
        learning_rate_critic=1e-4,
        learning_rate_actor=1e-4,
        ent_coef=0.005,
        vf_coef=1.,
        target_steps=1_000_000,
        batch_size=200_000,
        minibatch_size=100_000,
        n_bins=3,
        n_epochs=30,
        iterations_per_save=10
    )
    run_id = "Run2"
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(dir="wandb_store", name="CoyoteV2_1", project="Coyote", entity="kaiyotech", id=run_id, config=config)

    redis = Redis(username="user1", password=os.environ["redis_user1_key"])

    # ENSURE OBSERVATION, REWARD, AND ACTION CHOICES ARE THE SAME IN THE WORKER
    def obs():
        return ExpandAdvancedPaddedStackObs(stack_size=5, team_size=3)

    def rew():
        return MyRewardFunction(
            team_spirit=0,
            goal_w=7,
            aerial_goal_w=10,
            double_tap_goal_w=0,
            shot_w=0.8,
            save_w=1.2,
            demo_w=1,
            above_w=0,
            got_demoed_w=-1,
            behind_ball_w=0.05,
            save_boost_w=0.2,
            concede_w=-7,
            velocity_w=0.05,
            velocity_pb_w=0.5,
            velocity_bg_w=2,
            aerial_ball_touch_w=15,
            kickoff_w=1,
            ball_touch_w=0,
        )

    def act():
        return NectoAction()  # KBMAction(n_bins=N_BINS)

    # THE ROLLOUT GENERATOR CAPTURES INCOMING DATA THROUGH REDIS AND PASSES IT TO THE LEARNER.
    # -save_every SPECIFIES HOW OFTEN OLD VERSIONS ARE SAVED TO REDIS. THESE ARE USED FOR TRUESKILL
    # COMPARISON AND TRAINING AGAINST PREVIOUS VERSIONS
    rollout_gen = RedisRolloutGenerator(redis, obs, rew, act,
                                        logger=logger,
                                        save_every=logger.config.iterations_per_save,
                                        clear=False,  # update this if rolling back
                                        )

    # ROCKET-LEARN EXPECTS A SET OF DISTRIBUTIONS FOR EACH ACTION FROM THE NETWORK, NOT
    # THE ACTIONS THEMSELVES. SEE network_setup.readme.txt FOR MORE INFORMATION
    # split = (3, 3, 3, 3, 3,  2, 2, 2)
    split = (90,)  # updated for Necto parser
    total_output = sum(split)

    # TOTAL SIZE OF THE INPUT DATA
    # 107+stack_size*actions
    state_dim = 231 + (8*5)    # normal is 107

    critic = Sequential(
        Linear(state_dim, 256),
        ReLU(),
        Linear(256, 256),
        ReLU(),
        Linear(256, 1)
    )

    actor = DiscretePolicy(Sequential(
        Linear(state_dim, 512),
        ReLU(),
        Linear(512, 512),
        ReLU(),
        Linear(512, total_output),
        SplitLayer(splits=split)
    ), split)

    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": logger.config.learning_rate_actor},
        {"params": critic.parameters(), "lr": logger.config.learning_rate_critic}
    ])

    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)

    params = count_parameters(agent)
    print(f"Number of parameters is {params}")
    # logger.config["Params"] = params
    # logger.

    alg = PPO(
        rollout_gen,
        agent,
        ent_coef=logger.config.ent_coef,
        n_steps=logger.config.target_steps,  # target steps per rollout?
        batch_size=logger.config.batch_size,
        minibatch_size=logger.config.minibatch_size,
        epochs=logger.config.n_epochs,
        gamma=logger.config.gamma,
        gae_lambda=logger.config.gae_lambda,
        vf_coef=logger.config.vf_coef,
        logger=logger,
        device="cuda",
    )

    # alg.load("C:/Users/kchin/code/Kaiyotech/abad/checkpoint_save_directory/Coyote_1650839805.8645337/Coyote_240/checkpoint.pt")
    alg.load("checkpoint_save_directory/Coyote_1650856936.9338183/Coyote_390/checkpoint.pt")

    # SPECIFIES HOW OFTEN CHECKPOINTS ARE SAVED
    alg.run(iterations_per_save=logger.config.iterations_per_save, save_dir="checkpoint_save_directory")
