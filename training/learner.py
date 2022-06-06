import os
import wandb

import torch.jit
from torch.nn import Linear, Sequential, ReLU

from redis import Redis

from training.rewards import CoyoteReward
from utils.nectoparser import NectoAction

from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from training.agent import DiscretePolicy
from training.my_ppo import PPO
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator
from rocket_learn.utils.util import SplitLayer
from utils.mybots_obs import ExpandAdvancedPaddedObs

from training.Constants import *
from utils.misc import count_parameters


if __name__ == "__main__":
    config = dict(
        gamma=1 - (T_STEP / TIME_HORIZON),
        gae_lambda=0.95,
        learning_rate_critic=1e-4,
        learning_rate_actor=1e-4,
        ent_coef=0.01,
        vf_coef=1.,
        target_steps=2_000_000,  # testing 2M normal
        batch_size=400_000,  # testing 400k normal
        minibatch_size=200_000,  # testing 200k normal
        n_bins=3,
        n_epochs=30,
        iterations_per_save=1,
    )
    run_id = "RunV5_14"
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(dir="wandb_store",
                        name="CoyoteV5",
                        project="Coyote",
                        entity="kaiyotech",
                        id=run_id,
                        config=config,
                        )

    redis = Redis(username="user1", password=os.environ["redis_user1_key"])  # host="192.168.0.201",
    redis.delete("worker-ids")

    # ENSURE OBSERVATION, REWARD, AND ACTION CHOICES ARE THE SAME IN THE WORKER
    def obs():
        return ExpandAdvancedPaddedObs()

    def rew():
        return CoyoteReward()

    def act():
        return NectoAction()  # KBMAction(n_bins=N_BINS)

    # THE ROLLOUT GENERATOR CAPTURES INCOMING DATA THROUGH REDIS AND PASSES IT TO THE LEARNER.
    # -save_every SPECIFIES HOW OFTEN OLD VERSIONS ARE SAVED TO REDIS. THESE ARE USED FOR TRUESKILL
    # COMPARISON AND TRAINING AGAINST PREVIOUS VERSIONS
    rollout_gen = RedisRolloutGenerator(redis, obs, rew, act,
                                        logger=logger,
                                        save_every=logger.config.iterations_per_save,
                                        clear=False,  # update this if starting over
                                        )

    # ROCKET-LEARN EXPECTS A SET OF DISTRIBUTIONS FOR EACH ACTION FROM THE NETWORK, NOT
    # THE ACTIONS THEMSELVES. SEE network_setup.readme.txt FOR MORE INFORMATION
    # split = (3, 3, 3, 3, 3,  2, 2, 2)
    split = (90,)  # updated for Necto parser
    total_output = sum(split)

    # TOTAL SIZE OF THE INPUT DATA
    # 107+stack_size*actions
    state_dim = 231  # + (8*5)    # normal is 107

    critic = Sequential(
        Linear(state_dim, 1024),
        ReLU(),
        Linear(1024, 1024),
        ReLU(),
        Linear(1024, 1)
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
        zero_grads_with_none=True,
    )

    # alg.load("C:/Users/kchin/code/Kaiyotech/abad/checkpoint_save_directory/Coyote_1650839805.8645337/Coyote_240/checkpoint.pt")
    alg.load("checkpoint_save_directory/Coyote_1654428720.9855726/Coyote_2086/checkpoint.pt")
    alg.agent.optimizer.param_groups[0]["lr"] = logger.config.learning_rate_actor
    alg.agent.optimizer.param_groups[1]["lr"] = logger.config.learning_rate_critic

    # SPECIFIES HOW OFTEN CHECKPOINTS ARE SAVED
    alg.run(iterations_per_save=logger.config.iterations_per_save, save_dir="checkpoint_save_directory")
