#!/bin/env python
import gym
from gym.spaces import Box, Discrete
import macad_gym  # noqa F401

# import modules from macad_agents directly
import rllib
import a3c

from rllib.env_wrappers import wrap_deepmind
from rllib.models import register_mnih15_shared_weights_net

#native ray import 
import ray
from ray import tune
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from ray.tune import run_experiments
from ray.tune.registry import register_env


# additional useful imports:
import math
import random
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count

# not using torch(stable_baselines3) for this one
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

# if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#env = gym.make("HomoNcomIndePOIntrxMASS3CTWN3-v0")
##### The following is not used:
#configs = env.configs
#env_config = configs["env"]
#actor_configs = configs["actors"]


# Placeholder to enable use of a custom pre-processor
class ImagePreproc(Preprocessor):
    def _init_shape(self, obs_space, options):
        shape = (84, 84, 3)  # Adjust third dim if stacking frames
        return shape

    def transform(self, observation):
        return observation

ModelCatalog.register_custom_preprocessor("sq_im_84", ImagePreproc)


ray.init()

obs_space = Box(0.0, 255.0, shape=(84, 84, 3))
act_space = Discrete(9)


num_iters = 20         #
num_workers = 1        #Num workers (CPU cores) to use
num_gpus = 1           #
sample_bs_per_worker = 50   # Number of samples in a batch per worker. Default=50
train_bs = 150              # Train batch size. Use as per available GPU mem. Default=500   ----must>=128
envs_per_worker = 1         # Number of env instances per worker. Default=10
notes = None                # Custom experiment description to be added to comet logs

register_mnih15_shared_weights_net()        #??
model_name = "mnih15_shared_weights"

env_name = "HighwayCross3Car-v0"
env = gym.make(env_name)
env_actor_configs = env.configs
num_framestack = env_actor_configs["env"]["framestack"]


def env_creator(env_config):
    import macad_gym
    env = gym.make("HighwayCross3Car-v0")
    # Apply wrappers to: convert to Grayscale, resize to 84 x 84,
    # stack frames & some more op
    env = wrap_deepmind(env, dim=84, num_framestack=num_framestack)
    return env


register_env(env_name, lambda config: env_creator(config))


def gen_policy():
    config = {
        # Model and preprocessor options.
        "model": {
            "custom_model": model_name,
            "custom_options": {
                # Custom notes for the experiment
                "notes": {
                    "notes": notes
                },
            },
            # NOTE:Wrappers are applied by RLlib if custom_preproc is NOT
            # specified
            "custom_preprocessor": "sq_im_84",
            "dim": 84,
            "free_log_std": False,  # if discrete_actions else True,
            "grayscale": True,
            # conv_filters to be used with the custom CNN model.
            # "conv_filters": [[16, [4, 4], 2], [32, [3, 3], 2],
            # [16, [3, 3], 2]]
        },
        # preproc_pref is ignored if custom_preproc is specified
        # "preprocessor_pref": "deepmind",

        # env_config to be passed to env_creator
        "env_config": env_actor_configs
    }
    return (PPOPolicyGraph, obs_space, act_space, config)

policy_graphs = {
    a_id: gen_policy()
    for a_id in env_actor_configs["actors"].keys()
}

run_experiments({
    "MA-PPO-SSUI3CCARLA": {
        "run": "PPO",
        "env": env_name,
        "stop": {
            "training_iteration": num_iters
        },
        "config": {
            "log_level": "DEBUG",
            "num_sgd_iter": 10,
            "multiagent": {
                "policy_graphs": policy_graphs,
                "policy_mapping_fn":
                tune.function(lambda agent_id: agent_id),
            },
            "num_workers": num_workers,
            "num_envs_per_worker": envs_per_worker,
            "sample_batch_size": sample_bs_per_worker,
            "train_batch_size": train_bs
        },
        "checkpoint_freq": 5,
        "checkpoint_at_end": True,
    }
})
