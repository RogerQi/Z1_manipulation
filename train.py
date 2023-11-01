from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import numpy as np
import torch
import os

import torch.nn as nn

from skrl.envs.torch import wrap_env
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from envs.b1_cabinet import B1Cabinet
from envs.b1_door import B1Door
from envs.b1_pick import B1Pick
from envs.b1_pickplace import B1PickPlace
from envs.b1_pickplacescan import B1PickPlaceScan
from envs.b1_pickcamera import B1PickCamera
from envs.b1_picklowfreq import B1PickLowFreq
from envs.b1_z1pickfix import B1Z1PickFix
from envs.b1_z1pick import B1Z1Pick
from envs.b1_pickycb import B1PickYCB

from utils.config import load_cfg, get_params
import utils.wrapper as wrapper

set_seed(43)

# define models (stochastic and deterministic models) using mixins
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}
        
def create_env(cfg, args):
    cfg["env"]["enableDebugVis"] = args.debugvis
    _env = eval(args.task)(cfg=cfg, rl_device=args.rl_device, sim_device=args.sim_device, 
                         graphics_device_id=args.graphics_device_id, headless=args.headless)
    wrapped_env = wrapper.IsaacGymPreview3Wrapper(_env)
    return wrapped_env

def main():
    args = get_params()
    cfg_file = "b1_" + args.task[2:].lower() + ".yaml"
    file_path = "data/cfg/" + cfg_file
    cfg = load_cfg(file_path)
    env = create_env(cfg=cfg, args=args)
    device = env.rl_device
    memory = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)
    
    models_ppo = {}
    models_ppo["policy"] = Policy(env.observation_space, env.action_space, device)
    models_ppo["value"] = Value(env.observation_space, env.action_space, device)
    
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo["rollouts"] = 24  # memory_size
    cfg_ppo["learning_epochs"] = 5
    cfg_ppo["mini_batches"] = 6  # 24 * 8192 / 32768
    cfg_ppo["discount_factor"] = 0.99
    cfg_ppo["lambda"] = 0.95
    cfg_ppo["learning_rate"] = 5e-4
    cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
    cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg_ppo["random_timesteps"] = 0
    cfg_ppo["learning_starts"] = 0
    cfg_ppo["grad_norm_clip"] = 1.0
    cfg_ppo["ratio_clip"] = 0.2
    cfg_ppo["value_clip"] = 0.2
    cfg_ppo["clip_predicted_values"] = True
    cfg_ppo["value_loss_scale"] = 1.0
    cfg_ppo["kl_threshold"] = 0
    cfg_ppo["rewards_shaper"] = None
    cfg_ppo["state_preprocessor"] = RunningStandardScaler
    cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg_ppo["value_preprocessor"] = RunningStandardScaler
    cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    # logging to TensorBoard and write checkpoints each 120 and 1200 timesteps respectively
    cfg_ppo["experiment"]["write_interval"] = 120
    cfg_ppo["experiment"]["checkpoint_interval"] = 1200
    cfg_ppo["experiment"]["directory"] = args.experiment_dir
    cfg_ppo["experiment"]["experiment_name"] = args.wandb_name
    cfg_ppo["experiment"]["wandb"] = args.wandb
    if args.wandb:
        cfg_ppo["experiment"]["wandb_kwargs"] = {"project": args.wandb_project, "tensorboard": False, "name": args.wandb_name}
        
    agent = PPO(models=models_ppo,
            memory=memory,
            cfg=cfg_ppo,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)
    
    if args.checkpoint:
        agent.load(args.checkpoint)
        
    cfg_trainer = {"timesteps": args.timesteps, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    
    trainer.train()
    
if __name__ == "__main__":
    main()
    