from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import numpy as np
import torch
import os

import torch.nn as nn

from envs.b1_picklowfreq import B1PickLowFreq
from envs.b1_manip import B1Manip

from utils.config import load_cfg, get_params
import utils.wrapper as wrapper


def create_env(cfg, args):
    _env = eval(args.task)(cfg=cfg, rl_device=args.rl_device, sim_device=args.sim_device, 
                         graphics_device_id=args.graphics_device_id, headless=args.headless)
    wrapped_env = wrapper.IsaacGymPreview3Wrapper(_env)
    return wrapped_env


if __name__ == "__main__":
    args = get_params()
    cfg_file = "b1_" + args.task[2:].lower() + ".yaml"
    file_path = "data/cfg/" + cfg_file
    cfg = load_cfg(file_path)
    cfg["env"]["numEnvs"] = 1
    cfg["env"]["enableDebugVis"] = args.debugvis
    cfg["env"]["saveCameraImage"] = args.save_image
    args.headless = False
    env = create_env(cfg, args)
    device = env.rl_device
    
    obs, _ = env.reset()
    
    for _ in range(6000):
        action = torch.zeros((1, 9), device=device)
        obs, rwd, reset, truncated, extra = env.step(action)
        env.render("human")
        if reset.any() or truncated.any():
            states, info = env.reset()
    