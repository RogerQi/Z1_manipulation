import os
import yaml
import argparse

def load_cfg(file_path):
    with open(os.path.join(os.getcwd(), file_path), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
    return cfg

def get_params():
    parser = argparse.ArgumentParser(
        prog="Z1 training",
    )
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--timesteps", type=int, default=24000)
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=-1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="isaacgym")
    parser.add_argument("--wandb_name", type=str, default="isaacgym")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--experiment_dir", type=str, default="experiments")
    parser.add_argument("--debugvis", action="store_true")
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    
    return args
