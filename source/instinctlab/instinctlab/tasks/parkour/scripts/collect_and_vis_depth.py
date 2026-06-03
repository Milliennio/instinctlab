"""Script to collect and visualize depth_image observations from an RL agent."""

import argparse
import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.join(os.getcwd(), "scripts", "instinct_rl"))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect and visualize depth images from Instinct-RL observation.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--no_resume", default=None, action="store_true", help="Force play in no resume mode.")
parser.add_argument("--save_data", action="store_true", default=False, help="Enable saving collected depth data to disk.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")

# append Instinct-RL cli arguments
cli_args.add_instinct_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import pickle
import yaml

from instinct_rl.runners import OnPolicyRunner
from instinct_rl.utils.utils import get_obs_slice, get_subobs_by_components

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def load_yaml(filename):
    with open(filename, "r") as f:
        return yaml.safe_load(f)

def main():
    """Play and collect observation data."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.join("logs", "instinct_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    agent_cfg.load_run = args_cli.load_run
    
    if agent_cfg.load_run is not None:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if os.path.isabs(agent_cfg.load_run):
            resume_path = get_checkpoint_path(os.path.dirname(agent_cfg.load_run), os.path.basename(agent_cfg.load_run), agent_cfg.load_checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
    elif not args_cli.no_resume:
        raise RuntimeError("[ERROR] No checkpoint specified.")
    else:
        log_dir = os.path.join(log_root_path, agent_cfg.run_name + "_play")
        resume_path = "model_scratch.pt"

    agent_cfg_dict = load_yaml(os.path.join(log_dir, "params", "agent.yaml"))

    # Set up data saving directory
    save_dir = os.path.join(log_dir, "collected_data")
    if args_cli.save_data:
        os.makedirs(save_dir, exist_ok=True)
        print(f"[INFO] Data will be saved to: {save_dir}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = InstinctRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    if agent_cfg.load_run is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Get observation segments to parse the flattened obs tensor
    obs_segments = env.get_obs_segments()
    assert "depth_image" in obs_segments, "No depth_image found in observations!"
    
    depth_slice = get_obs_slice(obs_segments, "depth_image")
    
    depth_shape = obs_segments["depth_image"]  # Usually (History, Height, Width), e.g., (8, 18, 32)
    print(f"[INFO] Found depth_image at slice {depth_slice} with shape {depth_shape}")

    obs, _ = env.get_observations()
    timestep = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            # Step the policy
            actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)

            # 1. 调用 instinct_rl 底层完全相同的自动切片和重组函数 (开启 temporal=True)
            # 这与 ParallelLayer 将数据喂给 TransformerHeadModel 时的内部操作 100% 相同
            depth_3d = get_subobs_by_components(
                obs, ["depth_image"], obs_segments, temporal=True
            )
            
            # 2. 为了可视化，再将 Transformer 的特征维度 (576) 还原回 OpenCV 需要的二维空间维度 (Height, Width)
            depth_image = depth_3d.reshape(env.num_envs, *depth_shape).cpu().numpy()

            # Target Env 0 for visualization
            # Typically history dimension has the latest frame at index -1
            # Shape of latest_depth_frame: (Height, Width)
            latest_depth_frame = depth_image[0, -1]
            
            # Convert normalized depth [0.0, 1.0] to visualizable uint8 [0, 255]
            vis_img = (np.clip(latest_depth_frame, 0.0, 1.0) * 255).astype(np.uint8)
            
            # Scale up the original image (e.g. 18x32) to larger resolution via Nearest Neighbor 
            # so we can clearly see the exact individual pixels network receives
            vis_img_large = cv2.resize(vis_img, (640, 360), interpolation=cv2.INTER_NEAREST)

            cv2.imshow("Collected Depth Observation - Env 0", vis_img_large)
            
            # Use waitKey to update the image window and allow manual interruption
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if args_cli.save_data and timestep % 10 == 0:
                # Save current timestep depth data (Full history)
                np.save(os.path.join(save_dir, f"depth_obs_step_{timestep}.npy"), depth_image[0])
                
        timestep += 1

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    simulation_app.close()