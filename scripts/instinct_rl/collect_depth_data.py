"""Script to collect depth data with an RL agent from Instinct-RL."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect depth data with Instinct-RL.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--collect_steps", type=int, default=5000, help="Number of steps to collect.")
parser.add_argument("--save_dir", type=str, default="data_collection", help="Directory to save data.")

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
import numpy as np
import torch

from instinct_rl.runners import OnPolicyRunner
from instinct_rl.utils.utils import get_obs_slice

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.io import load_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
import instinctlab.tasks  # noqa: F401
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg

def main():
    # parse configuration directly from python file to ensure new sensors are included
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.join("logs", "instinct_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    if agent_cfg.load_run is not None:
        if os.path.isabs(agent_cfg.load_run):
            resume_path = get_checkpoint_path(
                os.path.dirname(agent_cfg.load_run), os.path.basename(agent_cfg.load_run), agent_cfg.load_checkpoint
            )
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
    else:
        raise RuntimeError("Please specify a checkpoint to load using --load_run.")

    agent_cfg_dict = load_yaml(os.path.join(log_dir, "params", "agent.yaml"))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for instinct-rl
    env = InstinctRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner.load(resume_path)

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    os.makedirs(args_cli.save_dir, exist_ok=True)

    obs, _ = env.get_observations()
    
    # 提取深度图在总观测中的切片位置
    depth_slice = get_obs_slice(env.get_obs_segments(), "depth_image")
    proprio_slice = slice(0, depth_slice[0].start)
    
    print("[INFO] Starting data collection...")
    
    depth_history_list = []
    proprio_list = []
    gt_elevation_list = []
    
    timestep = 0
    chunk_id = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, infos = env.step(actions)
            
            # 1. 获取网络输入的 8 帧历史降频深度图
            # Shape: (num_envs, 8 frames * 18 H * 32 W) -> 重塑为 (num_envs, 8, 18, 32)
            depth_history = obs[:, depth_slice[0]].view(env.num_envs, 8, 18, 32).cpu().numpy()
            
            # 2. 获取本体感觉状态
            proprio = obs[:, proprio_slice].cpu().numpy()
            
            # 3. 提取盲区真实高程图 (GT Elevation Map)
            raycaster = env.unwrapped.scene["blind_spot_gt"]
            robot = env.unwrapped.scene["robot"]
            
            # 射线击中点的世界坐标 Z 轴高度 (num_envs, 651)
            ground_z = raycaster.data.ray_hits_w[..., 2]
            # 机器人根节点的世界坐标 Z 轴高度 (num_envs, 1)
            base_z = robot.data.root_pos_w[:, 2].unsqueeze(1)
            
            # [修复维度和旋转问题] 
            # IsaacLab 的 GridPatternCfg 默认使用 xy-indexing，
            # 展平前在内存中的形状是 (num_y, num_x) = (21, 31)。
            # 先 reshape 回 (21, 31) 消除步长错位导致的规律性杂乱。
            gt_elevation = (ground_z - base_z).view(env.num_envs, 21, 31)
            # 翻转坐标轴并转置为 (31, 21)，使得：
            # 图像上方代表正前方 (X+)；图像左侧代表左侧 (Y+)
            gt_elevation = torch.flip(gt_elevation, dims=(1, 2)).transpose(1, 2).cpu().numpy()
            
            # 过滤掉当前帧正好重置的环境（避免不连续的坏数据）
            valid_mask = (dones == 0).cpu().numpy()
            
            depth_history_list.append(depth_history[valid_mask])
            proprio_list.append(proprio[valid_mask])
            gt_elevation_list.append(gt_elevation[valid_mask])
            
            timestep += 1
            
            # 每 1000 步保存一次 chunk，防止内存溢出
            if timestep % 1000 == 0:
                save_path = os.path.join(args_cli.save_dir, f"dataset_chunk_{chunk_id}.npz")
                np.savez(
                    save_path,
                    depth_history=np.concatenate(depth_history_list, axis=0),
                    proprio=np.concatenate(proprio_list, axis=0),
                    gt_elevation=np.concatenate(gt_elevation_list, axis=0)
                )
                print(f"[INFO] Saved {save_path} with {sum(len(x) for x in depth_history_list)} valid samples.")
                
                depth_history_list.clear()
                proprio_list.clear()
                gt_elevation_list.clear()
                chunk_id += 1
                
            if timestep >= args_cli.collect_steps:
                break

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()