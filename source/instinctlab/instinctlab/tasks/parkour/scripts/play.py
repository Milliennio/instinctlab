"""Script to play a checkpoint if an RL agent from Instinct-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import csv
import math
import os
import subprocess
import sys
import time

sys.path.append(os.path.join(os.getcwd(), "scripts", "instinct_rl"))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with Instinct-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=3000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_start_step", type=int, default=0, help="Start step for the simulation.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--exportonnx", action="store_true", default=False, help="Export policy as ONNX model.")
parser.add_argument("--useonnx", action="store_true", default=False, help="Use the exported ONNX model for inference.")
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
parser.add_argument("--no_resume", default=None, action="store_true", help="Force play in no resume mode.")
# custom play arguments
parser.add_argument("--env_cfg", action="store_true", default=False, help="Load configuration from file.")
parser.add_argument("--agent_cfg", action="store_true", default=False, help="Load configuration from file.")
parser.add_argument("--sample", action="store_true", default=False, help="Sample actions instead of using the policy.")
parser.add_argument("--zero_act_until", type=int, default=0, help="Zero actions until this timestep.")
parser.add_argument("--keyboard_control", action="store_true", default=False, help="Enable keyboard control.")
parser.add_argument("--keyboard_linvel_step", type=float, default=0.5, help="Linear velocity change per keyboard step.")
parser.add_argument("--keyboard_angvel", type=float, default=1.0, help="Angular velocity set by keyboard.")
parser.add_argument(
    "--terrain_name",
    type=str,
    default=None,
    help="Force a single sub-terrain type for play, e.g. pyramid_stairs.",
)
parser.add_argument(
    "--terrain_level",
    type=int,
    default=None,
    help="Optionally force the single environment onto a specific terrain difficulty row.",
)
parser.add_argument(
    "--log_moe_gate",
    action="store_true",
    default=False,
    help="Log MoE gate weights during play when actor_moe_gate.onnx is available.",
)
parser.add_argument(
    "--moe_gate_print_interval",
    type=float,
    default=0.5,
    help="Console print interval in seconds for MoE gate weights.",
)

# append Instinct-RL cli arguments
cli_args.add_instinct_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import carb.input
import omni.appwindow
from carb.input import KeyboardEventType
from instinct_rl.runners import OnPolicyRunner
from instinct_rl.utils.utils import get_obs_slice, get_subobs_by_components, get_subobs_size

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict

##############################
# from isaaclab.utils.io import load_pickle, load_yaml
import pickle
import yaml

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def load_yaml(filename):
    with open(filename, "r") as f:
        return yaml.safe_load(f)
##############################

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg

# wait for attach if in debug mode
if args_cli.debug:
    # import typing; typing.TYPE_CHECKING = True
    import debugpy

    ip_address = ("0.0.0.0", 6789)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()


def force_single_subterrain(env_cfg, terrain_name: str):
    terrain_generator = getattr(env_cfg.scene.terrain, "terrain_generator", None)
    if terrain_generator is None:
        raise ValueError("This task does not use a terrain generator, so --terrain_name is not applicable.")
    if terrain_name not in terrain_generator.sub_terrains:
        available = ", ".join(terrain_generator.sub_terrains.keys())
        raise ValueError(f"Unknown terrain '{terrain_name}'. Available sub-terrains: {available}")
    sub_terrain_cfg = copy.deepcopy(terrain_generator.sub_terrains[terrain_name])
    if hasattr(sub_terrain_cfg, "proportion"):
        sub_terrain_cfg.proportion = 1.0
    terrain_generator.sub_terrains = {terrain_name: sub_terrain_cfg}
    terrain_generator.num_cols = 1
    terrain_generator.curriculum = False
    base_velocity_cmd = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
    if base_velocity_cmd is not None:
        if getattr(base_velocity_cmd, "velocity_ranges", None) is not None:
            if terrain_name in base_velocity_cmd.velocity_ranges:
                base_velocity_cmd.velocity_ranges = {
                    terrain_name: copy.deepcopy(base_velocity_cmd.velocity_ranges[terrain_name])
                }
            else:
                base_velocity_cmd.velocity_ranges = None
        if getattr(base_velocity_cmd, "random_velocity_terrain", None) is not None:
            base_velocity_cmd.random_velocity_terrain = [
                name for name in base_velocity_cmd.random_velocity_terrain if name == terrain_name
            ] or None


def set_forced_terrain_level(env, terrain_level: int):
    terrain = env.unwrapped.scene.terrain
    if terrain is None:
        raise ValueError("Cannot set terrain level because the scene has no terrain.")
    max_level = terrain.terrain_origins.shape[0] - 1
    level = int(max(0, min(terrain_level, max_level)))
    terrain.terrain_levels[:] = level
    terrain.terrain_types[:] = 0
    terrain.env_origins[:] = terrain.terrain_origins[level, 0]
    print(f"[INFO] Forced terrain level to row {level} / {max_level}.")


class MoeGateLogger:
    def __init__(self, log_dir: str, print_interval: float):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"moe_gate_play_{timestamp}.csv")
        self.file = open(self.log_path, "w", newline="", buffering=1)
        self.writer = csv.writer(self.file)
        self.print_interval = print_interval
        self.start_time = time.time()
        self.last_print_time = 0.0
        self.step = 0
        self.header_written = False
        print(f"[INFO] Logging MoE gate weights to: {self.log_path}")

    def record(self, gate_weights):
        if gate_weights is None:
            return
        gate_weights = gate_weights[0].tolist()
        now = time.time()
        elapsed = now - self.start_time
        top1 = max(range(len(gate_weights)), key=lambda i: gate_weights[i])
        max_weight = gate_weights[top1]
        entropy = -sum(weight * math.log(weight + 1e-8) for weight in gate_weights)
        if not self.header_written:
            header = ["time_s", "wall_time", "step"]
            header += [f"expert_{i}" for i in range(len(gate_weights))]
            header += ["top1", "max_weight", "entropy"]
            self.writer.writerow(header)
            self.header_written = True
        row = [f"{elapsed:.6f}", f"{now:.6f}", self.step]
        row += [f"{weight:.8f}" for weight in gate_weights]
        row += [top1, f"{max_weight:.8f}", f"{entropy:.8f}"]
        self.writer.writerow(row)
        if now - self.last_print_time >= self.print_interval:
            print(
                "[INFO] MoE gate:"
                f" top1={top1}, max_weight={max_weight:.4f},"
                f" weights={[round(weight, 4) for weight in gate_weights]}"
            )
            self.last_print_time = now
        self.step += 1

    def close(self):
        self.file.close()


def main():
    """Play with Instinct-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)
    if args_cli.terrain_name is not None:
        if args_cli.num_envs not in (None, 1):
            raise ValueError("--terrain_name currently requires --num_envs=1.")
        env_cfg.scene.num_envs = 1
        force_single_subterrain(env_cfg, args_cli.terrain_name)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "instinct_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    agent_cfg.load_run = args_cli.load_run
    if agent_cfg.load_run is not None:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if os.path.isabs(agent_cfg.load_run):
            resume_path = get_checkpoint_path(
                os.path.dirname(agent_cfg.load_run), os.path.basename(agent_cfg.load_run), agent_cfg.load_checkpoint
            )
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
    elif not args_cli.no_resume:
        raise RuntimeError(
            f"\033[91m[ERROR] No checkpoint specified and play.py resumes from a checkpoint by default. Please specify"
            f" a checkpoint to resume from using --load_run or use --no_resume to disable this behavior.\033[0m"
        )
    else:
        print(f"[INFO] No experiment directory specified. Using default: {log_root_path}")
        log_dir = os.path.join(log_root_path, agent_cfg.run_name + "_play")
        resume_path = "model_scratch.pt"

    if args_cli.env_cfg:
        env_cfg = load_pickle(os.path.join(log_dir, "params", "env.pkl"))
    if args_cli.agent_cfg:
        agent_cfg_dict = load_yaml(os.path.join(log_dir, "params", "agent.yaml"))
    else:
        agent_cfg_dict = agent_cfg.to_dict()

    if args_cli.keyboard_control:
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 1e10

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.terrain_level is not None:
        if env.unwrapped.num_envs != 1:
            raise ValueError("--terrain_level currently requires a single environment.")
        set_forced_terrain_level(env, args_cli.terrain_level)
        env.reset()
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == args_cli.video_start_step,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": f"model_{resume_path.split('_')[-1].split('.')[0]}",
        }
        print("[INFO] Recording videos during playing.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for instinct-rl
    env = InstinctRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    if agent_cfg.load_run is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    if args_cli.sample:
        policy = ppo_runner.alg.actor_critic.act
    else:
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    if agent_cfg.load_run is not None:
        export_model_dir = os.path.join(log_dir, "exported")
        if args_cli.exportonnx:
            assert env.unwrapped.num_envs == 1, "Exporting to ONNX is only supported for single environment."
            if not os.path.exists(export_model_dir):
                os.makedirs(export_model_dir)
            obs, _ = env.get_observations()
            ppo_runner.alg.actor_critic.export_as_onnx(obs, export_model_dir)

    # use the exported model for inference
    if args_cli.useonnx:
        from onnxer import load_parkour_onnx_model

        # NOTE: This is only applicable with parkour task
        onnx_policy = load_parkour_onnx_model(
            model_dir=os.path.join(log_dir, "exported"),
            get_subobs_func=lambda obs: get_subobs_by_components(
                obs,
                agent_cfg.policy.encoder_configs.depth_encoder.component_names,
                env.get_obs_segments(),
                temporal=True,
            ),
            depth_shape=env.get_obs_segments()["depth_image"],
            proprio_slice=slice(
                0,
                get_subobs_size(
                    env.get_obs_segments(),
                    [
                        "base_lin_vel",
                        "base_ang_vel",
                        "projected_gravity",
                        "velocity_commands",
                        "joint_pos",
                        "joint_vel",
                        "actions",
                    ],
                ),
            ),
        )
    else:
        onnx_policy = None

    moe_gate_logger = None
    if args_cli.log_moe_gate:
        if not args_cli.useonnx:
            print("[WARN] --log_moe_gate currently logs ONNX gate weights only. Use it together with --useonnx.")
        elif getattr(onnx_policy, "actor_gate", None) is None:
            print("[WARN] actor_moe_gate.onnx not found. MoE gate logging is disabled.")
        else:
            moe_gate_logger = MoeGateLogger(os.path.join(log_dir, "play_logs"), args_cli.moe_gate_print_interval)

    override_command = torch.zeros(env.num_envs, 3, device=env.device)
    command_obs_slice = get_obs_slice(env.get_obs_segments(), "velocity_commands")

    def on_keyboard_input(e):
        if e.input == carb.input.KeyboardInput.W:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                override_command[:, 0] += args_cli.keyboard_linvel_step
        if e.input == carb.input.KeyboardInput.S:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                override_command[:, 2] = 0.0
        if e.input == carb.input.KeyboardInput.F:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                override_command[:, 2] = args_cli.keyboard_angvel
        if e.input == carb.input.KeyboardInput.G:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                override_command[:, 2] = -args_cli.keyboard_angvel
        if e.input == carb.input.KeyboardInput.X:
            if e.type == KeyboardEventType.KEY_PRESS or e.type == KeyboardEventType.KEY_REPEAT:
                override_command[:] = 0.0

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input = carb.input.acquire_input_interface()
    input.subscribe_to_keyboard_events(keyboard, on_keyboard_input)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            if args_cli.keyboard_control:
                obs[:, command_obs_slice[0]] = override_command.repeat(1, command_obs_slice[1][0] // 3)
            actions = policy(obs)
            if args_cli.useonnx:
                torch_actions = actions
                actions = onnx_policy(obs)
                if moe_gate_logger is not None:
                    moe_gate_logger.record(onnx_policy.last_gate_weights)
                if (actions - torch_actions).abs().max() > 1e-5:
                    print(
                        "[INFO]: ONNX model and PyTorch model have a difference of"
                        f" {(actions - torch_actions).abs().max()} in actions at joint"
                        f" {((actions - torch_actions).abs() > 1e-5).nonzero(as_tuple=True)[0]}"
                    )
            if timestep < args_cli.zero_act_until:
                actions[:] = 0.0
            # env stepping
            obs, rewards, dones, infos = env.step(actions)
        timestep += 1

        # exit the loop if video_length is meet
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    if moe_gate_logger is not None:
        moe_gate_logger.close()
    env.close()

    if args_cli.video:
        subprocess.run(
            [
                "code",
                "-r",
                os.path.join(log_dir, "videos", "play", f"model_{resume_path.split('_')[-1].split('.')[0]}-step-0.mp4"),
            ]
        )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
