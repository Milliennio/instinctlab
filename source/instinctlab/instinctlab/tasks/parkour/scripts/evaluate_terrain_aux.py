"""Batch evaluation for G1 Parkour TerrainAux / CrossAttn checkpoints."""

from __future__ import annotations

import argparse
import os
import sys
import time


def _append_instinct_rl_cli_path():
    current = os.path.abspath(os.path.dirname(__file__))
    for _ in range(10):
        candidate = os.path.join(current, "scripts", "instinct_rl")
        if os.path.exists(os.path.join(candidate, "cli_args.py")):
            sys.path.append(candidate)
            return
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    sys.path.append(os.path.join(os.getcwd(), "scripts", "instinct_rl"))


_append_instinct_rl_cli_path()

from isaaclab.app import AppLauncher  # noqa: E402

import cli_args  # isort: skip  # noqa: E402


parser = argparse.ArgumentParser(description="Evaluate TerrainAux / CrossAttn parkour checkpoints.")
parser.add_argument("--task", type=str, required=True, help="Eval task name.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=0, help="Evaluation seed.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--debug", action="store_true", default=False, help="Enable debugpy attach before evaluation.")

parser.add_argument("--eval_name", type=str, default=None, help="Name of this evaluation run.")
parser.add_argument("--model_name", type=str, default=None, help="Short model label written into metric files.")
parser.add_argument("--output_dir", type=str, default="logs/eval", help="Root directory for evaluation outputs.")

parser.add_argument(
    "--terrain_suite",
    type=str,
    default="mixed_stair",
    choices=("mixed_stair", "mixed_rough", "single"),
    help="Terrain grouping label. Use --terrain_name with terrain_suite=single.",
)
parser.add_argument("--terrain_name", type=str, default=None, help="Single terrain name for terrain_suite=single.")
parser.add_argument(
    "--terrain_levels",
    type=str,
    default="all",
    help="Comma-separated terrain levels, e.g. 0,1,2,3,4,5. Use all for all rows.",
)
parser.add_argument(
    "--single_num_cols",
    type=int,
    default=None,
    help="Number of terrain columns to keep when forcing a single terrain.",
)

parser.add_argument(
    "--command_bins",
    type=str,
    default="default",
    help="Comma-separated command bins, e.g. default,slow_forward,normal_forward.",
)
parser.add_argument("--episodes_per_condition", type=int, default=200, help="Episodes collected per condition.")
parser.add_argument(
    "--max_steps_per_condition",
    type=int,
    default=None,
    help="Safety cap for simulation steps per condition. Defaults to 4x expected horizon.",
)
parser.add_argument("--log_gate", action="store_true", default=True, help="Collect MoE gate episode metrics.")
parser.add_argument("--no_log_gate", dest="log_gate", action="store_false", help="Disable MoE gate metrics.")
parser.add_argument(
    "--gate_last_window_s",
    type=float,
    default=1.0,
    help="Window length in seconds for last-window gate metrics.",
)

cli_args.add_instinct_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from instinct_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg  # noqa: E402

import instinctlab.tasks  # noqa: F401, E402
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper  # noqa: E402

from eval_utils import (  # noqa: E402
    COMMAND_BINS,
    CsvRowWriter,
    EpisodeMetricsCollector,
    EvalRunMetadata,
    MoEGateMetricsCollector,
    apply_command_bin,
    build_summary,
    compute_tracking_errors,
    force_single_subterrain,
    force_terrain_level,
    get_eval_context,
    get_root_height,
    parse_csv_list,
    parse_int_list,
    write_json,
)


if args_cli.debug:
    import debugpy

    ip_address = ("0.0.0.0", 6789)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()


def _make_output_dir(args, agent_cfg):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    eval_name = args.eval_name or f"{agent_cfg.experiment_name}_{args.terrain_suite}"
    output_dir = os.path.abspath(os.path.join(args.output_dir, eval_name, timestamp))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _resolve_checkpoint(agent_cfg, args):
    if args.load_run is None:
        raise RuntimeError("--load_run is required for evaluation.")

    log_root_path = os.path.abspath(os.path.join("logs", "instinct_rl", agent_cfg.experiment_name))
    agent_cfg.load_run = args.load_run
    if os.path.isabs(agent_cfg.load_run):
        return get_checkpoint_path(
            os.path.dirname(agent_cfg.load_run),
            os.path.basename(agent_cfg.load_run),
            agent_cfg.load_checkpoint,
        )
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _terrain_levels_from_cfg(env_cfg, value: str):
    terrain_generator = getattr(env_cfg.scene.terrain, "terrain_generator", None)
    if value != "all":
        return parse_int_list(value)
    if terrain_generator is None:
        return [0]
    return list(range(int(terrain_generator.num_rows)))


def _prepare_env_cfg(args, command_bin: str):
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    env_cfg.seed = args.seed
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.terrain_suite == "single":
        if not args.terrain_name:
            raise ValueError("--terrain_name is required when --terrain_suite=single.")
        force_single_subterrain(env_cfg, args.terrain_name, args.single_num_cols)
    if command_bin not in COMMAND_BINS:
        available = ", ".join(COMMAND_BINS.keys())
        raise ValueError(f"Unknown command bin '{command_bin}'. Available command bins: {available}")
    apply_command_bin(env_cfg, command_bin)
    return env_cfg


def _create_env_and_runner(env_cfg, agent_cfg, agent_cfg_dict, checkpoint_path: str):
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = InstinctRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    runner.load(checkpoint_path)
    runner.eval_mode()
    return env, runner


def _normalize_for_gate(runner, obs, critic_obs):
    gate_obs = obs
    gate_critic_obs = critic_obs
    if "policy" in runner.normalizers:
        gate_obs = runner.normalizers["policy"](gate_obs)
    if critic_obs is not None and "critic" in runner.normalizers:
        gate_critic_obs = runner.normalizers["critic"](gate_critic_obs)
    return gate_obs, gate_critic_obs


def _as_int_max_episode_length(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.max().detach().cpu().item())
    return int(value)


def _evaluate_condition(
    args,
    env_cfg,
    agent_cfg,
    agent_cfg_dict,
    checkpoint_path: str,
    output_dir: str,
    terrain_level: int,
    command_bin: str,
    episode_writer: CsvRowWriter,
    gate_writer: CsvRowWriter,
    all_episode_rows: list[dict],
    all_gate_rows: list[dict],
):
    env, runner = _create_env_and_runner(env_cfg, agent_cfg, agent_cfg_dict, checkpoint_path)
    force_terrain_level(env, terrain_level)
    obs, extras = env.reset()
    critic_obs = extras["observations"].get("critic", obs)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    step_dt = float(env.unwrapped.step_dt)
    last_window_steps = max(1, int(args.gate_last_window_s / step_dt))
    metadata = EvalRunMetadata(
        model_name=args.model_name or agent_cfg.experiment_name,
        task_name=args.task,
        checkpoint=os.path.basename(checkpoint_path),
        seed=args.seed,
        terrain_suite=args.terrain_suite,
        command_bin=command_bin,
    )
    max_episode_length = _as_int_max_episode_length(env.max_episode_length)
    episode_collector = EpisodeMetricsCollector(
        env.num_envs,
        env.device,
        metadata,
        max_episode_length=max_episode_length,
    )
    gate_collector = MoEGateMetricsCollector(
        env.num_envs,
        env.device,
        metadata,
        episode_collector.episode_ids,
        last_window_steps=last_window_steps,
    )

    collected_episodes = 0
    max_steps = args.max_steps_per_condition or max(
        int(max_episode_length * max(4, args.episodes_per_condition // max(1, env.num_envs) + 2)),
        max_episode_length,
    )
    print(
        "[INFO] Eval condition:"
        f" terrain_suite={args.terrain_suite}, terrain_name={args.terrain_name},"
        f" level={terrain_level}, command_bin={command_bin}, target_episodes={args.episodes_per_condition}"
    )

    for step in range(max_steps):
        if collected_episodes >= args.episodes_per_condition:
            break
        if not simulation_app.is_running():
            break

        context = get_eval_context(
            env,
            {
                "terrain_level_condition": terrain_level,
                "condition_step": step,
            },
        )
        tracking_errors = compute_tracking_errors(env)
        root_height = get_root_height(env)

        with torch.inference_mode():
            actions = policy(obs)
            gate_weights = {}
            if args.log_gate and hasattr(runner.alg.actor_critic, "get_moe_gate_weights"):
                gate_obs, gate_critic_obs = _normalize_for_gate(runner, obs, critic_obs)
                gate_weights = runner.alg.actor_critic.get_moe_gate_weights(gate_obs, gate_critic_obs)
            obs, rewards, dones, infos = env.step(actions)

        next_critic_obs = infos["observations"].get("critic", obs)
        episode_rows, done_outcomes = episode_collector.update(
            env,
            rewards,
            dones,
            infos,
            context,
            tracking_errors=tracking_errors,
            root_height=root_height,
        )
        gate_rows = []
        if args.log_gate:
            gate_rows = gate_collector.update(gate_weights, dones, context, done_outcomes)

        if episode_rows:
            if len(episode_rows) + collected_episodes > args.episodes_per_condition:
                episode_rows = episode_rows[: args.episodes_per_condition - collected_episodes]
            kept_episode_keys = {(row["env_id"], row["episode_id"]) for row in episode_rows}
            gate_rows = [
                row for row in gate_rows if (row["env_id"], row["episode_id"]) in kept_episode_keys
            ]
            episode_writer.write_rows(episode_rows)
            all_episode_rows.extend(episode_rows)
            collected_episodes += len(episode_rows)

        if gate_rows:
            gate_writer.write_rows(gate_rows)
            all_gate_rows.extend(gate_rows)

        critic_obs = next_critic_obs

    env.close()
    if collected_episodes < args.episodes_per_condition:
        print(
            "[WARN] Condition ended before target episodes:"
            f" collected={collected_episodes}, target={args.episodes_per_condition}, max_steps={max_steps}"
        )
    write_json(
        os.path.join(output_dir, "last_condition.json"),
        {
            "terrain_level": terrain_level,
            "command_bin": command_bin,
            "collected_episodes": collected_episodes,
            "max_steps": max_steps,
        },
    )


def main():
    agent_cfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)
    agent_cfg.seed = args_cli.seed
    checkpoint_path = _resolve_checkpoint(agent_cfg, args_cli)
    output_dir = _make_output_dir(args_cli, agent_cfg)
    agent_cfg_dict = agent_cfg.to_dict()

    write_json(
        os.path.join(output_dir, "config.json"),
        {
            "task": args_cli.task,
            "load_run": args_cli.load_run,
            "checkpoint": args_cli.checkpoint,
            "resolved_checkpoint": checkpoint_path,
            "terrain_suite": args_cli.terrain_suite,
            "terrain_name": args_cli.terrain_name,
            "terrain_levels": args_cli.terrain_levels,
            "command_bins": args_cli.command_bins,
            "episodes_per_condition": args_cli.episodes_per_condition,
            "num_envs": args_cli.num_envs,
            "seed": args_cli.seed,
            "log_gate": args_cli.log_gate,
        },
    )

    command_bins = parse_csv_list(args_cli.command_bins, default=["default"])
    first_env_cfg = _prepare_env_cfg(args_cli, command_bins[0])
    terrain_levels = _terrain_levels_from_cfg(first_env_cfg, args_cli.terrain_levels)

    episode_writer = CsvRowWriter(os.path.join(output_dir, "episode_metrics.csv"))
    gate_writer = CsvRowWriter(os.path.join(output_dir, "gate_episode_metrics.csv"))
    all_episode_rows = []
    all_gate_rows = []

    try:
        for command_bin in command_bins:
            env_cfg = _prepare_env_cfg(args_cli, command_bin)
            for terrain_level in terrain_levels:
                _evaluate_condition(
                    args_cli,
                    env_cfg,
                    agent_cfg,
                    agent_cfg_dict,
                    checkpoint_path,
                    output_dir,
                    terrain_level,
                    command_bin,
                    episode_writer,
                    gate_writer,
                    all_episode_rows,
                    all_gate_rows,
                )
    finally:
        episode_writer.close()
        gate_writer.close()

    summary = build_summary(all_episode_rows, all_gate_rows)
    write_json(os.path.join(output_dir, "summary.json"), summary)
    print(f"[INFO] Evaluation complete. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
