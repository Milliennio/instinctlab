from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch


EPS = 1.0e-8


COMMAND_BINS = {
    "default": None,
    "stand": {
        "rel_standing_envs": 1.0,
        "lin_vel_x": (0.0, 0.0),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (0.0, 0.0),
    },
    "slow_forward": {
        "rel_standing_envs": 0.0,
        "lin_vel_x": (0.3, 0.3),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (0.0, 0.0),
    },
    "normal_forward": {
        "rel_standing_envs": 0.0,
        "lin_vel_x": (0.6, 0.6),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (0.0, 0.0),
    },
    "fast_forward": {
        "rel_standing_envs": 0.0,
        "lin_vel_x": (0.9, 0.9),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (0.0, 0.0),
    },
    "turn_left": {
        "rel_standing_envs": 0.0,
        "lin_vel_x": (0.4, 0.4),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (0.5, 0.5),
    },
    "turn_right": {
        "rel_standing_envs": 0.0,
        "lin_vel_x": (0.4, 0.4),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-0.5, -0.5),
    },
}


def parse_csv_list(value: str | None, default: list[str] | None = None) -> list[str]:
    if value is None or value == "":
        return list(default or [])
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str | None, default: list[int] | None = None) -> list[int]:
    if value is None or value == "" or value == "all":
        return list(default or [])
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def tensor_to_list(tensor: torch.Tensor) -> list:
    return tensor.detach().cpu().tolist()


def scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


class CsvRowWriter:
    def __init__(self, path: str):
        ensure_dir(os.path.dirname(path))
        self.path = path
        self.file = open(path, "w", newline="", buffering=1)
        self.writer: csv.DictWriter | None = None
        self.fieldnames: list[str] | None = None

    def write_rows(self, rows: list[dict[str, Any]]):
        for row in rows:
            self.write_row(row)

    def write_row(self, row: dict[str, Any]):
        if self.writer is None:
            self.fieldnames = list(row.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
            self.writer.writeheader()
        else:
            missing = [key for key in row.keys() if key not in self.fieldnames]
            if missing:
                raise ValueError(f"CSV row for {self.path} has new fields after header creation: {missing}")
        self.writer.writerow(row)

    def close(self):
        self.file.close()


def write_json(path: str, data: dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def apply_command_bin(env_cfg, command_bin: str):
    cfg = COMMAND_BINS.get(command_bin)
    if cfg is None:
        return

    base_velocity = env_cfg.commands.base_velocity
    base_velocity.rel_standing_envs = cfg["rel_standing_envs"]
    ranges = {
        "lin_vel_x": cfg["lin_vel_x"],
        "lin_vel_y": cfg["lin_vel_y"],
        "ang_vel_z": cfg["ang_vel_z"],
    }
    base_velocity.ranges.lin_vel_x = ranges["lin_vel_x"]
    base_velocity.ranges.lin_vel_y = ranges["lin_vel_y"]
    base_velocity.ranges.ang_vel_z = ranges["ang_vel_z"]
    if base_velocity.velocity_ranges is not None:
        base_velocity.velocity_ranges = {
            name: {
                "lin_vel_x": ranges["lin_vel_x"],
                "lin_vel_y": ranges["lin_vel_y"],
                "ang_vel_z": ranges["ang_vel_z"],
            }
            for name in base_velocity.velocity_ranges.keys()
        }
    base_velocity.random_velocity_terrain = None


def force_single_subterrain(env_cfg, terrain_name: str, num_cols: int | None = None):
    terrain_generator = getattr(env_cfg.scene.terrain, "terrain_generator", None)
    if terrain_generator is None:
        raise ValueError("This task does not use a terrain generator, so terrain_name is not applicable.")
    if terrain_name not in terrain_generator.sub_terrains:
        available = ", ".join(terrain_generator.sub_terrains.keys())
        raise ValueError(f"Unknown terrain '{terrain_name}'. Available sub-terrains: {available}")

    sub_terrain_cfg = terrain_generator.sub_terrains[terrain_name]
    if hasattr(sub_terrain_cfg, "proportion"):
        sub_terrain_cfg.proportion = 1.0
    terrain_generator.sub_terrains = {terrain_name: sub_terrain_cfg}
    terrain_generator.num_cols = int(num_cols or max(1, terrain_generator.num_cols))
    terrain_generator.curriculum = False

    base_velocity = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
    if base_velocity is not None:
        if getattr(base_velocity, "velocity_ranges", None) is not None:
            if terrain_name in base_velocity.velocity_ranges:
                base_velocity.velocity_ranges = {terrain_name: base_velocity.velocity_ranges[terrain_name]}
            else:
                base_velocity.velocity_ranges = None
        if getattr(base_velocity, "random_velocity_terrain", None) is not None:
            base_velocity.random_velocity_terrain = [
                name for name in base_velocity.random_velocity_terrain if name == terrain_name
            ] or None


def force_terrain_level(env, terrain_level: int):
    terrain = env.unwrapped.scene.terrain
    max_level = terrain.terrain_origins.shape[0] - 1
    level = int(max(0, min(int(terrain_level), max_level)))
    terrain.terrain_levels[:] = level
    terrain.env_origins[:] = terrain.terrain_origins[level, terrain.terrain_types]


def get_terrain_names(env) -> list[str]:
    terrain = env.unwrapped.scene.terrain
    terrain_generator = terrain.cfg.terrain_generator
    if terrain_generator is None:
        return []
    return list(terrain_generator.sub_terrains.keys())


def get_eval_context(env, metadata: dict[str, Any]) -> dict[str, Any]:
    terrain = env.unwrapped.scene.terrain
    terrain_names = get_terrain_names(env)
    terrain_types = terrain.terrain_types.detach().long().cpu()
    terrain_levels = terrain.terrain_levels.detach().long().cpu()
    names = [
        terrain_names[int(type_id)] if 0 <= int(type_id) < len(terrain_names) else f"type_{int(type_id)}"
        for type_id in terrain_types
    ]
    return {
        **metadata,
        "terrain_type_ids": terrain_types,
        "terrain_levels": terrain_levels,
        "terrain_names": names,
    }


def get_command_tensor(env) -> torch.Tensor | None:
    command_manager = getattr(env.unwrapped, "command_manager", None)
    if command_manager is None:
        return None
    try:
        command_term = command_manager.get_term("base_velocity")
    except Exception:
        return None
    command = getattr(command_term, "command", None)
    if command is None:
        return None
    return command.detach()


def get_robot_state(env) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    try:
        robot = env.unwrapped.scene["robot"]
    except Exception:
        return None, None, None
    root_lin_vel_b = getattr(robot.data, "root_lin_vel_b", None)
    root_ang_vel_b = getattr(robot.data, "root_ang_vel_b", None)
    root_pos_w = getattr(robot.data, "root_pos_w", None)
    return root_lin_vel_b, root_ang_vel_b, root_pos_w


def compute_tracking_errors(env) -> tuple[torch.Tensor, torch.Tensor]:
    command = get_command_tensor(env)
    root_lin_vel_b, root_ang_vel_b, _ = get_robot_state(env)
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    if command is None or root_lin_vel_b is None or root_ang_vel_b is None:
        return torch.zeros(num_envs, device=device), torch.zeros(num_envs, device=device)
    xy_error = torch.norm(command[:, :2] - root_lin_vel_b[:, :2], dim=-1)
    yaw_error = torch.abs(command[:, 2] - root_ang_vel_b[:, 2])
    return xy_error, yaw_error


def get_root_height(env) -> torch.Tensor:
    _, _, root_pos_w = get_robot_state(env)
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    if root_pos_w is None:
        return torch.zeros(num_envs, device=device)
    return root_pos_w[:, 2]


def get_time_outs(extras: dict, dones: torch.Tensor) -> torch.Tensor:
    time_outs = extras.get("time_outs", None)
    if time_outs is None:
        return torch.zeros_like(dones, dtype=torch.bool)
    return time_outs.detach().to(device=dones.device, dtype=torch.bool).reshape_as(dones.bool()).clone()


@dataclass
class EvalRunMetadata:
    model_name: str
    task_name: str
    checkpoint: str
    seed: int
    terrain_suite: str
    command_bin: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "task_name": self.task_name,
            "checkpoint": self.checkpoint,
            "seed": self.seed,
            "terrain_suite": self.terrain_suite,
            "command_bin": self.command_bin,
        }


class EpisodeMetricsCollector:
    def __init__(self, num_envs: int, device: torch.device, metadata: EvalRunMetadata, max_episode_length: int):
        self.num_envs = num_envs
        self.device = device
        self.metadata = metadata
        self.max_episode_length = int(max_episode_length)
        self.episode_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.returns = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.lengths = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.xy_error_sum = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.yaw_error_sum = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.root_height_min = torch.full((num_envs,), float("inf"), dtype=torch.float, device=device)

    def update(
        self,
        env,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
        context: dict[str, Any],
        tracking_errors: tuple[torch.Tensor, torch.Tensor] | None = None,
        root_height: torch.Tensor | None = None,
    ):
        rewards_1d = rewards.sum(dim=-1) if rewards.ndim > 1 else rewards
        rewards_1d = rewards_1d.detach().to(device=self.device, dtype=torch.float).reshape(self.num_envs).clone()
        dones_bool = dones.detach().to(device=self.device, dtype=torch.bool).reshape(-1).clone()
        time_outs = get_time_outs(extras, dones_bool)
        xy_error, yaw_error = tracking_errors if tracking_errors is not None else compute_tracking_errors(env)
        root_height = root_height if root_height is not None else get_root_height(env)
        xy_error = xy_error.detach().to(device=self.device, dtype=torch.float).reshape(self.num_envs).clone()
        yaw_error = yaw_error.detach().to(device=self.device, dtype=torch.float).reshape(self.num_envs).clone()
        root_height = root_height.detach().to(device=self.device, dtype=torch.float).reshape(self.num_envs).clone()

        self.returns += rewards_1d
        self.lengths += 1
        self.xy_error_sum += xy_error
        self.yaw_error_sum += yaw_error
        self.root_height_min = torch.minimum(self.root_height_min, root_height)

        rows = []
        done_outcomes = {}
        done_ids = dones_bool.nonzero(as_tuple=False).flatten()
        for env_id_tensor in done_ids:
            env_id = int(env_id_tensor.item())
            length = max(1, int(self.lengths[env_id].item()))
            timed_out = bool(time_outs[env_id].item())
            success = timed_out or length >= self.max_episode_length
            failure_reason = "timeout" if success else "failure"
            episode_id = int(self.episode_ids[env_id].item())
            terrain_name = context["terrain_names"][env_id]
            terrain_level = int(context["terrain_levels"][env_id].item())
            terrain_type_id = int(context["terrain_type_ids"][env_id].item())
            done_outcomes[env_id] = {
                "episode_id": episode_id,
                "success": int(success),
                "failure_reason": failure_reason,
            }
            row = {
                **self.metadata.to_dict(),
                "env_id": env_id,
                "episode_id": episode_id,
                "terrain_name": terrain_name,
                "terrain_type_id": terrain_type_id,
                "terrain_level": terrain_level,
                "success": int(success),
                "failure_reason": failure_reason,
                "episode_length": length,
                "episode_return": float(self.returns[env_id].detach().cpu().item()),
                "tracking_error_xy": float((self.xy_error_sum[env_id] / length).detach().cpu().item()),
                "tracking_error_yaw": float((self.yaw_error_sum[env_id] / length).detach().cpu().item()),
                "root_height_min": float(self.root_height_min[env_id].detach().cpu().item()),
            }
            rows.append(row)

        if done_ids.numel() > 0:
            self.episode_ids[done_ids] += 1
            self.returns[done_ids] = 0.0
            self.lengths[done_ids] = 0
            self.xy_error_sum[done_ids] = 0.0
            self.yaw_error_sum[done_ids] = 0.0
            self.root_height_min[done_ids] = float("inf")

        return rows, done_outcomes


class MoEGateMetricsCollector:
    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        metadata: EvalRunMetadata,
        episode_ids: torch.Tensor,
        last_window_steps: int,
    ):
        self.num_envs = num_envs
        self.device = device
        self.metadata = metadata
        self.episode_ids = episode_ids
        self.last_window_steps = max(1, int(last_window_steps))
        self.states: dict[str, dict[str, torch.Tensor | list[dict[str, torch.Tensor]]]] = {}

    def _init_gate_state(self, gate_name: str, num_experts: int):
        self.states[gate_name] = {
            "num_experts": torch.tensor(num_experts, device=self.device),
            "count": torch.zeros(self.num_envs, device=self.device),
            "entropy_sum": torch.zeros(self.num_envs, device=self.device),
            "entropy_sq_sum": torch.zeros(self.num_envs, device=self.device),
            "norm_entropy_sum": torch.zeros(self.num_envs, device=self.device),
            "top1_weight_sum": torch.zeros(self.num_envs, device=self.device),
            "eff_experts_sum": torch.zeros(self.num_envs, device=self.device),
            "gate_tv_sum": torch.zeros(self.num_envs, device=self.device),
            "switch_count": torch.zeros(self.num_envs, device=self.device),
            "expert_top1_counts": torch.zeros(self.num_envs, num_experts, device=self.device),
            "expert_weight_sums": torch.zeros(self.num_envs, num_experts, device=self.device),
            "last_weights": torch.zeros(self.num_envs, num_experts, device=self.device),
            "last_top1": torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device),
            "has_last": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "recent": [],
        }

    def update(
        self,
        gate_weights: dict[str, torch.Tensor],
        dones: torch.Tensor,
        context: dict[str, Any],
        episode_outcomes: dict[int, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        rows = []
        dones_bool = dones.detach().to(device=self.device, dtype=torch.bool).reshape(-1).clone()
        done_ids = dones_bool.nonzero(as_tuple=False).flatten()

        for gate_name, weights in gate_weights.items():
            if weights is None or weights.numel() == 0:
                continue
            weights = weights.detach().to(device=self.device, dtype=torch.float).reshape(self.num_envs, -1).clone()
            num_experts = weights.shape[-1]
            if gate_name not in self.states:
                self._init_gate_state(gate_name, num_experts)
            state = self.states[gate_name]

            clipped = weights.clamp_min(EPS)
            entropy = -(clipped * clipped.log()).sum(dim=-1)
            norm_entropy = entropy / math.log(float(num_experts))
            eff_experts = entropy.exp()
            top1_weight, top1 = weights.max(dim=-1)

            has_last = state["has_last"]
            last_weights = state["last_weights"]
            last_top1 = state["last_top1"]
            gate_tv = 0.5 * torch.abs(weights - last_weights).sum(dim=-1)
            gate_tv = torch.where(has_last, gate_tv, torch.zeros_like(gate_tv))
            switch = (top1 != last_top1) & has_last

            state["count"] += 1.0
            state["entropy_sum"] += entropy
            state["entropy_sq_sum"] += entropy.square()
            state["norm_entropy_sum"] += norm_entropy
            state["top1_weight_sum"] += top1_weight
            state["eff_experts_sum"] += eff_experts
            state["gate_tv_sum"] += gate_tv
            state["switch_count"] += switch.float()
            state["expert_weight_sums"] += weights
            state["expert_top1_counts"].scatter_add_(1, top1.unsqueeze(-1), torch.ones(self.num_envs, 1, device=self.device))

            state["last_weights"] = weights
            state["last_top1"] = top1
            state["has_last"] = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            recent = state["recent"]
            recent.append(
                {
                    "entropy": entropy.detach().clone(),
                    "top1_weight": top1_weight.detach().clone(),
                    "gate_tv": gate_tv.detach().clone(),
                    "switch": switch.float().detach().clone(),
                }
            )
            if len(recent) > self.last_window_steps:
                del recent[0]

            for env_id_tensor in done_ids:
                env_id = int(env_id_tensor.item())
                rows.append(
                    self._build_done_row(
                        gate_name,
                        state,
                        env_id,
                        context,
                        episode_outcomes or {},
                    )
                )

            if done_ids.numel() > 0:
                self._reset_gate_state_entries(state, done_ids)

        return rows

    def _recent_mean(self, state: dict[str, Any], key: str, env_id: int) -> float:
        recent = state["recent"]
        if not recent:
            return 0.0
        values = torch.stack([item[key][env_id] for item in recent])
        return float(values.mean().detach().cpu().item())

    def _build_done_row(
        self,
        gate_name: str,
        state: dict[str, Any],
        env_id: int,
        context: dict[str, Any],
        episode_outcomes: dict[int, dict[str, Any]],
    ) -> dict[str, Any]:
        count = max(1.0, float(state["count"][env_id].detach().cpu().item()))
        num_experts = int(state["num_experts"].item())
        entropy_mean = state["entropy_sum"][env_id] / count
        entropy_sq_mean = state["entropy_sq_sum"][env_id] / count
        entropy_std = torch.clamp(entropy_sq_mean - entropy_mean.square(), min=0.0).sqrt()
        expert_top1_counts = state["expert_top1_counts"][env_id]
        expert_weight_sums = state["expert_weight_sums"][env_id]
        dominant_fraction, dominant_expert = (expert_top1_counts / count).max(dim=0)
        outcome = episode_outcomes.get(env_id, {})

        row = {
            **self.metadata.to_dict(),
            "env_id": env_id,
            "episode_id": int(outcome.get("episode_id", self.episode_ids[env_id].item())),
            "gate_name": gate_name,
            "terrain_name": context["terrain_names"][env_id],
            "terrain_type_id": int(context["terrain_type_ids"][env_id].item()),
            "terrain_level": int(context["terrain_levels"][env_id].item()),
            "success": int(outcome.get("success", 0)),
            "failure_reason": outcome.get("failure_reason", "unknown"),
            "entropy_mean": float(entropy_mean.detach().cpu().item()),
            "entropy_std": float(entropy_std.detach().cpu().item()),
            "normalized_entropy_mean": float((state["norm_entropy_sum"][env_id] / count).detach().cpu().item()),
            "top1_weight_mean": float((state["top1_weight_sum"][env_id] / count).detach().cpu().item()),
            "effective_num_experts_mean": float((state["eff_experts_sum"][env_id] / count).detach().cpu().item()),
            "gate_tv_mean": float((state["gate_tv_sum"][env_id] / count).detach().cpu().item()),
            "top1_switch_rate": float((state["switch_count"][env_id] / count).detach().cpu().item()),
            "dominant_expert": int(dominant_expert.detach().cpu().item()),
            "dominant_expert_fraction": float(dominant_fraction.detach().cpu().item()),
            "entropy_last_1s": self._recent_mean(state, "entropy", env_id),
            "top1_weight_last_1s": self._recent_mean(state, "top1_weight", env_id),
            "gate_tv_last_1s": self._recent_mean(state, "gate_tv", env_id),
            "top1_switch_rate_last_1s": self._recent_mean(state, "switch", env_id),
        }
        for expert_id in range(num_experts):
            row[f"expert_{expert_id}_top1_fraction"] = float(
                (expert_top1_counts[expert_id] / count).detach().cpu().item()
            )
            row[f"expert_{expert_id}_weight_mean"] = float(
                (expert_weight_sums[expert_id] / count).detach().cpu().item()
            )
        return row

    def _reset_gate_state_entries(self, state: dict[str, Any], env_ids: torch.Tensor):
        state["count"][env_ids] = 0.0
        state["entropy_sum"][env_ids] = 0.0
        state["entropy_sq_sum"][env_ids] = 0.0
        state["norm_entropy_sum"][env_ids] = 0.0
        state["top1_weight_sum"][env_ids] = 0.0
        state["eff_experts_sum"][env_ids] = 0.0
        state["gate_tv_sum"][env_ids] = 0.0
        state["switch_count"][env_ids] = 0.0
        state["expert_top1_counts"][env_ids] = 0.0
        state["expert_weight_sums"][env_ids] = 0.0
        state["last_weights"][env_ids] = 0.0
        state["last_top1"][env_ids] = -1
        state["has_last"][env_ids] = False


def success_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float]]:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    summary = {}
    for name, group_rows in groups.items():
        success_values = [float(row["success"]) for row in group_rows]
        returns = [float(row["episode_return"]) for row in group_rows]
        lengths = [float(row["episode_length"]) for row in group_rows]
        tracking_xy = [float(row["tracking_error_xy"]) for row in group_rows]
        summary[name] = {
            "num_episodes": len(group_rows),
            "success_rate": sum(success_values) / max(1, len(success_values)),
            "episode_return_mean": sum(returns) / max(1, len(returns)),
            "episode_length_mean": sum(lengths) / max(1, len(lengths)),
            "tracking_error_xy_mean": sum(tracking_xy) / max(1, len(tracking_xy)),
        }
    return summary


def gate_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["gate_name"])].append(row)
    summary = {}
    for gate_name, group_rows in groups.items():
        summary[gate_name] = {
            "num_episodes": len(group_rows),
            "entropy_mean": sum(float(row["entropy_mean"]) for row in group_rows) / max(1, len(group_rows)),
            "top1_weight_mean": sum(float(row["top1_weight_mean"]) for row in group_rows) / max(1, len(group_rows)),
            "gate_tv_mean": sum(float(row["gate_tv_mean"]) for row in group_rows) / max(1, len(group_rows)),
            "top1_switch_rate_mean": sum(float(row["top1_switch_rate"]) for row in group_rows)
            / max(1, len(group_rows)),
        }
    return summary


def build_summary(episode_rows: list[dict[str, Any]], gate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    success_values = [float(row["success"]) for row in episode_rows]
    return {
        "overall": {
            "num_episodes": len(episode_rows),
            "success_rate": sum(success_values) / max(1, len(success_values)),
        },
        "by_terrain": success_by(episode_rows, "terrain_name"),
        "by_level": success_by(episode_rows, "terrain_level"),
        "by_command": success_by(episode_rows, "command_bin"),
        "gate": gate_summary(gate_rows),
    }
