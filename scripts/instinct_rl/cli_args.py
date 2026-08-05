from __future__ import annotations

import argparse
import os
import yaml
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg


def _get_gate_slice_component_names():
    """Lazily import gate-slice constants after SimulationApp has been created."""
    from instinctlab.tasks.parkour.config.g1.agents.gate_slice import (
        ACTOR_MOE_GATE_COMPONENT_NAMES,
        CRITIC_MOE_GATE_COMPONENT_NAMES,
    )

    return list(ACTOR_MOE_GATE_COMPONENT_NAMES), list(CRITIC_MOE_GATE_COMPONENT_NAMES)


def add_instinct_rl_args(parser: argparse.ArgumentParser):
    """Add INSTINCT-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("instinct_rl", description="Arguments for Instinct-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", default=None, action="store_true", help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    arg_group.add_argument(
        "--gate_slice",
        dest="gate_slice",
        action="store_true",
        default=None,
        help="Enable sliced MoE gate inputs for parkour-style encoder MoE policies.",
    )
    arg_group.add_argument(
        "--no_gate_slice",
        dest="gate_slice",
        action="store_false",
        help="Disable sliced MoE gate inputs and use full encoded observations for MoE gates.",
    )
    arg_group.add_argument(
        "--gate_mode",
        type=str,
        choices=("full", "slice", "factorized"),
        default=None,
        help="Select full, sliced, or terrain-motion factorized MoE routing when supported by the policy.",
    )
    arg_group.add_argument(
        "--enable_gate_phase_summary",
        dest="gate_phase_summary",
        action="store_true",
        default=None,
        help="Enable the compact current-joint phase summary in a factorized MoE gate.",
    )
    arg_group.add_argument(
        "--disable_gate_phase_summary",
        dest="gate_phase_summary",
        action="store_false",
        help="Disable the compact phase summary in a factorized MoE gate.",
    )
    arg_group.add_argument(
        "--enable_gate_contrastive",
        dest="gate_contrastive",
        action="store_true",
        default=None,
        help="Enable terrain gate memory/height contrastive supervision.",
    )
    arg_group.add_argument(
        "--disable_gate_contrastive",
        dest="gate_contrastive",
        action="store_false",
        help="Disable terrain gate contrastive supervision.",
    )
    arg_group.add_argument("--gate_beta", type=float, default=None, help="Motion-logit scale in the factorized gate.")
    arg_group.add_argument(
        "--gate_temperature", type=float, default=None, help="Softmax temperature in the factorized gate."
    )
    arg_group.add_argument(
        "--gate_contrastive_weight",
        type=float,
        default=None,
        help="PPO loss coefficient for factorized gate contrastive supervision.",
    )
    arg_group.add_argument(
        "--gate_balance_weight",
        type=float,
        default=None,
        help="PPO loss coefficient for global expert load balancing.",
    )
    # # -- logger arguments
    # arg_group.add_argument(
    #     "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    # )
    # arg_group.add_argument(
    #     "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    # )


def parse_instinct_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> InstinctRlOnPolicyRunnerCfg:
    """Parse configuration for Instinct-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for Instinct-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    instinctrl_cfg: InstinctRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "instinct_rl_cfg_entry_point")
    instinctrl_cfg = update_instinct_rl_cfg(instinctrl_cfg, args_cli)
    return instinctrl_cfg


def update_instinct_rl_cfg(agent_cfg: InstinctRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for Instinct-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for Instinct-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for Instinct-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    gate_mode = getattr(args_cli, "gate_mode", None)
    gate_slice = getattr(args_cli, "gate_slice", None)
    if hasattr(agent_cfg, "policy") and gate_mode is not None:
        policy_cfg = agent_cfg.policy
        if not hasattr(policy_cfg, "moe_gate_mode"):
            raise ValueError(
                f"Policy {type(policy_cfg).__name__} does not support --gate_mode. "
                "Use a factorized-gate task configuration."
            )
        policy_cfg.moe_gate_mode = gate_mode
        if gate_mode == "slice":
            actor_gate_component_names, critic_gate_component_names = _get_gate_slice_component_names()
            policy_cfg.moe_actor_gate_component_names = actor_gate_component_names
            policy_cfg.moe_critic_gate_component_names = critic_gate_component_names
        elif gate_mode == "full":
            policy_cfg.moe_actor_gate_component_names = None
            policy_cfg.moe_critic_gate_component_names = None
    elif gate_slice is not None and hasattr(agent_cfg, "policy"):
        policy_cfg = agent_cfg.policy
        actor_gate_component_names = critic_gate_component_names = None
        if gate_slice:
            actor_gate_component_names, critic_gate_component_names = _get_gate_slice_component_names()
        if hasattr(policy_cfg, "moe_actor_gate_component_names"):
            policy_cfg.moe_actor_gate_component_names = actor_gate_component_names
        if hasattr(policy_cfg, "moe_critic_gate_component_names"):
            policy_cfg.moe_critic_gate_component_names = critic_gate_component_names
        if hasattr(policy_cfg, "moe_gate_mode"):
            policy_cfg.moe_gate_mode = "slice" if gate_slice else "full"

    if hasattr(agent_cfg, "policy"):
        policy_cfg = agent_cfg.policy
        policy_overrides = {
            "moe_gate_enable_phase_summary": getattr(args_cli, "gate_phase_summary", None),
            "moe_gate_enable_contrastive": getattr(args_cli, "gate_contrastive", None),
            "moe_gate_beta": getattr(args_cli, "gate_beta", None),
            "moe_gate_temperature": getattr(args_cli, "gate_temperature", None),
        }
        for field_name, value in policy_overrides.items():
            if value is not None:
                if not hasattr(policy_cfg, field_name):
                    raise ValueError(f"Policy {type(policy_cfg).__name__} does not support {field_name}.")
                setattr(policy_cfg, field_name, value)
        max_iterations = getattr(args_cli, "max_iterations", None)
        if max_iterations is not None and hasattr(policy_cfg, "moe_gate_total_iterations"):
            policy_cfg.moe_gate_total_iterations = max_iterations

    if hasattr(agent_cfg, "algorithm"):
        algorithm_cfg = agent_cfg.algorithm
        algorithm_overrides = {
            "moe_gate_contrastive_loss_coef": getattr(args_cli, "gate_contrastive_weight", None),
            "moe_gate_balance_loss_coef": getattr(args_cli, "gate_balance_weight", None),
        }
        for field_name, value in algorithm_overrides.items():
            if value is not None:
                if not hasattr(algorithm_cfg, field_name):
                    raise ValueError(f"Algorithm {type(algorithm_cfg).__name__} does not support {field_name}.")
                setattr(algorithm_cfg, field_name, value)
    # if args_cli.logger is not None:
    #     agent_cfg.logger = args_cli.logger
    # # set the project name for wandb and neptune
    # if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
    #     agent_cfg.wandb_project = args_cli.log_project_name
    #     agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg
