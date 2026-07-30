# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

task_entry = "instinctlab.tasks.parkour.config.g1"


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_cfg:G1ParkourEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:G1ParkourPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_cfg:G1ParkourEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:G1ParkourPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-Stair-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_stair_cfg:G1ParkourStairEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:G1ParkourPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-Stair-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_stair_cfg:G1ParkourStairEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:G1ParkourPPORunnerCfg",
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-Gate-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_cfg:G1ParkourEnvCfg",
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_amp_cfg_gate:G1ParkourGateSeparatedPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-Gate-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_cfg:G1ParkourEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_amp_cfg_gate:G1ParkourGateSeparatedPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-Stair-Gate-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_stair_cfg:G1ParkourStairEnvCfg",
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_amp_cfg_gate:G1ParkourGateSeparatedPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-Stair-Gate-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_stair_cfg:G1ParkourStairEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_amp_cfg_gate:G1ParkourGateSeparatedPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-TerrainAux-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_terrain_aux_cfg:G1ParkourTerrainAuxEnvCfg",
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_amp_cfg_terrain_aux:G1ParkourTerrainAuxPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-TerrainAux-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_terrain_aux_cfg:G1ParkourTerrainAuxEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_amp_cfg_terrain_aux:G1ParkourTerrainAuxPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-TerrainAux-CrossAttn-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_terrain_aux_cfg:G1ParkourTerrainAuxEnvCfg",
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_amp_cfg_terrain_aux_cross_attn:"
            "G1ParkourTerrainAuxCrossAttentionPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-TerrainAux-CrossAttn-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_terrain_aux_cfg:G1ParkourTerrainAuxEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_amp_cfg_terrain_aux_cross_attn:"
            "G1ParkourTerrainAuxCrossAttentionPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-TerrainAux-Eval-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_eval_cfg:G1ParkourTerrainAuxEnvCfg_EVAL",
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_amp_cfg_terrain_aux:G1ParkourTerrainAuxPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-TerrainAux-CrossAttn-Eval-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_eval_cfg:G1ParkourTerrainAuxEnvCfg_EVAL",
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_amp_cfg_terrain_aux_cross_attn:"
            "G1ParkourTerrainAuxCrossAttentionPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-Stair-TerrainAux-Eval-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_eval_cfg:G1ParkourStairTerrainAuxEnvCfg_EVAL",
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_amp_cfg_terrain_aux:G1ParkourTerrainAuxPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-Stair-TerrainAux-CrossAttn-Eval-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.g1_parkour_target_amp_eval_cfg:G1ParkourStairTerrainAuxEnvCfg_EVAL",
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_amp_cfg_terrain_aux_cross_attn:"
            "G1ParkourStairTerrainAuxCrossAttentionPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-Stair-TerrainAux-CrossAttn-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{task_entry}.g1_parkour_target_amp_terrain_aux_cfg:G1ParkourStairTerrainAuxEnvCfg"
        ),
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_amp_cfg_terrain_aux_cross_attn:"
            "G1ParkourStairTerrainAuxCrossAttentionPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-Stair-TerrainAux-CrossAttn-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{task_entry}.g1_parkour_target_amp_terrain_aux_cfg:G1ParkourStairTerrainAuxEnvCfg_PLAY"
        ),
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_amp_cfg_terrain_aux_cross_attn:"
            "G1ParkourStairTerrainAuxCrossAttentionPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-Stair-TerrainAux-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{task_entry}.g1_parkour_target_amp_terrain_aux_cfg:G1ParkourStairTerrainAuxEnvCfg"
        ),
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_amp_cfg_terrain_aux:G1ParkourTerrainAuxPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-Stair-TerrainAux-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{task_entry}.g1_parkour_target_amp_terrain_aux_cfg:G1ParkourStairTerrainAuxEnvCfg_PLAY"
        ),
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_amp_cfg_terrain_aux:G1ParkourTerrainAuxPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-Stair-Carry-TerrainAux-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{task_entry}.g1_parkour_target_amp_carry_terrain_aux_cfg:"
            "G1ParkourStairCarryTerrainAuxEnvCfg"
        ),
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_amp_cfg_terrain_aux:G1ParkourTerrainAuxPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Instinct-Parkour-Target-Amp-G1-Stair-Carry-TerrainAux-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{task_entry}.g1_parkour_target_amp_carry_terrain_aux_cfg:"
            "G1ParkourStairCarryTerrainAuxEnvCfg_PLAY"
        ),
        "instinct_rl_cfg_entry_point": (
            f"{agents.__name__}.instinct_rl_amp_cfg_terrain_aux:G1ParkourTerrainAuxPPORunnerCfg"
        ),
    },
)
