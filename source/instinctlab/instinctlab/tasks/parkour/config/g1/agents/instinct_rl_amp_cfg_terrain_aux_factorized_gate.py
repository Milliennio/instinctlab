from isaaclab.utils import configclass

from .instinct_rl_amp_cfg_terrain_aux import TerrainAuxAmpAlgoCfg
from .instinct_rl_amp_cfg_terrain_aux_cross_attn import (
    G1ParkourTerrainAuxCrossAttentionPPORunnerCfg,
    TerrainAuxCrossAttentionMoEPolicyCfg,
)


@configclass
class TerrainAuxCrossAttentionFactorizedGatePolicyCfg(TerrainAuxCrossAttentionMoEPolicyCfg):
    class_name = (
        "instinct_rl.modules.factorized_terrain_aux_actor_critic:"
        "FactorizedTerrainAuxEncoderMoEActorCritic"
    )

    moe_gate_mode = "factorized"
    moe_gate_beta = 0.5
    moe_gate_temperature = 1.0

    moe_gate_terrain_dim = 128
    moe_gate_terrain_hidden_dims = [64, 32]
    moe_gate_motion_hidden_dims = [64, 32]

    moe_gate_phase_component_names = ["joint_pos", "joint_vel"]
    moe_gate_phase_component_current_dim = 29
    moe_gate_phase_hidden_dim = 32
    moe_gate_phase_dim = 16
    moe_gate_enable_phase_summary = True

    moe_actor_motion_component_names = [
        "projected_gravity",
        "velocity_commands",
        "base_ang_vel",
    ]
    moe_critic_motion_component_names = [
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
    ]

    moe_gate_enable_contrastive = False
    moe_gate_projection_dim = 32
    moe_gate_height_view_weight = 0.25
    moe_gate_contrastive_temperature = 0.1
    moe_gate_contrastive_max_samples = 1024
    moe_gate_warmup_ratio = 0.1
    moe_gate_total_iterations = 30000


@configclass
class FactorizedGateTerrainAuxAmpAlgoCfg(TerrainAuxAmpAlgoCfg):
    moe_gate_contrastive_loss_coef = 0.01
    moe_gate_balance_loss_coef = 0.001


@configclass
class G1ParkourTerrainAuxFactorizedGatePPORunnerCfg(
    G1ParkourTerrainAuxCrossAttentionPPORunnerCfg
):
    policy = TerrainAuxCrossAttentionFactorizedGatePolicyCfg()
    algorithm = FactorizedGateTerrainAuxAmpAlgoCfg()
    experiment_name = "g1_parkour_terrain_aux_cross_attn_factorized_gate"


@configclass
class G1ParkourStairTerrainAuxFactorizedGatePPORunnerCfg(
    G1ParkourTerrainAuxFactorizedGatePPORunnerCfg
):
    experiment_name = "g1_parkour_stair_terrain_aux_cross_attn_factorized_gate"
