from isaaclab.utils import configclass

from .instinct_rl_amp_cfg_terrain_aux import TerrainAuxAmpAlgoCfg
from .instinct_rl_amp_cfg_terrain_aux_cross_attn import (
    G1ParkourTerrainAuxCrossAttentionPPORunnerCfg,
    TerrainAuxCrossAttentionMoEPolicyCfg,
)


@configclass
class TerrainAuxCrossAttentionTerrainCommandGatePolicyCfg(
    TerrainAuxCrossAttentionMoEPolicyCfg
):
    class_name = (
        "instinct_rl.modules.terrain_command_terrain_aux_actor_critic:"
        "TerrainCommandTerrainAuxEncoderMoEActorCritic"
    )

    actor_gate_mode = "factorized"
    critic_gate_mode = "baseline"

    factorized_gate_num_experts = 4
    factorized_gate_history_length = 8
    factorized_gate_command_frame_dim = 3
    factorized_gate_command_component_names = [
        "velocity_commands",
        "projected_gravity",
        "base_ang_vel",
    ]

    factorized_gate_terrain_dim = 128
    factorized_gate_pool_hidden_dim = 32
    factorized_gate_terrain_hidden_dims = [64, 32]
    factorized_gate_command_hidden_dims = [64, 32]
    factorized_gate_beta = 0.5
    factorized_gate_temperature = 1.0

    factorized_gate_geo_aux_enabled = False
    factorized_gate_geo_aux_weight = 0.01
    factorized_gate_balance_weight = 0.001
    factorized_gate_geo_hidden_dim = 64
    factorized_gate_geo_aux_warmup_ratio = 0.1
    factorized_gate_geo_aux_ramp_ratio = 0.1
    factorized_gate_edge_weight = 0.0
    factorized_gate_total_iterations = 30000

    factorized_gate_grid_resolution = 0.12
    factorized_gate_grid_size = [1.2, 0.96]
    factorized_gate_grid_offset_xy = [0.45, 0.0]
    factorized_gate_blind_x_range = [-0.15, 0.33]
    factorized_gate_blind_y_range = [-0.48, 0.48]
    factorized_gate_blind_mask_version = "camera_crop_v1_x_le_0p33"


@configclass
class TerrainAuxCrossAttentionTerrainCommandGateGeoAuxPolicyCfg(
    TerrainAuxCrossAttentionTerrainCommandGatePolicyCfg
):
    factorized_gate_geo_aux_enabled = True


@configclass
class TerrainCommandGateTerrainAuxAmpAlgoCfg(TerrainAuxAmpAlgoCfg):
    terrain_command_gate_geometry_loss_coef = 0.01
    terrain_command_gate_balance_loss_coef = 0.001


@configclass
class G1ParkourTerrainAuxTerrainCommandGatePPORunnerCfg(
    G1ParkourTerrainAuxCrossAttentionPPORunnerCfg
):
    policy = TerrainAuxCrossAttentionTerrainCommandGatePolicyCfg()
    algorithm = TerrainCommandGateTerrainAuxAmpAlgoCfg()
    experiment_name = "g1_parkour_terrain_aux_cross_attn_terrain_command_gate"


@configclass
class G1ParkourStairTerrainAuxTerrainCommandGatePPORunnerCfg(
    G1ParkourTerrainAuxTerrainCommandGatePPORunnerCfg
):
    experiment_name = "g1_parkour_stair_terrain_aux_cross_attn_terrain_command_gate"


@configclass
class G1ParkourTerrainAuxTerrainCommandGateGeoAuxPPORunnerCfg(
    G1ParkourTerrainAuxTerrainCommandGatePPORunnerCfg
):
    policy = TerrainAuxCrossAttentionTerrainCommandGateGeoAuxPolicyCfg()
    experiment_name = "g1_parkour_terrain_aux_cross_attn_terrain_command_gate_geo_aux"


@configclass
class G1ParkourStairTerrainAuxTerrainCommandGateGeoAuxPPORunnerCfg(
    G1ParkourTerrainAuxTerrainCommandGateGeoAuxPPORunnerCfg
):
    experiment_name = (
        "g1_parkour_stair_terrain_aux_cross_attn_terrain_command_gate_geo_aux"
    )
