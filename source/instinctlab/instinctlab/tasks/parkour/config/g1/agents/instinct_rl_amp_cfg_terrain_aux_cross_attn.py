from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlDepthCommandCrossAttentionHeadCfg,
    InstinctRlEncoderMoEActorCriticCfg,
    InstinctRlOnPolicyRunnerCfg,
)

from .instinct_rl_amp_cfg_terrain_aux import TERRAIN_AUX_OUTPUT_SHAPE, TerrainAuxAmpAlgoCfg


@configclass
class DepthCommandCrossAttentionTerrainCfg(InstinctRlDepthCommandCrossAttentionHeadCfg):
    output_size = 96
    cnn_channels = [16, 32, 64, 128]
    cnn_kernel_sizes = [3, 3, 3, (3, 4)]
    cnn_strides = [2, 2, 2, 1]
    cnn_paddings = [1, 1, 1, 0]
    d_model = 128
    num_heads = 4
    num_layers = 1
    dim_feedforward = 256
    dropout = 0.1
    activation = "relu"
    nonlinearity = "ReLU"
    norm_first = True
    use_temporal_pos_embedding = True
    command_hidden_sizes = []
    query_hidden_sizes = []
    output_hidden_sizes = []
    component_names = ["velocity_commands", "depth_image"]
    takeout_component_names = ["depth_image"]


@configclass
class EncoderConfigs:
    depth_encoder = DepthCommandCrossAttentionTerrainCfg()


@configclass
class TerrainAuxCrossAttentionMoEPolicyCfg(InstinctRlEncoderMoEActorCriticCfg):
    class_name = "instinct_rl.modules.terrain_aux_actor_critic:TerrainAuxEncoderMoEActorCritic"

    init_noise_std = 1.0
    num_moe_experts = 4
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]
    activation = "elu"
    encoder_configs = EncoderConfigs()
    critic_encoder_configs = EncoderConfigs()

    moe_gate_hidden_dims = [128]
    moe_actor_gate_component_names = None
    moe_critic_gate_component_names = None

    terrain_aux_group_name = "terrain_aux"
    terrain_aux_latent_component_name = "parallel_latent_0_depth_encoder"
    terrain_aux_output_shape = TERRAIN_AUX_OUTPUT_SHAPE
    terrain_aux_hidden_dims = [96]
    terrain_aux_activation = "elu"
    terrain_aux_loss_func = "smooth_l1"
    terrain_aux_smooth_l1_beta = 0.05


@configclass
class G1ParkourTerrainAuxCrossAttentionPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 5000
    experiment_name = "g1_parkour_terrain_aux_cross_attn"
    resume = False
    load_run = ""
    empirical_normalization = False
    policy = TerrainAuxCrossAttentionMoEPolicyCfg()
    algorithm = TerrainAuxAmpAlgoCfg()
