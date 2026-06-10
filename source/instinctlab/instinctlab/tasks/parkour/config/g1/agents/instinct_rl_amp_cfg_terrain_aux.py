from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlConvTemporalTransformerHeadCfg,
    InstinctRlEncoderMoEActorCriticCfg,
    InstinctRlOnPolicyRunnerCfg,
)

from .instinct_rl_amp_cfg import AmpAlgoCfg


TERRAIN_AUX_OUTPUT_SHAPE = (99,)

ACTOR_MOE_GATE_COMPONENT_NAMES = [
    "projected_gravity",
    "velocity_commands",
    "base_ang_vel",
    "parallel_latent_0_depth_encoder",
]

CRITIC_MOE_GATE_COMPONENT_NAMES = [
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "parallel_latent_0_depth_encoder",
]


@configclass
class DepthEncoderTemporalTerrainCfg(InstinctRlConvTemporalTransformerHeadCfg):
    output_size = 128
    cnn_channels = [32, 64, 128, 256]
    cnn_kernel_sizes = [3, 3, 3, (3, 4)]
    cnn_strides = [2, 2, 2, 1]
    cnn_paddings = [1, 1, 1, 0]
    d_model = 256
    num_heads = 4
    num_layers = 1
    dim_feedforward = 512
    dropout = 0.1
    activation = "relu"
    nonlinearity = "ReLU"
    norm_first = True
    temporal_pool = "latest"
    use_temporal_pos_embedding = True
    output_hidden_sizes = []
    component_names = ["depth_image"]


@configclass
class EncoderConfigs:
    depth_encoder = DepthEncoderTemporalTerrainCfg()


@configclass
class TerrainAuxMoEPolicyCfg(InstinctRlEncoderMoEActorCriticCfg):
    class_name = "instinct_rl.modules.terrain_aux_actor_critic:TerrainAuxEncoderMoEActorCritic"

    init_noise_std = 1.0
    num_moe_experts = 4
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]
    activation = "elu"
    encoder_configs = EncoderConfigs()
    critic_encoder_configs = EncoderConfigs()

    moe_gate_hidden_dims = [128]
    moe_actor_gate_component_names = ACTOR_MOE_GATE_COMPONENT_NAMES
    moe_critic_gate_component_names = CRITIC_MOE_GATE_COMPONENT_NAMES

    terrain_aux_group_name = "terrain_aux"
    terrain_aux_latent_component_name = "parallel_latent_0_depth_encoder"
    terrain_aux_output_shape = TERRAIN_AUX_OUTPUT_SHAPE
    terrain_aux_hidden_dims = [128]
    terrain_aux_activation = "elu"
    terrain_aux_loss_func = "smooth_l1"
    terrain_aux_smooth_l1_beta = 0.05


@configclass
class TerrainAuxAmpAlgoCfg(AmpAlgoCfg):
    auxiliary_observation_group_names = ["terrain_aux"]
    terrain_reconstruction_loss_coef = 0.1


@configclass
class G1ParkourTerrainAuxPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 5000
    experiment_name = "g1_parkour_terrain_aux"
    resume = False
    load_run = ""
    empirical_normalization = False
    policy = TerrainAuxMoEPolicyCfg()
    algorithm = TerrainAuxAmpAlgoCfg()
