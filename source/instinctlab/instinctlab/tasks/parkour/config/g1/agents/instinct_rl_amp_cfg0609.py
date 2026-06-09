from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlConvTemporalTransformerHeadCfg,
    InstinctRlEncoderMoEActorCriticCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)


@configclass
class DepthEncoderConvTemporalTransformerCfg(InstinctRlConvTemporalTransformerHeadCfg):
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
    component_names = [
        "depth_image",
    ]


@configclass
class EncoderConfigs:
    depth_encoder = DepthEncoderConvTemporalTransformerCfg()


@configclass
class MoEPolicyCfg(InstinctRlEncoderMoEActorCriticCfg):
    init_noise_std = 1.0
    num_moe_experts = 4
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]
    activation = "elu"
    encoder_configs = EncoderConfigs()
    critic_encoder_configs = EncoderConfigs()


@configclass
class AmpAlgoCfg(InstinctRlPpoAlgorithmCfg):
    class_name = "WasabiPPO"
    discriminator_kwargs = {
        "hidden_sizes": [1024, 512],
        "nonlinearity": "ReLU",
    }

    discriminator_reward_coef = 0.25
    discriminator_reward_type = "quad"
    discriminator_loss_func = "MSELoss"
    discriminator_gradient_penalty_coef = 5.0
    discriminator_optimizer_class_name = "AdamW"
    discriminator_weight_decay_coef = 3e-4
    discriminator_logit_weight_decay_coef = 0.04
    discriminator_optimizer_kwargs = {
        "lr": 1.0e-4,
        "betas": [0.9, 0.999],
    }
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.006
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0


@configclass
class G1ParkourPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 5000
    experiment_name = "g1_parkour"
    resume = False
    load_run = ""
    empirical_normalization = False
    policy = MoEPolicyCfg()
    algorithm = AmpAlgoCfg()
