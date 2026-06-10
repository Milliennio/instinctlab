from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlEncoderMoEActorCriticCfg,
    InstinctRlOnPolicyRunnerCfg,
)

from .instinct_rl_amp_cfg import AmpAlgoCfg, EncoderConfigs


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
class GateSeparatedMoEPolicyCfg(InstinctRlEncoderMoEActorCriticCfg):
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


@configclass
class G1ParkourGateSeparatedPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 5000
    experiment_name = "g1_parkour_gate"
    resume = False
    load_run = ""
    empirical_normalization = False
    policy = GateSeparatedMoEPolicyCfg()
    algorithm = AmpAlgoCfg()
