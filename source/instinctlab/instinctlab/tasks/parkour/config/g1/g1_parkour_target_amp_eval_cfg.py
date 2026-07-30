import copy

from isaaclab.utils import configclass

from .g1_parkour_target_amp_cfg import ROUGH_TERRAINS_CFG
from .g1_parkour_target_amp_stair_cfg import STAIR_TERRAIN_NAMES
from .g1_parkour_target_amp_terrain_aux_cfg import (
    G1ParkourStairTerrainAuxEnvCfg,
    G1ParkourTerrainAuxEnvCfg,
)


ROUGH_EVAL_TERRAIN_NAMES = (
    "perlin_rough",
    "perlin_rough_stand",
    "square_gaps",
    "pyramid_stairs",
    "pyramid_stairs_high",
    "pyramid_stairs_inv",
    "pyramid_stairs_inv_high",
    "boxes",
)

EVAL_NUM_ENVS = 512
EVAL_EPISODE_LENGTH_S = 10.0
EVAL_NUM_ROWS = 6
ROUGH_EVAL_NUM_COLS = 20
STAIR_EVAL_NUM_COLS = 12
EVAL_TERRAIN_SEED = 0


def _zero_wall_prob(terrain_cfg):
    for sub_terrain_cfg in terrain_cfg.sub_terrains.values():
        if hasattr(sub_terrain_cfg, "wall_prob"):
            sub_terrain_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]


def build_eval_terrain_cfg(
    terrain_names: tuple[str, ...],
    *,
    num_rows: int = EVAL_NUM_ROWS,
    num_cols: int = ROUGH_EVAL_NUM_COLS,
    zero_walls: bool = True,
):
    """Build a deterministic terrain-generator config for batch evaluation."""
    cfg = copy.deepcopy(ROUGH_TERRAINS_CFG)
    cfg.sub_terrains = {name: copy.deepcopy(cfg.sub_terrains[name]) for name in terrain_names}
    cfg.num_rows = num_rows
    cfg.num_cols = num_cols
    cfg.curriculum = False
    cfg.seed = EVAL_TERRAIN_SEED
    if zero_walls:
        _zero_wall_prob(cfg)
    return cfg


def _filter_command_ranges(env_cfg, terrain_names: tuple[str, ...]):
    base_velocity = env_cfg.commands.base_velocity
    if base_velocity.velocity_ranges is not None:
        base_velocity.velocity_ranges = {
            name: copy.deepcopy(value)
            for name, value in base_velocity.velocity_ranges.items()
            if name in terrain_names
        }
    if base_velocity.random_velocity_terrain is not None:
        base_velocity.random_velocity_terrain = [
            name for name in base_velocity.random_velocity_terrain if name in terrain_names
        ] or None


def _apply_common_eval_config(env_cfg):
    env_cfg.scene.num_envs = EVAL_NUM_ENVS
    env_cfg.episode_length_s = EVAL_EPISODE_LENGTH_S

    terrain_generator = getattr(env_cfg.scene.terrain, "terrain_generator", None)
    if terrain_generator is not None:
        terrain_generator.curriculum = False
        env_cfg.scene.terrain.max_init_terrain_level = min(
            env_cfg.scene.terrain.max_init_terrain_level,
            terrain_generator.num_rows - 1,
        )

    if hasattr(env_cfg.curriculum, "terrain_levels"):
        env_cfg.curriculum.terrain_levels = None

    env_cfg.scene.terrain.debug_vis = False
    env_cfg.scene.camera.debug_vis = False
    env_cfg.scene.leg_volume_points.debug_vis = False
    if hasattr(env_cfg.scene, "terrain_height_map_scanner"):
        env_cfg.scene.terrain_height_map_scanner.debug_vis = False

    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.observations.policy.depth_image.params["debug_vis"] = False
    env_cfg.observations.critic.depth_image.params["debug_vis"] = False

    if hasattr(env_cfg.events, "physics_material"):
        env_cfg.events.physics_material = None
    if hasattr(env_cfg.events, "reset_robot_joints"):
        env_cfg.events.reset_robot_joints.params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }


@configclass
class G1ParkourTerrainAuxEnvCfg_EVAL(G1ParkourTerrainAuxEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = build_eval_terrain_cfg(
            ROUGH_EVAL_TERRAIN_NAMES,
            num_rows=EVAL_NUM_ROWS,
            num_cols=ROUGH_EVAL_NUM_COLS,
            zero_walls=True,
        )
        _filter_command_ranges(self, ROUGH_EVAL_TERRAIN_NAMES)
        _apply_common_eval_config(self)


@configclass
class G1ParkourStairTerrainAuxEnvCfg_EVAL(G1ParkourStairTerrainAuxEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = build_eval_terrain_cfg(
            STAIR_TERRAIN_NAMES,
            num_rows=EVAL_NUM_ROWS,
            num_cols=STAIR_EVAL_NUM_COLS,
            zero_walls=True,
        )
        _filter_command_ranges(self, STAIR_TERRAIN_NAMES)
        _apply_common_eval_config(self)
