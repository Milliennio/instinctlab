import copy

from isaaclab.utils import configclass

from .g1_parkour_target_amp_cfg import (
    G1ParkourRoughEnvCfg,
    G1ParkourRoughEnvCfg_PLAY,
    ROUGH_TERRAINS_CFG_PLAY,
    ROUGH_TERRAINS_CFG,
    ShoeConfigMixin,
)


STAIR_TERRAIN_NAMES = (
    "perlin_rough",
    "perlin_rough_stand",
    "pyramid_stairs",
    "pyramid_stairs_high",
    "pyramid_stairs_inv",
    "pyramid_stairs_inv_high",
)

STAIR_TRAIN_NUM_ROWS = 6
STAIR_TRAIN_NUM_COLS = 12
STAIR_PLAY_NUM_ROWS = 4
STAIR_PLAY_NUM_COLS = 12


def _build_stair_terrain_cfg(base_cfg, num_rows: int, num_cols: int, zero_walls: bool = False):
    cfg = copy.deepcopy(base_cfg)
    cfg.sub_terrains = {name: copy.deepcopy(cfg.sub_terrains[name]) for name in STAIR_TERRAIN_NAMES}
    cfg.num_rows = num_rows
    cfg.num_cols = num_cols
    if zero_walls:
        for sub_terrain_cfg in cfg.sub_terrains.values():
            sub_terrain_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]
    return cfg


def _apply_stair_command_cfg(env_cfg):
    base_velocity = env_cfg.commands.base_velocity
    base_velocity.velocity_ranges = {
        name: copy.deepcopy(value)
        for name, value in base_velocity.velocity_ranges.items()
        if name in STAIR_TERRAIN_NAMES
    }
    if base_velocity.random_velocity_terrain is not None:
        base_velocity.random_velocity_terrain = [
            name for name in base_velocity.random_velocity_terrain if name in STAIR_TERRAIN_NAMES
        ] or None


STAIR_TERRAINS_CFG = _build_stair_terrain_cfg(
    ROUGH_TERRAINS_CFG,
    num_rows=STAIR_TRAIN_NUM_ROWS,
    num_cols=STAIR_TRAIN_NUM_COLS,
)
STAIR_TERRAINS_CFG_PLAY = _build_stair_terrain_cfg(
    ROUGH_TERRAINS_CFG_PLAY,
    num_rows=STAIR_PLAY_NUM_ROWS,
    num_cols=STAIR_PLAY_NUM_COLS,
    zero_walls=True,
)


@configclass
class G1ParkourStairEnvCfg(G1ParkourRoughEnvCfg, ShoeConfigMixin):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = copy.deepcopy(STAIR_TERRAINS_CFG)
        _apply_stair_command_cfg(self)
        self.apply_shoe_config()


@configclass
class G1ParkourStairEnvCfg_PLAY(G1ParkourRoughEnvCfg_PLAY, ShoeConfigMixin):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = copy.deepcopy(STAIR_TERRAINS_CFG_PLAY)
        _apply_stair_command_cfg(self)
        self.apply_shoe_config()
