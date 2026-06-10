from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

import instinctlab.tasks.parkour.mdp as mdp

from .g1_parkour_target_amp_cfg import G1ParkourEnvCfg, G1ParkourEnvCfg_PLAY
from .g1_parkour_target_amp_stair_cfg import G1ParkourStairEnvCfg, G1ParkourStairEnvCfg_PLAY


TERRAIN_AUX_GRID_RESOLUTION = 0.12
TERRAIN_AUX_GRID_SIZE = [1.2, 0.96]
TERRAIN_AUX_OUTPUT_SHAPE = (99,)


@configclass
class TerrainAuxObsCfg(ObsGroup):
    height_map = ObsTerm(
        func=mdp.local_terrain_height_map,
        params={
            "sensor_cfg": SceneEntityCfg("terrain_height_map_scanner"),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            "support_height_offset": 0.058,
            "min_height": -0.5,
            "max_height": 0.8,
        },
        noise=None,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


class TerrainAuxConfigMixin:
    def apply_terrain_aux_config(self):
        self.scene.terrain_height_map_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.45, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(
                resolution=TERRAIN_AUX_GRID_RESOLUTION,
                size=TERRAIN_AUX_GRID_SIZE,
            ),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
            update_period=0.02,
        )
        self.observations.terrain_aux = TerrainAuxObsCfg()


@configclass
class G1ParkourTerrainAuxEnvCfg(G1ParkourEnvCfg, TerrainAuxConfigMixin):
    def __post_init__(self):
        super().__post_init__()
        self.apply_terrain_aux_config()


@configclass
class G1ParkourTerrainAuxEnvCfg_PLAY(G1ParkourEnvCfg_PLAY, TerrainAuxConfigMixin):
    def __post_init__(self):
        super().__post_init__()
        self.apply_terrain_aux_config()


@configclass
class G1ParkourStairTerrainAuxEnvCfg(G1ParkourStairEnvCfg, TerrainAuxConfigMixin):
    def __post_init__(self):
        super().__post_init__()
        self.apply_terrain_aux_config()


@configclass
class G1ParkourStairTerrainAuxEnvCfg_PLAY(G1ParkourStairEnvCfg_PLAY, TerrainAuxConfigMixin):
    def __post_init__(self):
        super().__post_init__()
        self.apply_terrain_aux_config()
