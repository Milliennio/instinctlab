from isaaclab.utils import configclass

from instinctlab.sensors import get_link_prim_targets

from .g1_parkour_target_amp_cfg import G1_with_shoe_carry_box_CFG
from .g1_parkour_target_amp_terrain_aux_cfg import (
    G1ParkourStairTerrainAuxEnvCfg,
    G1ParkourStairTerrainAuxEnvCfg_PLAY,
)


CARRY_OBJECT_LINKS = ["carry_box_link"]


class CarryBoxConfigMixin:
    def apply_carry_box_config(self):
        self.scene.robot = G1_with_shoe_carry_box_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.camera.mesh_prim_paths.extend(get_link_prim_targets(CARRY_OBJECT_LINKS))


@configclass
class G1ParkourStairCarryTerrainAuxEnvCfg(G1ParkourStairTerrainAuxEnvCfg, CarryBoxConfigMixin):
    def __post_init__(self):
        super().__post_init__()
        self.apply_carry_box_config()


@configclass
class G1ParkourStairCarryTerrainAuxEnvCfg_PLAY(G1ParkourStairTerrainAuxEnvCfg_PLAY, CarryBoxConfigMixin):
    def __post_init__(self):
        super().__post_init__()
        self.apply_carry_box_config()
        self.scene.camera.debug_vis = True
        self.observations.policy.depth_image.params["debug_vis"] = True
