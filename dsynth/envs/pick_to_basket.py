import torch
import numpy as np
import pandas as pd
import os
import sapien
import sapien.physx as physx
from transforms3d.euler import euler2quat
from mani_skill.utils import common, sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env

from dsynth.envs.darkstore_cell_base import DarkstoreCellBaseEnv
from dsynth.envs.darkstore_cont_base import DarkstoreContinuousBaseEnv

from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
from mani_skill.utils.structs.pose import Pose
from dsynth.scene_gen.arrangements import CELL_SIZE
import copy

@register_env('PickToBasketEnv', max_episode_steps=200000)
class PickToBasketEnv(DarkstoreCellBaseEnv):
    TARGET_PRODUCT_NAME = None
    ROBOT_INIT_POSE_RANDOM_ENABLED = True
    
    def _load_scene(self, options: dict):
        super()._load_scene(options)
        
        self.target_sizes = np.array([0.3, 0.3, 0.3])

        if self.markers_enabled:
            self.target_volumes = {}
            for n_env in range(self.num_envs):
                self.target_volumes[n_env] = actors.build_box(
                    self.scene,
                    half_sizes=list(self.target_sizes/2),
                    color=[0, 1, 0, 0.5],
                    name=f"target_box_{n_env}",
                    body_type="kinematic",
                    add_collision=False,
                    scene_idxs=[n_env],
                    initial_pose=sapien.Pose(p=[0, 0, 0]),
                )
                self.hide_object(self.target_volumes[n_env])
        
            self.target_markers = {}
            for n_env in range(self.num_envs):
                self.target_markers[n_env] = []
                for i in range(self.NUM_MARKERS):
                    self.target_markers[n_env].append(
                                    actors.build_sphere(
                                        self.scene,
                                        radius=0.05,
                                        color=[0, 1, 0, 1],
                                        name=f"target_product_{n_env}_{i}",
                                        body_type="kinematic",
                                        add_collision=False,
                                        initial_pose=sapien.Pose(p=[0., 0., 0.]),
                                        scene_idxs=[n_env]
                                    )
                                )
                    self.hide_object(self.target_markers[n_env][-1])
    
    def setup_target_objects(self, env_idxs):
        self.target_product_names = {}
        self.target_zones = {}
        self.target_shelves = {}
        self.target_products_df = None
        
        if self.markers_enabled:
            target_markers_iterator = {key: iter(val) for key, val in self.target_markers.items()}

        self.target_product_names = {idx: self.TARGET_PRODUCT_NAME for idx in range(self.num_envs)}

        for scene_idx in env_idxs:
            scene_idx = scene_idx.cpu().item()
            scene_prducts_df = self.products_df[self.products_df['scene_idx'] == scene_idx]
            
            if self.TARGET_PRODUCT_NAME is None:
                # select random zone, shelf and product
                zone_id = self._batched_episode_rng[scene_idx].choice(sorted(scene_prducts_df['zone_id'].unique()))
                self.target_zones[scene_idx] = zone_id

                zone_products_df = scene_prducts_df[scene_prducts_df['zone_id'] == zone_id]
                shelf_id = self._batched_episode_rng[scene_idx].choice(sorted(zone_products_df['shelf_id'].unique()))
                self.target_shelves[scene_idx] = shelf_id

                shelf_products_df = zone_products_df[zone_products_df['shelf_id'] == shelf_id]
                product_name = self._batched_episode_rng[scene_idx].choice(sorted(shelf_products_df['product_name'].unique()))
                self.target_product_names[scene_idx] = product_name

                if self.target_products_df is None:
                    self.target_products_df = shelf_products_df[shelf_products_df['product_name'] == product_name]
                else:
                    self.target_products_df = pd.concat([self.target_products_df,
                        shelf_products_df[shelf_products_df['product_name'] == product_name]
                                                      ])
            else:
                # select random zone and shelf with self.TARGET_PRODUCT_NAME

                if not self.TARGET_PRODUCT_NAME in scene_prducts_df['product_name'].unique():
                    raise RuntimeError(f"Product {self.TARGET_PRODUCT_NAME} is not present on scene #{scene_idx}")
                
                zones_w_target_product = scene_prducts_df[scene_prducts_df['product_name'] == self.TARGET_PRODUCT_NAME]
                zone_id = self._batched_episode_rng[scene_idx].choice(sorted(zones_w_target_product['zone_id'].unique()))
                self.target_zones[scene_idx] = zone_id

                shelves_w_target_zone = zones_w_target_product[zones_w_target_product['zone_id'] == zone_id]
                shelf_id = self._batched_episode_rng[scene_idx].choice(sorted(shelves_w_target_zone['shelf_id'].unique()))
                self.target_shelves[scene_idx] = shelf_id

                if self.target_products_df is None:
                    self.target_products_df = shelves_w_target_zone[shelves_w_target_zone['shelf_id'] == shelf_id]
                else:
                    self.target_products_df = pd.concat([self.target_products_df,
                        shelves_w_target_zone[shelves_w_target_zone['shelf_id'] == shelf_id]
                    ])

            if self.markers_enabled:
                target_products = self.target_products_df[self.target_products_df['scene_idx'] == scene_idx]
                for actor_name in target_products['actor_name']:

                    # select only 4th in each column - they are near the edge
                    if int(actor_name.split(':')[-1]) % 4 == 0: # TODO: redo
                        actor = self.actors['products'][actor_name]
                        try:
                            target_marker = next(target_markers_iterator[scene_idx])
                        except StopIteration:
                            raise RuntimeError(f"Number of target objects exceeds number of markers ({self.NUM_MARKERS}) for scene #{scene_idx}")
                        target_marker.set_pose(actor.pose)

    def _compute_robot_init_pose(self, env_idx = None):
        origins = []
        init_cells = []
        angles = []
        directions_to_shelf = []

        for idx in env_idx:
            idx = idx.cpu().item()
            scene_target_products = self.target_products_df[self.target_products_df['scene_idx'] == idx].reset_index()
            shelf_i, shelf_j = scene_target_products['i'][0], scene_target_products['j'][0]
            rot = self.scene_builder.rotations[idx][shelf_i][shelf_j]

            if rot == 0:
                origin, angle, direction_to_shelf = np.array([shelf_i, shelf_j - 1, 0.]), np.pi / 2, np.array([0, 1, 0])
            if rot == -90:
                origin, angle, direction_to_shelf = np.array([shelf_i - 1, shelf_j, 0.]), 0 , np.array([1, 0, 0])
            if rot == 90:
                origin, angle, direction_to_shelf = np.array([shelf_i + 1, shelf_j, 0.]), np.pi, np.array([-1, 0, 0])
            if rot == 180:
                origin, angle, direction_to_shelf = np.array([shelf_i, shelf_j + 1, 0.]), - np.pi / 2, np.array([0, -1, 0])
            
            # self.target_drive_position = origin.copy() + direction_to_shelf * CELL_SIZE * 0.2
            
            init_cell = np.array([origin[0], origin[1]])
            origin = origin * CELL_SIZE
            origin[:2] += CELL_SIZE / 2

            if self.ROBOT_INIT_POSE_RANDOM_ENABLED:
                # base movement enabled, add initial pose randomization
                perp_direction = np.cross(direction_to_shelf, [0, 0, 1])

                delta_par = self._batched_episode_rng[idx].rand() * CELL_SIZE * 0.4
                delta_perp = (self._batched_episode_rng[idx].rand() - 0.5) * 2 * CELL_SIZE * 0.4

                origin += - direction_to_shelf * delta_par + perp_direction * delta_perp

                angle += (self._batched_episode_rng[idx].rand() - 0.5) * np.pi / 4

            origins.append(origin)
            init_cells.append(init_cell)
            angles.append(angle)
            directions_to_shelf.append(direction_to_shelf)

        return np.array(origins), np.array(init_cells), np.array(angles), np.array(directions_to_shelf)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.setup_target_objects(env_idx)
        self.setup_language_instructions(env_idx)

        b = len(env_idx)
        if self.robot_uids == "panda_wristcam":
            qpos = np.array(
                [
                    -0.006,        
                    -1.467,
                    0.012,        
                    -2.823,
                    0.003,        
                    2.928,
                    0.796,
                    0.04,       
                    0.04,       
                ]
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(sapien.Pose([0.5, 1.7, 0.0]))

        elif self.robot_uids in ["ds_fetch_basket", "ds_fetch", "fetch"]:
            qpos = np.array(
                [
                    0,
                    0,
                    0,
                    0.36,
                    0,
                    0,
                    0,
                    1.4,
                    0,
                    0.76,
                    0,
                    - 2 * np.pi / 3,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.agent.reset(qpos)
            self.robot_origins, self.init_cells, self.robot_angles, self.directions_to_shelf = self._compute_robot_init_pose(env_idx)
            quats = np.array([euler2quat(0, 0, robot_angle) for robot_angle in self.robot_angles])
            self.agent.robot.set_pose(Pose.create_from_pq(p=self.robot_origins, q=quats))
        elif self.robot_uids in ["ds_fetch_static", "ds_fetch_basket_static"]:
            qpos = np.array(
                [
                    0.36,
                    0,
                    0,
                    0,
                    1.4,
                    0,
                    0.76,
                    0,
                    - 2 * np.pi / 3,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.agent.reset(qpos)
            self.robot_origins, self.init_cells, self.robot_angles, self.directions_to_shelf = self._compute_robot_init_pose(env_idx)
            quats = np.array([euler2quat(0, 0, robot_angle) for robot_angle in self.robot_angles])
            self.agent.robot.set_pose(Pose.create_from_pq(p=self.robot_origins, q=quats))

        if self.markers_enabled:
            target_pose = self.calc_target_pose()
            for scene_idx in env_idx:
                scene_idx = scene_idx.cpu().item()
                self.target_volumes[scene_idx].set_pose(
                    Pose.create_from_pq(p=target_pose.p[scene_idx],
                                        q=target_pose.q[scene_idx])
                )
    
    def evaluate(self):
        target_pos = self.calc_target_pose().p 
        target_pos[:, 2] -= self.target_sizes[2] / 2
        tolerance = torch.tensor(self.target_sizes / 2, dtype=torch.float32).to(self.device)
        is_obj_placed = []

        for scene_idx in range(self.num_envs):
            scene_is_obj_placed = False
            scene_target_products_df = self.target_products_df[self.target_products_df['scene_idx'] == scene_idx]
            for actor_name in scene_target_products_df['actor_name']:
                target_product_pos = self.actors['products'][actor_name].pose.p
                scene_is_obj_placed = torch.all(
                    (target_product_pos >= (target_pos[scene_idx] - tolerance)) & 
                    (target_product_pos <= (target_pos[scene_idx] + tolerance)),
                    dim=-1
                )
                if scene_is_obj_placed:
                    break
            
            is_obj_placed.append(scene_is_obj_placed)

        is_obj_placed = torch.cat(is_obj_placed)
        
        is_robot_static = self.agent.is_static(0.2)

        is_non_target_produncts_replaced = torch.zeros_like(is_robot_static, dtype=bool)

        for scene_idx in range(self.num_envs):
            scene_products_df = self.products_df[self.products_df['scene_idx'] == scene_idx]

            # to speed up evaluation only check products from the target shelf
            scene_products_df = scene_products_df[scene_products_df['shelf_id'] == self.target_shelves[scene_idx]]

            scene_target_products_df = self.target_products_df[self.target_products_df['scene_idx'] == scene_idx]
            non_target_actors = set(scene_products_df['actor_name']) - set(scene_target_products_df['actor_name'])
            
            for actor_name in non_target_actors:
                actor = self.actors['products'][actor_name]
                if actor_name in self.products_initial_poses:
                    if not torch.all(torch.isclose(actor.pose.raw_pose, self.products_initial_poses[actor_name], rtol=0.1, atol=0.1)):
                        is_non_target_produncts_replaced[scene_idx] = True

                        if self.markers_enabled:
                            # make marker red if non-target product moved
                            render_component = self.target_volumes[scene_idx]._objs[0].find_component_by_type(
                                sapien.pysapien.render.RenderBodyComponent
                            )
                            render_component.render_shapes[0].material.base_color = [1.0, 0.0, 0.0, 0.5]

                        break


        return {
            "is_obj_placed" : is_obj_placed,
            "is_robot_static" : is_robot_static,
            "is_non_target_produncts_displaced" : is_non_target_produncts_replaced,
            "success": is_obj_placed & is_robot_static & (~is_non_target_produncts_replaced),
            # "success": is_obj_placed & is_robot_static,
        }

    def calc_target_pose(self):
        robot_pose = self.agent.base_link.pose
        basket_shift = Pose.create_from_pq(p=[[0.3, 0.25, 0.14]] * self.num_envs)
        return robot_pose * basket_shift 
       

    def setup_language_instructions(self, env_idx):
        self.language_instructions = []
        for scene_idx in env_idx:
            scene_idx = scene_idx.cpu().item()
            self.language_instructions.append(f'move to the shelf and pick {self.target_product_names[scene_idx]} and put to the basket')

    def _after_simulation_step(self):
        #does not work on gpu sim
        if self.markers_enabled:
            target_pose = self.calc_target_pose()
            for scene_idx in range(self.num_envs):
                self.target_volumes[scene_idx].set_pose(
                    Pose.create_from_pq(p=target_pose.p[scene_idx],
                                        q=target_pose.q[scene_idx])
                )
            # self.target_volume.set_pose(target_pose)

@register_env('PickToBasketStaticEnv', max_episode_steps=200000)
class PickToBasketStaticEnv(PickToBasketEnv):
    ROBOT_INIT_POSE_RANDOM_ENABLED = False
@register_env('PickToBasketSpriteEnv', max_episode_steps=200000)
class PickToBasketSpriteEnv(PickToBasketEnv):
    TARGET_PRODUCT_NAME = 'sprite'

@register_env('PickToBasketStaticSpriteEnv', max_episode_steps=200000)
class PickToBasketStaticSpriteEnv(PickToBasketEnv):
    TARGET_PRODUCT_NAME = 'sprite'
    ROBOT_INIT_POSE_RANDOM_ENABLED = False

@register_env('PickToBasketContEnv', max_episode_steps=200000)
class PickToBasketContEnv(DarkstoreContinuousBaseEnv):
    """
    **Task Description:**
    Approach the shelf and pick up the item specified by `TARGET_PRODUCT_NAME`, placing it into the basket attached to the Fetch robot.
    If `TARGET_PRODUCT_NAME` is None, it is randomly selected from the set of item names present in the scene.
    The robot is spawned in close proximity to the shelf.

    **Randomizations:**
    - scene layout, object arrangement, wall and floor textures
    - initial robot position, if `ROBOT_INIT_POSE_RANDOM_ENABLED` is enabled (True by default)

    **Success Conditions:**
    - any product item with the name `TARGET_PRODUCT_NAME` is within `TARGET_POS_THRESH` Euclidean distance of the goal position (the Fetch robot's basket).
    - other items remain untouched (their positions change by no more than 0.1 m)
    - the robot is static (q velocity < 0.2)
    """

    TARGET_PRODUCT_NAME = None
    ROBOT_INIT_POSE_RANDOM_ENABLED = True

    TARGET_POS_THRESH = 0.2
    
    def _load_scene(self, options: dict):
        super()._load_scene(options)
        
        self.target_sizes = np.array([self.TARGET_POS_THRESH, self.TARGET_POS_THRESH, self.TARGET_POS_THRESH])
    
    def setup_target_objects(self, env_idxs):
        self.target_product_names = {}
        self.target_products_df = None
        
        if self.markers_enabled:
            target_markers_iterator = {key: iter(val) for key, val in self.target_markers.items()}

        self.target_product_names = {idx: self.TARGET_PRODUCT_NAME for idx in range(self.num_envs)}

        for scene_idx in env_idxs:
            scene_idx = scene_idx.cpu().item()
            scene_prducts_df = self.products_df[self.products_df['scene_idx'] == scene_idx]
            
            if self.TARGET_PRODUCT_NAME is None:
                product_name = self._batched_episode_rng[scene_idx].choice(sorted(scene_prducts_df['product_name'].unique()))
                self.target_product_names[scene_idx] = product_name
        
            else:
                product_name = self.TARGET_PRODUCT_NAME
                if not self.TARGET_PRODUCT_NAME in scene_prducts_df['product_name'].unique():
                    raise RuntimeError(f"Product {self.TARGET_PRODUCT_NAME} is not present on scene #{scene_idx}")
            
            if self.target_products_df is None:
                self.target_products_df = scene_prducts_df[scene_prducts_df['product_name'] == product_name]
            else:
                self.target_products_df = pd.concat([self.target_products_df,
                    scene_prducts_df[scene_prducts_df['product_name'] == product_name]
                                                    ])
            
            if self.markers_enabled:
                target_products = self.target_products_df[self.target_products_df['scene_idx'] == scene_idx]
                for actor_name in target_products['actor_name']:
                    actor = self.actors['products'][actor_name]
                    try:
                        target_marker = next(target_markers_iterator[scene_idx])
                    except StopIteration:
                        raise RuntimeError(f"Number of target objects exceeds number of markers ({self.NUM_MARKERS}) for scene #{scene_idx}")
                    target_marker.set_pose(actor.pose)

    def _compute_robot_init_pose(self, env_idx = None):
        robot_origins, robot_angles, directions_to_shelf = super()._compute_robot_init_pose(env_idx)
        for idx in env_idx:
            if self.ROBOT_INIT_POSE_RANDOM_ENABLED:
                # base movement enabled, add initial pose randomization
                batched_rng = self._batched_episode_rng
                if self.extra_robot_pose_randomization:
                    batched_rng = self._batched_init_pose_rng
                idx = idx.cpu().item()
                direction_to_shelf = directions_to_shelf[idx]
                perp_direction = np.cross(direction_to_shelf, [0, 0, 1])

                delta_par = batched_rng[idx].rand() * 0.2
                delta_perp = (batched_rng[idx].rand() - 0.5) * 0.5

                robot_origins[idx] += -direction_to_shelf * delta_par + perp_direction * delta_perp
                robot_angles[idx] += (batched_rng[idx].rand() - 0.5) * np.pi / 4

        return robot_origins, robot_angles, directions_to_shelf
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        if self.robot_uids in ["fetch", "ds_fetch", "ds_fetch_basket"]:
            qpos = np.array(
                [
                    0,
                    0,
                    0,
                    0.36,
                    0, #unused
                    0,
                    0, #unused
                    0.75,
                    0,
                    0.81,
                    0,
                    -0.78,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.agent.reset(qpos)

    def evaluate(self):
        target_pos = self.calc_target_pose().p 
        # target_pos[:, 2] -= self.target_sizes[2] / 2
        # tolerance = torch.tensor(self.target_sizes / 2, dtype=torch.float32).to(self.device)
        tolerance = torch.tensor([self.TARGET_POS_THRESH, self.TARGET_POS_THRESH, self.TARGET_POS_THRESH]).to(self.device)
        is_obj_placed = []

        for scene_idx in range(self.num_envs):
            scene_is_obj_placed = False
            scene_target_products_df = self.target_products_df[self.target_products_df['scene_idx'] == scene_idx]
            for actor_name in scene_target_products_df['actor_name']:
                target_product_pos = self.actors['products'][actor_name].pose.p
                scene_is_obj_placed = torch.all(
                    (target_product_pos >= (target_pos[scene_idx] - tolerance)) & 
                    (target_product_pos <= (target_pos[scene_idx] + tolerance)),
                    dim=-1
                )
                if scene_is_obj_placed:
                    break
            
            is_obj_placed.append(scene_is_obj_placed)

        is_obj_placed = torch.cat(is_obj_placed)
        
        is_robot_static = self.agent.is_static(0.2)

        is_non_target_produncts_replaced = torch.zeros_like(is_robot_static, dtype=bool)

        for scene_idx in range(self.num_envs):
            scene_products_df = self.products_df[self.products_df['scene_idx'] == scene_idx]

            scene_target_products_df = self.target_products_df[self.target_products_df['scene_idx'] == scene_idx]
            non_target_actors = set(scene_products_df['actor_name']) - set(scene_target_products_df['actor_name'])
            
            for actor_name in non_target_actors:
                actor = self.actors['products'][actor_name]
                if actor_name in self.products_initial_poses:
                    if not torch.all(torch.isclose(actor.pose.raw_pose, self.products_initial_poses[actor_name], rtol=0.1, atol=0.1)):
                        is_non_target_produncts_replaced[scene_idx] = True

                        if self.markers_enabled:
                            # make marker red if non-target product moved
                            render_component = self.target_volumes[scene_idx][0]._objs[0].find_component_by_type(
                                sapien.pysapien.render.RenderBodyComponent
                            )
                            render_component.render_shapes[0].material.base_color = [1.0, 0.0, 0.0, 0.5]

                        break


        return {
            "is_obj_placed" : is_obj_placed,
            "is_robot_static" : is_robot_static,
            "is_non_target_produncts_displaced" : is_non_target_produncts_replaced,
            "success": is_obj_placed & is_robot_static & (~is_non_target_produncts_replaced),
            # "success": is_obj_placed & is_robot_static,
        }

    def calc_target_pose(self):
        robot_pose = self.agent.base_link.pose
        basket_shift = Pose.create_from_pq(p=[[0.3, 0.25, 0.14]] * self.num_envs)
        return robot_pose * basket_shift 
       

    def setup_language_instructions(self, env_idx):
        self.language_instructions = []
        for scene_idx in env_idx:
            scene_idx = scene_idx.cpu().item()
            self.language_instructions.append(f'move to shelf and pick {self.target_product_names[scene_idx]} to basket')

    def _after_simulation_step(self):
        #does not work on gpu sim
        if self.markers_enabled:
            target_pose = self.calc_target_pose()
            for scene_idx in range(self.num_envs):
                self.target_volumes[scene_idx][0].set_pose(
                    Pose.create_from_pq(p=target_pose.p[scene_idx],
                                        q=target_pose.q[scene_idx])
                )
            # self.target_volume.set_pose(target_pose)


@register_env('PickToBasketContActorEvalEnv', max_episode_steps=200000)
class PickToBasketContActorEvalEnv(PickToBasketContEnv):
    """
    PickToBasket continuous env where the evaluation target can be overridden
    to a specific actor (per scene) at runtime.

    Default behavior (no override) matches :class:`PickToBasketContEnv`: on
    every reset/reconfigure, :meth:`setup_target_objects` samples a random
    product per scene. Call :meth:`set_target_actor` AFTER reset to re-scope
    the env's native ``evaluate()`` (and therefore per-step ``info["success"]``
    and the ``success`` field recorded in .h5 / .json) to a specific actor
    that you actually intend to grasp.

    Usage::

        obs, _ = env.reset(seed=seed, options={"reconfigure": True})
        env.unwrapped.set_target_actor("food.CANNED.SlamLuncheonMeat:0:0:0:0")
        # ...run policy; env.step() will now report success against that actor
    """

    # --- Dedicated high-res camera used only for GraspGen point cloud capture.
    # It is intentionally separate from the robot's standard cameras so the
    # user can safely downscale obs/sensor cameras (e.g. 256x256) to get
    # smaller .h5 / video files, without breaking GraspGen (which needs
    # ~1024x1024 point clouds to work well).
    GRASPGEN_PC_CAM_UID: str = "graspgen_pc_camera"
    GRASPGEN_PC_RES: int = 1024

    @property
    def _default_sensor_configs(self):
        base = list(super()._default_sensor_configs or [])
        mount = None
        try:
            mount = self.agent.robot.links_map["head_camera_link"]
        except Exception:
            mount = None
        # Pose cloned from DSFetch.left_base_camera_link so the frustum covers
        # the same shelf/target region, just at high resolution.
        pose = Pose.create_from_pq(
            [-0.5, 0.5, 0], euler2quat(0, 0.3, -0.2)
        )
        pc_cam = CameraConfig(
            uid=self.GRASPGEN_PC_CAM_UID,
            pose=pose,
            width=self.GRASPGEN_PC_RES,
            height=self.GRASPGEN_PC_RES,
            fov=1.5,
            near=0.01,
            far=100,
            mount=mount,
            shader_pack="minimal",
        )
        return base + [pc_cam]

    def _setup_sensors(self, options):
        """Ensure the GraspGen PC camera keeps high-res regardless of the
        user-supplied ``sensor_configs`` override in ``gym.make``.

        A *flat* ``sensor_configs=dict(width=256, height=256, ...)`` normally
        applies to every camera. Here we inject a per-camera override for
        :attr:`GRASPGEN_PC_CAM_UID` that restores the full resolution *after*
        the global override is applied (per-camera entries are applied last by
        ManiSkill).
        """
        uid = self.GRASPGEN_PC_CAM_UID
        custom = self._custom_sensor_configs
        if isinstance(custom, dict):
            patched = dict(custom)
            cam_spec = dict(patched.get(uid) or {})
            cam_spec.setdefault("width", self.GRASPGEN_PC_RES)
            cam_spec.setdefault("height", self.GRASPGEN_PC_RES)
            patched[uid] = cam_spec
            self._custom_sensor_configs = patched
        super()._setup_sensors(options)

    def _get_obs_sensor_data(self, apply_texture_transforms: bool = True) -> dict:
        """Same as :meth:`BaseEnv._get_obs_sensor_data` but skips the
        dedicated high-res GraspGen PC camera, so it is not rendered/read
        during normal ``step()`` (which would defeat the point of using
        low-res obs cameras)."""
        from mani_skill.sensors.camera import Camera

        uid = self.GRASPGEN_PC_CAM_UID
        for obj in self._hidden_objects:
            obj.hide_visual()
        self.scene.update_render(
            update_sensors=True, update_human_render_cameras=False
        )
        for name, sensor in self._sensors.items():
            if name == uid:
                continue
            sensor.capture()
        sensor_obs: dict = {}
        for name, sensor in self.scene.sensors.items():
            if name == uid:
                continue
            if not isinstance(sensor, Camera):
                continue
            if self.obs_mode in ("state", "state_dict"):
                sensor_obs[name] = sensor.get_obs(
                    position=False,
                    segmentation=False,
                    apply_texture_transforms=apply_texture_transforms,
                )
            else:
                sensor_obs[name] = sensor.get_obs(
                    rgb=self.obs_mode_struct.visual.rgb,
                    depth=self.obs_mode_struct.visual.depth,
                    position=self.obs_mode_struct.visual.position,
                    segmentation=self.obs_mode_struct.visual.segmentation,
                    normal=self.obs_mode_struct.visual.normal,
                    albedo=self.obs_mode_struct.visual.albedo,
                    apply_texture_transforms=apply_texture_transforms,
                )
        if self.backend.render_device.is_cuda():
            torch.cuda.synchronize()
        return sensor_obs

    def get_sensor_params(self) -> dict:
        """Strip GraspGen PC camera params from the standard observation.
        They are still accessible internally for :meth:`capture_pointcloud`.
        """
        params = super().get_sensor_params()
        if isinstance(params, dict):
            params.pop(self.GRASPGEN_PC_CAM_UID, None)
        return params

    def set_target_actor(
        self,
        actor_name: str,
        scene_idx: int = 0,
    ) -> bool:
        """Scope :meth:`evaluate` to a specific actor for the given scene.

        Rebuilds ``self.target_products_df`` (row(s) for ``scene_idx``) and
        ``self.target_product_names`` to point at ``actor_name``. Other scenes'
        targets are preserved. The override is implicitly wiped on the next
        ``reset(reconfigure=True)`` because :meth:`setup_target_objects`
        reinitializes both fields from scratch.

        Returns:
            True if ``actor_name`` was found in ``products_df`` for ``scene_idx``
            and the override was applied; False otherwise.
        """
        products_df = getattr(self, "products_df", None)
        if products_df is None or len(products_df) == 0:
            return False
        if "actor_name" not in products_df.columns or "scene_idx" not in products_df.columns:
            return False

        actor_row = products_df[
            (products_df["scene_idx"] == int(scene_idx))
            & (products_df["actor_name"] == str(actor_name))
        ]
        if actor_row.empty:
            return False

        current = getattr(self, "target_products_df", None)
        if current is not None and len(current) > 0:
            keep = current[current["scene_idx"] != int(scene_idx)]
            self.target_products_df = pd.concat(
                [keep, actor_row], ignore_index=True
            )
        else:
            self.target_products_df = actor_row.reset_index(drop=True)

        if self.target_product_names is None:
            self.target_product_names = {}
        if "product_name" in actor_row.columns:
            self.target_product_names[int(scene_idx)] = str(
                actor_row.iloc[0]["product_name"]
            )

        return True

    def get_target_segmentation_ids(self) -> dict:
        """Return ``{scene_idx: per_scene_id}`` for the current target actor(s).

        ``per_scene_id`` is the integer that ManiSkill writes into the
        ``segmentation`` texture for the target actor's entity, so a
        target-only binary mask can be obtained as
        ``(segmentation == per_scene_id).astype(uint8)``.
        """
        out: dict = {}
        target_df = getattr(self, "target_products_df", None)
        if target_df is None or len(target_df) == 0:
            return out
        products = getattr(self, "actors", {}).get("products", {})
        if not products:
            return out
        for _, row in target_df.iterrows():
            actor_name = row.get("actor_name")
            try:
                scene_idx = int(row.get("scene_idx", 0))
            except Exception:
                continue
            actor = products.get(actor_name)
            if actor is None:
                continue
            try:
                out[scene_idx] = int(actor._objs[0].per_scene_id)
            except Exception:
                continue
        return out

    @staticmethod
    def _compute_target_mask(seg, target_seg_ids: dict):
        """Build a per-scene binary target mask matching the layout of ``seg``.

        ``seg`` is the segmentation tensor as produced by ManiSkill cameras /
        pointcloud (shape ``[B, ..., 1]``, integer dtype). We return a tensor
        of the same shape and ``torch.uint8``, where element is ``1`` iff the
        segmentation id equals the target actor's ``per_scene_id`` for that
        scene (batch) index.

        Scenes not present in ``target_seg_ids`` (or out of range) receive an
        all-zero mask.
        """
        mask = torch.zeros_like(seg, dtype=torch.uint8)
        if seg.ndim == 0:
            return mask
        batch = seg.shape[0]
        for scene_idx, sid in target_seg_ids.items():
            if 0 <= int(scene_idx) < batch:
                mask[int(scene_idx)] = (seg[int(scene_idx)] == int(sid)).to(torch.uint8)
        return mask

    def get_obs(self, info=None, unflattened: bool = False):
        """Add ``target_mask`` next to ``segmentation`` in the observation.

        - For camera-based obs modes (``rgb+depth+segmentation`` etc.): every
          camera in ``obs["sensor_data"]`` that carries a ``segmentation`` key
          will also get a ``target_mask`` key with shape ``[B, H, W, 1]`` and
          dtype ``torch.uint8``.
        - For ``pointcloud`` obs mode: if ``obs["pointcloud"]["segmentation"]``
          exists, a matching flat ``target_mask`` of shape ``[B, N, 1]`` will
          be added.

        If the segmentation texture is not requested in ``obs_mode`` (e.g.
        plain ``rgb+depth``), nothing is added and behavior is unchanged.
        """
        obs = super().get_obs(info=info, unflattened=unflattened)
        if not isinstance(obs, dict):
            return obs
        target_seg_ids = self.get_target_segmentation_ids()
        if not target_seg_ids:
            return obs

        sensor_data = obs.get("sensor_data")
        if isinstance(sensor_data, dict):
            for cam_data in sensor_data.values():
                if not isinstance(cam_data, dict):
                    continue
                seg = cam_data.get("segmentation")
                if seg is None:
                    continue
                cam_data["target_mask"] = self._compute_target_mask(seg, target_seg_ids)

        pcd = obs.get("pointcloud")
        if isinstance(pcd, dict):
            seg = pcd.get("segmentation")
            if seg is not None:
                pcd["target_mask"] = self._compute_target_mask(seg, target_seg_ids)

        return obs

    def capture_pointcloud(self, rgb: bool = True) -> dict:
        """Force-capture a fused pointcloud from the dedicated high-res
        GraspGen camera (``GRASPGEN_PC_CAM_UID``), independent of ``obs_mode``.

        This lets you run the env in a low-res camera-based ``obs_mode``
        (so the recorder writes compact rgb/depth/segmentation/target_mask
        into .h5 for LeRobot/RLDS training) while still feeding GraspGen a
        proper ~1024x1024 fused point cloud on demand.

        Returns:
            A dict with the same layout that ``obs_mode="pointcloud"`` would
            produce under ``obs["pointcloud"]``:

                - ``xyzw``:         ``[B, N, 4]`` ``torch.float32`` (world frame,
                  ``w == 0`` for invalid / far points, else ``1``).
                - ``rgb``:          ``[B, N, 3]`` ``torch.uint8`` (only if
                  ``rgb=True``).
                - ``segmentation``: ``[B, N, 1]`` ``torch.int16``.
                - ``target_mask``:  ``[B, N, 1]`` ``torch.uint8`` — present only
                  when a target actor is known (i.e. after
                  :meth:`set_target_actor` or the default sampling).
        """
        from mani_skill.sensors.camera import Camera
        from mani_skill.envs.utils.observations.observations import (
            sensor_data_to_pointcloud,
        )

        uid = self.GRASPGEN_PC_CAM_UID
        pc_sensor = self._sensors.get(uid)
        if pc_sensor is None or not isinstance(pc_sensor, Camera):
            raise RuntimeError(
                f"capture_pointcloud(): dedicated camera {uid!r} is not "
                f"registered. Available sensors: {list(self._sensors)}"
            )

        for obj in self._hidden_objects:
            obj.hide_visual()
        self.scene.update_render(
            update_sensors=True, update_human_render_cameras=False
        )
        pc_sensor.capture()

        sensor_obs = {
            uid: pc_sensor.get_obs(
                rgb=bool(rgb),
                depth=False,
                position=True,
                segmentation=True,
                apply_texture_transforms=True,
            )
        }
        if self.backend.render_device.is_cuda():
            torch.cuda.synchronize()

        fake_obs = dict(
            sensor_data=sensor_obs,
            sensor_param={uid: pc_sensor.get_params()},
        )
        fake_obs = sensor_data_to_pointcloud(fake_obs, {uid: pc_sensor})
        pcd = fake_obs["pointcloud"]

        target_seg_ids = self.get_target_segmentation_ids()
        if target_seg_ids and "segmentation" in pcd:
            pcd["target_mask"] = self._compute_target_mask(
                pcd["segmentation"], target_seg_ids
            )
        return pcd


PICK_TO_BASKET_DOC_STRING="""
**Task Description:**
Approach the shelf and pick up any item with the name '{product_name}', placing it into the basket attached to the Fetch robot.
The robot is spawned in close proximity to the shelf.

**Randomizations:**
- scene layout, object arrangement, wall and floor textures
- initial robot position, if `ROBOT_INIT_POSE_RANDOM_ENABLED` is enabled (True by default)

**Success Conditions:**
- any product item with the name '{product_name}' is within `TARGET_POS_THRESH` Euclidean distance of the goal position (the Fetch robot's basket).
- other items remain untouched (their positions change by no more than 0.1 m)
- the robot is static (q velocity < 0.2)
"""


# train items
@register_env('PickToBasketContNiveaEnv', max_episode_steps=200000)
class PickToBasketContNiveaEnv(PickToBasketContEnv):
    TARGET_PRODUCT_NAME = 'Nivea Body Milk'

PickToBasketContNiveaEnv.__doc__ = PICK_TO_BASKET_DOC_STRING.format(product_name='Nivea Body Milk')

@register_env('PickToBasketContStarsEnv', max_episode_steps=200000)
class PickToBasketContStarsEnv(PickToBasketContEnv):
    TARGET_PRODUCT_NAME = 'Nestle Honey Stars'
    TARGET_POS_THRESH = 0.25

PickToBasketContStarsEnv.__doc__ = PICK_TO_BASKET_DOC_STRING.format(product_name='Nestle Honey Stars')

@register_env('PickToBasketContFantaEnv', max_episode_steps=200000)
class PickToBasketContFantaEnv(PickToBasketContEnv):
    TARGET_PRODUCT_NAME = 'Fanta Sabor Naranja 2L'

PickToBasketContFantaEnv.__doc__ = PICK_TO_BASKET_DOC_STRING.format(product_name='Fanta Sabor Naranja 2L')

# unseen test items
@register_env('PickToBasketContNestleEnv', max_episode_steps=200000)
class PickToBasketContNestleEnv(PickToBasketContEnv):
    TARGET_PRODUCT_NAME = 'Nestle Fitness Chocolate Cereals'

PickToBasketContNestleEnv.__doc__ = PICK_TO_BASKET_DOC_STRING.format(product_name='Nestle Fitness Chocolate Cereals')

@register_env('PickToBasketContSlamEnv', max_episode_steps=200000)
class PickToBasketContSlamEnv(PickToBasketContEnv):
    TARGET_PRODUCT_NAME = 'SLAM luncheon meat'

PickToBasketContSlamEnv.__doc__ = PICK_TO_BASKET_DOC_STRING.format(product_name='SLAM luncheon meat')

@register_env('PickToBasketContDuffEnv', max_episode_steps=200000)
class PickToBasketContDuffEnv(PickToBasketContEnv):
    TARGET_PRODUCT_NAME = 'Duff Beer Can'

PickToBasketContDuffEnv.__doc__ = PICK_TO_BASKET_DOC_STRING.format(product_name='Duff Beer Can')

@register_env('PickToBasketContOreoEnv', max_episode_steps=200000)
class PickToBasketContOreoEnv(PickToBasketContEnv):
    TARGET_PRODUCT_NAME = 'Oreo Lemon Creme Sandwich Cookies'

PickToBasketContOreoEnv.__doc__ = PICK_TO_BASKET_DOC_STRING.format(product_name='Oreo Lemon Creme Sandwich Cookies')

@register_env('PickToBasketContMonsterEnv', max_episode_steps=200000)
class PickToBasketContMonsterEnv(PickToBasketContEnv):
    TARGET_PRODUCT_NAME = 'Monster Energy Drink'

PickToBasketContMonsterEnv.__doc__ = PICK_TO_BASKET_DOC_STRING.format(product_name='Monster Energy Drink')

@register_env('PickToBasketContVanishEnv', max_episode_steps=200000)
class PickToBasketContVanishEnv(PickToBasketContEnv):
    TARGET_PRODUCT_NAME = 'Vanish Stain Remover'

PickToBasketContVanishEnv.__doc__ = PICK_TO_BASKET_DOC_STRING.format(product_name='Vanish Stain Remover')
