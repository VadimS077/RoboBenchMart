#!/usr/bin/env python3
"""
Бенчмарк success-rate motion planning: с GraspGen-ZMQ и без (dummy OBB solver).

По умолчанию (--random-target-per-episode): после каждого reset(reconfigure=True)
читает актуальный products_df сцены (как и scene_items.csv на диске), случайно
выбирает один из --products и экземпляр на полке (предпочтительно передний ряд),
затем запускает оба пайплайна на одной и той же раскладке.

  1) GraspGen-ZMQ: pointcloud → ZMQ infer_scene → MP → basket
  2) Dummy OBB:    тот же MP-пайплайн, поза хвата по OBB

С флагом --legacy-per-product-loop сохраняется старый режим: отдельный env на
каждый продукт и выбор цели из scene_items.csv до запуска (устаревает при
нескольких под-сценах в конфиге).

Пример:
  python scripts/benchmark_graspgen.py \
      --scenes-root generated_envs \
      --products Oreo Monster Vanish \
      --num-traj 5 \
      --host localhost --port 5556 \
      --vis
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
import sapien
from tqdm import tqdm
from transforms3d.quaternions import mat2quat

from mani_skill.utils.wrappers.record import RecordEpisode
#from mani_skill.utils.wrappers.record import RecordEpisodegraspgen

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

_GRASPGEN_ROOT = Path(os.environ.get("GRASPGEN_ROOT", "")).expanduser() if os.environ.get("GRASPGEN_ROOT") else None
if _GRASPGEN_ROOT and not (_GRASPGEN_ROOT / "grasp_gen").is_dir():
    _GRASPGEN_ROOT = None
if _GRASPGEN_ROOT is None:
    _GRASPGEN_ROOT = ROOT_DIR.parent / "GraspGen"
if str(_GRASPGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GRASPGEN_ROOT))

from mani_skill.utils import common
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

from dsynth.envs import *  # noqa: F401,F403
from dsynth.robots import *  # noqa: F401,F403
from dsynth.planning.motionplanner import FetchMotionPlanningSapienSolver
from dsynth.planning.utils import (
    get_fcl_object_name,
    is_mesh_cylindrical,
)

from grasp_gen.serving.zmq_client import GraspGenClient

PRODUCT_SLUG_MAP = {
    "oreo": "OreoLemonCremeSandwichCookies",
    "monster": "MonsterEnergyDrink",
    "vanish": "VanishStainRemover",
}

# Имена product_name в products_df (как в PickToBasketCont*Env.TARGET_PRODUCT_NAME)
PRODUCT_DISPLAY_NAMES = {
    "oreo": "Oreo Lemon Creme Sandwich Cookies",
    "monster": "Monster Energy Drink",
    "vanish": "Vanish Stain Remover",
}


def _to_numpy(x):
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def _calc_basket_target_pose_world(env) -> sapien.Pose:
    base_pose = env.unwrapped.agent.base_link.pose.sp
    basket_shift = sapien.Pose(p=[0.3, 0.25, 0.14])
    return base_pose * basket_shift


def _basket_center_world(env) -> np.ndarray:
    if hasattr(env.unwrapped, "calc_target_pose"):
        return _to_numpy(env.unwrapped.calc_target_pose().sp.p).reshape(3)
    return _to_numpy(_calc_basket_target_pose_world(env).p).reshape(3)


def _episode_success_by_basket_proximity(env, product_actor, radius: float) -> Tuple[bool, float]:
    product_p = _to_numpy(product_actor.pose.sp.p).reshape(3)
    basket_p = _basket_center_world(env)
    dist = float(np.linalg.norm(product_p - basket_p))
    return dist <= radius, dist


def pick_front_row_instances(scene_dir: Path, product_slug: str) -> List[str]:
    """
    Из scene_items.csv -> список actor_name, где asset_name содержит slug
    и последний элемент имени (row_idxs) == 0.
    """
    csv_path = scene_dir / "scene_items.csv"
    if not csv_path.is_file():
        return []
    df = pd.read_csv(csv_path)
    slug = PRODUCT_SLUG_MAP.get(product_slug.lower(), product_slug)
    mask = df["actor_name"].str.contains(slug, na=False)
    candidates = df[mask]["actor_name"].tolist()
    front = [a for a in candidates if a.rsplit(":", 1)[-1] == "0"]
    return front if front else candidates


def front_row_instances_from_products_df(df: pd.DataFrame, product_slug: str) -> List[str]:
    """
    Экземпляры товара в текущей сцене (как после reset+reconfigure).
    Предпочтение переднему ряду: row_idxs == 0 в products_df.
    """
    slug = product_slug.lower().strip()
    display = PRODUCT_DISPLAY_NAMES.get(slug)
    if display is not None:
        sub = df[df["product_name"] == display]
    else:
        needle = PRODUCT_SLUG_MAP.get(slug, slug)
        sub = df[df["actor_name"].str.contains(needle, na=False)]
    if sub.empty:
        return []
    if "row_idxs" in sub.columns:
        ri = pd.to_numeric(sub["row_idxs"], errors="coerce").fillna(-1).astype(int)
        front = sub.loc[ri == 0, "actor_name"].tolist()
        if front:
            return front
    return sub["actor_name"].tolist()


def sample_random_product_and_target(
    env_u, rng: np.random.Generator, product_slugs: List[str]
) -> Optional[Tuple[str, str]]:
    """
    Равномерно по всем парам (slug из product_slugs, actor_name).
    Возвращает (product_slug, actor_name) или None, если в сцене нет ни одного кандидата.
    """
    df = getattr(env_u, "products_df", None)
    if df is None or len(df) == 0:
        return None
    pool: List[Tuple[str, str]] = []
    for ps in product_slugs:
        for an in front_row_instances_from_products_df(df, ps):
            pool.append((ps, an))
    if not pool:
        return None
    i = int(rng.integers(len(pool)))
    return pool[i]


def _resolve_selected_id(env, name_arg: str) -> Optional[int]:
    seg_map = getattr(env.unwrapped, "segmentation_id_map", None)
    if seg_map is None:
        return None
    for obj_id, obj in seg_map.items():
        try:
            if name_arg in obj.name:
                return obj_id
        except Exception:
            continue
    return None


def capture_pointcloud(env, target_actor_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Из уже построенного env: obs_mode=pointcloud, вернуть (object_xyz, scene_xyz).
    """
    obs, _ = env.reset(seed=0, options={"reconfigure": False})
    env_pc = env.unwrapped
    obs_mode = getattr(env_pc, "_obs_mode", None) or "none"
    if obs_mode != "pointcloud":
        raise RuntimeError(f"env obs_mode={obs_mode!r}, need 'pointcloud' for GraspGen capture")

    xyz = obs["pointcloud"]["xyzw"][0, ..., :3].cpu().numpy()
    seg_raw = obs["pointcloud"]["segmentation"][0].cpu().numpy()
    selected_id = _resolve_selected_id(env, target_actor_name)
    if selected_id is not None and seg_raw is not None:
        mask = (seg_raw == selected_id).reshape(-1)
    else:
        mask = np.ones(len(xyz), dtype=bool)
    return xyz[mask].astype(np.float32), xyz.astype(np.float32)


def quat_wxyz_from_rot_matrix(R: np.ndarray) -> np.ndarray:
    return np.asarray(mat2quat(R), dtype=np.float64)


def _axis_angle_rotation(axis: str, degrees: float) -> np.ndarray:
    angle = np.deg2rad(float(degrees))
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _rotation_angle_deg(R_a, R_b):
    R_rel = np.asarray(R_a, dtype=np.float64).T @ np.asarray(R_b, dtype=np.float64)
    tr = np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(tr)))


def _get_tcp_rotation_world(env):
    tcp_T = _to_numpy(env.unwrapped.agent.tcp.pose.sp.to_transformation_matrix()).reshape(4, 4)
    return tcp_T[:3, :3].astype(np.float64)


def _align_grasp_rotation_with_tcp(R_grasp, R_tcp, flip_axis="z"):
    if flip_axis == "none":
        return R_grasp
    R_flip = _axis_angle_rotation(flip_axis, 180.0)
    R_candidate = R_grasp @ R_flip
    if _rotation_angle_deg(R_tcp, R_candidate) < _rotation_angle_deg(R_tcp, R_grasp):
        return R_candidate
    return R_grasp


def grasp_mat_to_tcp_pose(mat_4x4: np.ndarray) -> sapien.Pose:
    R = mat_4x4[:3, :3]
    p = mat_4x4[:3, 3]
    return sapien.Pose(p=p.tolist(), q=quat_wxyz_from_rot_matrix(R).tolist())


@dataclass
class CollisionSceneFromPointCloud:
    obstacle_points: np.ndarray
    object_bbox_min: np.ndarray
    object_bbox_max: np.ndarray
    object_center: np.ndarray

    @staticmethod
    def from_arrays(
        object_pc: np.ndarray,
        scene_pc: np.ndarray,
        *,
        obstacle_downsample: int = 8000,
        exclude_object_from_obstacles: bool = True,
        exclude_bbox_margin: float = 0.01,
        rng_seed: int = 0,
    ) -> "CollisionSceneFromPointCloud":
        obj = np.asarray(object_pc, dtype=np.float64)
        scene = np.asarray(scene_pc, dtype=np.float64)
        bbox_min = obj.min(axis=0)
        bbox_max = obj.max(axis=0)
        center = obj.mean(axis=0)

        if exclude_object_from_obstacles:
            m = exclude_bbox_margin
            keep = ~(
                (scene[:, 0] >= bbox_min[0] - m) & (scene[:, 0] <= bbox_max[0] + m)
                & (scene[:, 1] >= bbox_min[1] - m) & (scene[:, 1] <= bbox_max[1] + m)
                & (scene[:, 2] >= bbox_min[2] - m) & (scene[:, 2] <= bbox_max[2] + m)
            )
            obs = scene[keep]
        else:
            obs = scene

        rng = np.random.default_rng(rng_seed)
        if obs.shape[0] > obstacle_downsample:
            idx = rng.integers(0, obs.shape[0], size=obstacle_downsample, endpoint=False)
            obs = obs[idx]

        return CollisionSceneFromPointCloud(
            obstacle_points=obs.astype(np.float32),
            object_bbox_min=bbox_min.astype(np.float32),
            object_bbox_max=bbox_max.astype(np.float32),
            object_center=center.astype(np.float32),
        )


def solve_graspgen(
    env,
    target_actor_name: str,
    graspgen_client: GraspGenClient,
    *,
    seed: int = 0,
    vis: bool = False,
    debug: bool = False,
    max_grasps: int = 10,
    basket_success_radius: float = 0.25,
    approach_axis: str = "z",
    grasp_rot_deg: float = -90.0,
    grasp_forward_offset: float = 0.2,
    post_grasp_backoff: float = 0.3,
    base_standoff: float = 1.1,
    skip_reset: bool = False,
    initial_obs: Optional[Any] = None,
) -> Tuple[bool, float]:
    """
    GraspGen pipeline: capture PC -> ZMQ infer_scene -> plan_grasps_from_json-style loop -> basket.
    Returns (success, dist_to_basket).

    Если skip_reset=True, reset не вызывается: нужен уже выполненный reset(reconfigure)
    и initial_obs с pointcloud (как после такого reset).
    """
    if skip_reset:
        if initial_obs is None:
            raise ValueError("solve_graspgen(..., skip_reset=True) требует initial_obs")
        obs = initial_obs
    else:
        obs, _ = env.reset(seed=seed, options={"reconfigure": True})
    env_u = env.unwrapped
    products = getattr(env_u, "actors", {}).get("products", {})
    if target_actor_name not in products:
        print(f"[GraspGen] target {target_actor_name!r} not found in products")
        return False, float("inf")

    target_actor = products[target_actor_name]
    target_center = _to_numpy(target_actor.pose.sp.p).reshape(3)

    # --- Capture pointcloud via segmentation ---
    xyz_all = obs["pointcloud"]["xyzw"][0, ..., :3].cpu().numpy().astype(np.float32)
    seg_raw = obs["pointcloud"]["segmentation"][0].cpu().numpy()
    selected_id = _resolve_selected_id(env, target_actor_name)
    if selected_id is not None and seg_raw is not None:
        mask = (seg_raw == selected_id).reshape(-1)
    else:
        mask = np.ones(len(xyz_all), dtype=bool)
    obj_pc = xyz_all[mask]
    scene_pc = xyz_all
    if len(obj_pc) < 10:
        print(f"[GraspGen] Too few object points ({len(obj_pc)})")
        return False, float("inf")
    # --- Approach direction: от робота к полке (directions_to_shelf) ---
    approach_dir = None
    robot_pos = None
    if hasattr(env_u, "directions_to_shelf") and len(env_u.directions_to_shelf) > 0:
        approach_dir = np.asarray(env_u.directions_to_shelf[0], dtype=np.float64).reshape(3)
        approach_dir[2] = 0.0
        n = np.linalg.norm(approach_dir)
        if n > 1e-6:
            approach_dir = (approach_dir / n).astype(np.float32)
        else:
            approach_dir = None
    if approach_dir is None:
        robot_pos = _to_numpy(env_u.agent.base_link.pose.sp.p).reshape(3).astype(np.float32)

    # --- ZMQ infer_scene ---
    try:
        grasps, confs, meta = graspgen_client.infer_scene(
            obj_pc,
            scene_pc,
            filter_collisions=True,
            output_topk=max_grasps,
            num_grasps=200,
            approach_direction=approach_dir,
            robot_position=robot_pos,
            approach_cos_threshold=0.3,
        )
    except Exception as e:
        print(f"[GraspGen] ZMQ error: {e}")
        return False, float("inf")

    if len(grasps) == 0:
        print("[GraspGen] No grasps returned")
        return False, float("inf")

    print(f"[GraspGen] Got {len(grasps)} grasps, conf {confs.min():.3f}..{confs.max():.3f}")

    # --- Plan and execute ---
    planner = FetchMotionPlanningSapienSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env_u.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        disable_actors_collision=False,
        joint_vel_limits=0.6,
        joint_acc_limits=0.6
    )

    collision_scene = CollisionSceneFromPointCloud.from_arrays(obj_pc, scene_pc)

    # _prepare_base_and_lift
    if hasattr(env_u, "active_shelves") and len(env_u.active_shelves) > 0:
        actor_shelf_name = env_u.active_shelves[0][0]
        shelf_pose = env_u.actors["fixtures"]["shelves"][actor_shelf_name].pose.sp
        direction_to_shelf = np.asarray(env_u.directions_to_shelf[0], dtype=np.float64)
        direction_to_shelf[2] = 0.0
        n_norm = np.linalg.norm(direction_to_shelf)
        if n_norm > 1e-6:
            n = direction_to_shelf / n_norm
            t = np.cross([0, 0, 1.0], n)
            t[2] = 0.0
            t_norm = np.linalg.norm(t)
            if t_norm > 1e-6:
                t = t / t_norm
            ref = np.asarray(shelf_pose.p, dtype=np.float64).reshape(3)
            lat = float(np.dot(target_center[:2] - ref[:2], t[:2]))
            base_p = np.asarray(env_u.agent.base_link.pose.sp.p, dtype=np.float64).reshape(3)

            view_to = target_center - base_p
            view_to[2] = 0
            if np.linalg.norm(view_to) > 1e-4:
                planner.rotate_base_z(view_to)
                
                planner.planner.update_from_simulation()

            base_p = np.asarray(env_u.agent.base_link.pose.sp.p, dtype=np.float64).reshape(3)
            lat_cur = float(np.dot(base_p[:2] - ref[:2], t[:2]))
            delta_lat = lat - lat_cur
            p_par = base_p + delta_lat * t
            p_par[2] = base_p[2]
            if abs(delta_lat) > 1e-3:
                planner.drive_base(p_par)
                planner.planner.update_from_simulation()

            planner.rotate_base_z(n)

            planner.planner.update_from_simulation()

            tcp_pose = env_u.agent.tcp.pose.sp
            lift_p = np.asarray(tcp_pose.p, dtype=np.float64).copy()
            lift_p[2] = float(target_center[2])
            lift_pose = sapien.Pose(p=lift_p.tolist(), q=np.asarray(tcp_pose.q, dtype=np.float64).tolist())
            lift_pose = lift_pose * sapien.Pose(p=[0.1, 0.0, 0.0])
            planner.static_manipulation(lift_pose, n_init_qpos=80, disable_lift_joint=False)
        
            planner.planner.update_from_simulation()
            
            base_p = env_u.agent.base_link.pose.sp
            base_z = float(np.asarray(base_p.p, dtype=np.float64).reshape(3)[2])
            goal = ref - float(base_standoff) * n + lat * t
            goal[2] = base_z
            planner.drive_base(goal)
           
            planner.planner.update_from_simulation()

    planner.open_gripper()

    grasp_rot_correction = _axis_angle_rotation("z", grasp_rot_deg)
    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_idx = axis_map[approach_axis]

    for k in range(min(max_grasps, len(grasps))):
        mat_world = grasps[k].astype(np.float64).copy()
        mat_world[:3, :3] = mat_world[:3, :3] @ grasp_rot_correction
        mat_world[:3, :3] = _align_grasp_rotation_with_tcp(
            mat_world[:3, :3], _get_tcp_rotation_world(env), "z"
        )

        R = mat_world[:3, :3]
        forward_dir = R[:, axis_idx]
        mat_world[:3, 3] += forward_dir * float(grasp_forward_offset)
        tcp_pose = grasp_mat_to_tcp_pose(mat_world)

        planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
            get_fcl_object_name(target_actor), True
        )
        res = planner.static_manipulation(tcp_pose, n_init_qpos=400, disable_lift_joint=False)
        if res == -1:
            continue

        planner.close_gripper()
        kwargs = {
            "name": get_fcl_object_name(target_actor),
            "art_name": "scene-0_ds_fetch_basket_1",
            "link_id": planner.planner.move_group_link_id,
        }
        planner.planner.planning_world.attach_object(**kwargs)
        planner.planner.update_from_simulation()

        if post_grasp_backoff > 0:
            res_b = planner.move_forward_delta(delta=-float(post_grasp_backoff))
            if res_b == -1:
                continue
            planner.planner.update_from_simulation()

        if hasattr(env_u, "calc_target_pose"):
            goal_center = env_u.calc_target_pose().sp.p
        else:
            goal_center = _calc_basket_target_pose_world(env).p
        goal_approaching = np.array([0, 0, -1.0], dtype=np.float64)
        goal_closing = -env_u.agent.base_link.pose.sp.to_transformation_matrix()[:3, 1]
        goal_pose = env_u.agent.build_grasp_pose(goal_approaching, goal_closing, goal_center)
        goal_pose = goal_pose * sapien.Pose(p=[-0.03, 0.0, -0.35])

        res_place = planner.static_manipulation(goal_pose, n_init_qpos=100, disable_lift_joint=False)
        if res_place == -1:
            continue
        planner.open_gripper()
        planner.planner.update_from_simulation()
        planner.idle_steps(t=10)

        ok, dist = _episode_success_by_basket_proximity(env, target_actor, basket_success_radius)
        return ok, dist

    ok, dist = _episode_success_by_basket_proximity(env, target_actor, basket_success_radius)
    return ok, dist


def solve_dummy(
    env,
    target_actor_name: str,
    *,
    seed: int = 0,
    vis: bool = False,
    debug: bool = False,
    basket_success_radius: float = 0.25,
    post_grasp_backoff: float = 0.3,
    base_standoff: float = 1.1,
    skip_reset: bool = False,
) -> Tuple[bool, float]:
    """
    Dummy OBB pipeline: тот же подъезд/lift/граспинг/backoff/place, что solve_graspgen,
    но поза хвата считается по OBB (is_mesh_cylindrical / compute_grasp_info_by_obb).

    При skip_reset=True окружение уже сброшено снаружи (тот же кадр, что и для GraspGen).
    """
    if not skip_reset:
        env.reset(seed=seed, options={"reconfigure": True})
    env_u = env.unwrapped
    products = getattr(env_u, "actors", {}).get("products", {})
    if target_actor_name not in products:
        print(f"[Dummy] target {target_actor_name!r} not found in products")
        return False, float("inf")

    FINGER_LENGTH = 0.03
    target_actor = products[target_actor_name]
    obb = get_actor_obb(target_actor)
    target_center = np.array(obb.primitive.transform)[:3, 3]

    planner = FetchMotionPlanningSapienSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env_u.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        disable_actors_collision=False,
        joint_vel_limits=0.6,
        joint_acc_limits=0.6,
    )

    if len(planner.planner.planning_world.check_collision()) > 0:
        print("[Dummy] Initial collision detected")
        return False, float("inf")

    def get_base_pose():
        return env_u.agent.base_link.pose

    def get_tcp_pose():
        return env_u.agent.tcp.pose

    def get_tcp_matrix():
        return get_tcp_pose().to_transformation_matrix()[0].cpu().numpy()

    def get_tcp_center():
        return get_tcp_matrix()[:3, 3]

    # --- Подъезд к полке (тот же алгоритм, что solve_graspgen) ---
    if hasattr(env_u, "active_shelves") and len(env_u.active_shelves) > 0:
        actor_shelf_name = env_u.active_shelves[0][0]
        shelf_pose = env_u.actors["fixtures"]["shelves"][actor_shelf_name].pose.sp
        direction_to_shelf = np.asarray(env_u.directions_to_shelf[0], dtype=np.float64)
        direction_to_shelf[2] = 0.0
        n_norm = np.linalg.norm(direction_to_shelf)
        if n_norm > 1e-6:
            n = direction_to_shelf / n_norm
            t = np.cross([0, 0, 1.0], n)
            t[2] = 0.0
            t_norm = np.linalg.norm(t)
            if t_norm > 1e-6:
                t = t / t_norm
            ref = np.asarray(shelf_pose.p, dtype=np.float64).reshape(3)
            lat = float(np.dot(target_center[:2] - ref[:2], t[:2]))
            base_p = np.asarray(env_u.agent.base_link.pose.sp.p, dtype=np.float64).reshape(3)

            view_to = target_center - base_p
            view_to[2] = 0
            if np.linalg.norm(view_to) > 1e-4:
                res = planner.rotate_base_z(view_to)
                if res == -1:
                    return False, float("inf")
                planner.planner.update_from_simulation()

            base_p = np.asarray(env_u.agent.base_link.pose.sp.p, dtype=np.float64).reshape(3)
            lat_cur = float(np.dot(base_p[:2] - ref[:2], t[:2]))
            delta_lat = lat - lat_cur
            p_par = base_p + delta_lat * t
            p_par[2] = base_p[2]
            if abs(delta_lat) > 1e-3:
                res = planner.drive_base(p_par)
                if res == -1:
                    return False, float("inf")
                planner.planner.update_from_simulation()

            res = planner.rotate_base_z(n)
            if res == -1:
                return False, float("inf")
            planner.planner.update_from_simulation()

            tcp_sp = env_u.agent.tcp.pose.sp
            lift_p = np.asarray(tcp_sp.p, dtype=np.float64).copy()
            lift_p[2] = float(target_center[2])
            lift_pose = sapien.Pose(p=lift_p.tolist(), q=np.asarray(tcp_sp.q, dtype=np.float64).tolist())
            lift_pose = lift_pose * sapien.Pose(p=[0.05, 0.0, 0.0]) #почему то ломаются верхние полки, если больше 0.1
            res = planner.static_manipulation(lift_pose, n_init_qpos=80, disable_lift_joint=False)
            if res == -1:
                return False, float("inf")
            planner.planner.update_from_simulation()

            base_p = np.asarray(env_u.agent.base_link.pose.sp.p, dtype=np.float64).reshape(3)
            goal = ref - float(base_standoff) * n + lat * t
            goal[2] = base_p[2]
            res = planner.drive_base(goal)
            if res == -1:
                return False, float("inf")
            planner.planner.update_from_simulation()

    planner.open_gripper()

    # --- Поза хвата: OBB (как в pick_to_cart.py) ---
    if is_mesh_cylindrical(target_actor):
        grasp_approaching = env_u.directions_to_shelf[0].copy()
        grasp_approaching[2] = 0.0
        grasp_approaching = common.np_normalize_vector(grasp_approaching)
        grasp_closing = np.cross(grasp_approaching, [0.0, 0.0, 1.0])
        grasp_center = target_center
    else:
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=get_tcp_matrix()[:3, 2],
            target_closing=get_tcp_matrix()[:3, 1],
            depth=FINGER_LENGTH,
        )
        grasp_closing = grasp_info["closing"]
        grasp_center = grasp_info["center"]
        grasp_approaching = grasp_info["approaching"]

    grasp_pose = env_u.agent.build_grasp_pose(grasp_approaching, grasp_closing, grasp_center)

    planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
        get_fcl_object_name(target_actor), True
    )
    res = planner.static_manipulation(grasp_pose, n_init_qpos=400, disable_lift_joint=False)
    if res == -1:
        print("[Dummy] grasp failed")
        return False, float("inf")

    planner.close_gripper()
    kwargs = {
        "name": get_fcl_object_name(target_actor),
        "art_name": "scene-0_ds_fetch_basket_1",
        "link_id": planner.planner.move_group_link_id,
    }
    planner.planner.planning_world.attach_object(**kwargs)
    planner.planner.update_from_simulation()

    # --- Lift ---
    lift_pose = grasp_pose * sapien.Pose([0.06, 0.0, 0.0])
    res = planner.static_manipulation(lift_pose, n_init_qpos=200, disable_lift_joint=False)
    if res == -1:
        return False, float("inf")
    planner.planner.update_from_simulation()

    # --- Backoff ---
    if post_grasp_backoff > 0:
        res = planner.move_forward_delta(delta=-float(post_grasp_backoff))
        if res == -1:
            return False, float("inf")
        planner.planner.update_from_simulation()

    # --- Place to basket ---
    if hasattr(env_u, "calc_target_pose"):
        goal_center = env_u.calc_target_pose().sp.p
    else:
        goal_center = _calc_basket_target_pose_world(env).p
    goal_approaching = np.array([0, 0, -1.0], dtype=np.float64)
    goal_closing = -env_u.agent.base_link.pose.sp.to_transformation_matrix()[:3, 1]
    goal_pose = env_u.agent.build_grasp_pose(goal_approaching, goal_closing, goal_center)
    goal_pose = goal_pose * sapien.Pose(p=[-0.03, 0.0, -0.35])

    res = planner.static_manipulation(goal_pose, n_init_qpos=100, disable_lift_joint=False)
    if res == -1:
        return False, float("inf")

    planner.open_gripper()
    planner.planner.update_from_simulation()
    planner.idle_steps(t=10)

    ok, dist = _episode_success_by_basket_proximity(env, target_actor, basket_success_radius)
    return ok, dist


def _make_env(
    args,
    scene_dir,
    sensor_configs,
    obs_mode: str = "pointcloud",
    *,
    env_id: str = "PickToBasketContEnv",
):
    """Создать gym env. env_id по умолчанию — базовый PickToBasketContEnv (любой товар в сцене)."""
    return gym.make(
        env_id,
        robot_uids=args.robot_uids,
        config_dir_path=str(scene_dir),
        num_envs=1,
        control_mode="pd_joint_pos",
        obs_mode=obs_mode,
        render_mode="human" if args.vis else "rgb_array",
        enable_shadow=True,
        viewer_camera_configs=dict(shader_pack=args.shader, width=320, height=240),
        human_render_camera_configs=dict(shader_pack=args.shader, width=320, height=240),
        sensor_configs=sensor_configs,
        parallel_in_single_scene=False,
        sim_backend="auto",
    )


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark GraspGen vs dummy MP")
    p.add_argument("--scenes-root", type=Path, default=ROOT_DIR / "generated_envs", help="Dir with ds_small_scene_* dirs")
    p.add_argument("--products", nargs="+", default=["oreo", "monster", "vanish"])
    p.add_argument("--num-traj", type=int, default=5,
                   help="Число эпизодов на сцену (random-режим) или на (сцену, продукт) в legacy")
    p.add_argument("--legacy-per-product-loop", action="store_true",
                   help="Старый режим: отдельный env на каждый продукт, цели из scene_items.csv до reset")
    p.add_argument("--env-id", type=str, default="DarkstoreContinuousBaseEnv")
    p.add_argument("--robot-uids", type=str, default="ds_fetch_basket")
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--vis", action="store_true")
    p.add_argument("--basket-success-radius", type=float, default=0.25)
    p.add_argument("--max-grasps", type=int, default=10)
    p.add_argument("--shader", type=str, default="default")
    p.add_argument("--skip-graspgen", action="store_true", help="Skip GraspGen runs (only dummy)")
    p.add_argument("--skip-dummy", action="store_true", help="Skip dummy runs (only GraspGen)")
    p.add_argument("--debug", action="store_true", help="Debug mode")
    p.add_argument("--save-video", action="store_true", help="Save episode videos")
    p.add_argument("--save-traj", action="store_true", help="Save trajectory .h5 files")
    p.add_argument("--only-count-success", action="store_true",
                    help="Save only successful trajectories/videos")
    p.add_argument("--video-fps", type=int, default=30, help="FPS for saved videos")
    return p.parse_args()


def _wrap_record(env, record_dir: str, traj_name: str, save_video: bool, video_fps: int):
    """Wrap env with RecordEpisode for trajectory/video saving."""
    env.sensor_configs = dict(width=124, height=124, shader_pack="minimal")
    env.human_render_camera_configs = dict(width=124, height=124, shader_pack="minimal")
    env.viewer_camera_configs = dict(width=124, height=124, shader_pack="minimal")
    return RecordEpisode(
        env,
        output_dir=record_dir,
        trajectory_name=traj_name,
        save_video=save_video,
        source_type="motionplanning",
        source_desc="benchmark_graspgen trajectory",
        video_fps=video_fps,
        record_reward=False,
        save_on_reset=False,
    )


def _flush_episode(env, success: bool, *, save_video: bool, only_success: bool):
    """Flush trajectory/video after one episode. Saves if allowed."""
    save = success or not only_success
    #env.flush_trajectory(save=save)
    if save_video:
        env.flush_video(save=save)


def _print_results_summary(results: Dict[str, Any]) -> None:
    print(f"\n\n{'='*80}")
    print("  BENCHMARK RESULTS")
    print(f"{'='*80}")
    header = f"{'Scene':<25} {'Product':<14} {'GraspGen SR':>12} {'Dummy SR':>12} {'n':>6}"
    print(header)
    print("-" * len(header))

    all_gg: List[float] = []
    all_dm: List[float] = []

    for scene_name, payload in results.items():
        if isinstance(payload, dict) and payload.get("mode") == "random_after_reset":
            ov = payload.get("overall", {})
            n = int(ov.get("n_episodes", 0))
            gg = ov.get("with", float("nan"))
            dm = ov.get("without", float("nan"))
            gg_str = f"{gg:.3f}" if n > 0 and not np.isnan(gg) else "N/A"
            dm_str = f"{dm:.3f}" if n > 0 and not np.isnan(dm) else "N/A"
            print(f"{scene_name:<25} {'(overall)':<14} {gg_str:>12} {dm_str:>12} {n:>6}")
            if n > 0 and not np.isnan(gg):
                all_gg.append(float(gg))
            if n > 0 and not np.isnan(dm):
                all_dm.append(float(dm))
            for prod, pr in payload.get("by_product", {}).items():
                n_p = int(pr.get("n_episodes", 0))
                if n_p == 0:
                    continue
                ggp = pr.get("with", float("nan"))
                dmp = pr.get("without", float("nan"))
                print(
                    f"{scene_name:<25} {prod:<14} "
                    f"{(f'{ggp:.3f}' if not np.isnan(ggp) else 'N/A'):>12} "
                    f"{(f'{dmp:.3f}' if not np.isnan(dmp) else 'N/A'):>12} {n_p:>6}"
                )
        else:
            for prod, rates in payload.items():
                if not isinstance(rates, dict):
                    continue
                gg = rates.get("with", float("nan"))
                dm = rates.get("without", float("nan"))
                if rates.get("skipped"):
                    continue
                gg_str = f"{gg:.3f}" if not np.isnan(gg) else "N/A"
                dm_str = f"{dm:.3f}" if not np.isnan(dm) else "N/A"
                print(f"{scene_name:<25} {prod:<14} {gg_str:>12} {dm_str:>12} {'':>6}")
                if not np.isnan(gg):
                    all_gg.append(gg)
                if not np.isnan(dm):
                    all_dm.append(dm)

    print("-" * len(header))
    avg_gg = f"{np.mean(all_gg):.3f}" if all_gg else "N/A"
    avg_dm = f"{np.mean(all_dm):.3f}" if all_dm else "N/A"
    print(f"{'AVERAGE':<25} {'':14} {avg_gg:>12} {avg_dm:>12} {'':>6}")
    print(f"{'='*80}")


def main():
    args = parse_args()
    scenes_root = args.scenes_root.resolve()
    scene_dirs = sorted([d for d in scenes_root.iterdir() if d.is_dir() and (d / "scene_items.csv").is_file()])

    if not scene_dirs:
        print(f"No scene dirs with scene_items.csv found in {scenes_root}")
        return

    print(f"Scenes: {[d.name for d in scene_dirs]}")
    print(f"Product pool: {args.products}")
    print(f"Episodes per scene: {args.num_traj}")
    if args.legacy_per_product_loop:
        print("Mode: legacy (per-product env + scene_items.csv before reset)")
    else:
        print("Mode: random product & instance after each reset(reconfigure) from products_df")

    record = args.save_video or args.save_traj

    graspgen_client = None
    if not args.skip_graspgen:
        graspgen_client = GraspGenClient(host=args.host, port=args.port)
        print(f"GraspGen server: {graspgen_client.server_metadata}")

    sensor_configs = dict(width=1024, height=1024, shader_pack="minimal")
    results: Dict[str, Any] = {}

    if not args.legacy_per_product_loop:
        # ---------- Случайный продукт и экземпляр после reset ----------
        for scene_dir in scene_dirs:
            scene_name = scene_dir.name
            record_dir = str(scene_dir / "demos" / "benchmark")

            by_prod_gg: Dict[str, List[bool]] = {p: [] for p in args.products}
            by_prod_dm: Dict[str, List[bool]] = {p: [] for p in args.products}
            overall_gg: List[bool] = []
            overall_dm: List[bool] = []

            env_gg = None
            if not args.skip_graspgen and graspgen_client is not None:
                env_gg = _make_env(args, scene_dir, sensor_configs, obs_mode="pointcloud", env_id="PickToBasketContEnv")
                if record:
                    env_gg = _wrap_record(
                        env_gg, record_dir, traj_name="graspgen_random",
                        save_video=args.save_video, video_fps=args.video_fps,
                    )

            env_dm = None
            if not args.skip_dummy:
                env_dm = _make_env(args, scene_dir, sensor_configs, obs_mode="pointcloud", env_id="PickToBasketContEnv")
                if record:
                    env_dm = _wrap_record(
                        env_dm, record_dir, traj_name="dummy_random",
                        save_video=args.save_video, video_fps=args.video_fps,
                    )

            pbar = tqdm(range(args.num_traj), desc=f"[Bench] {scene_name}")
            for seed_idx in pbar:
                sampled_gg: Optional[Tuple[str, str]] = None

                if env_gg is not None:
                    obs_gg, _ = env_gg.reset(seed=seed_idx, options={"reconfigure": True})
                    sampled_gg = sample_random_product_and_target(
                        env_gg.unwrapped, np.random.default_rng(int(seed_idx)), args.products
                    )
                    if sampled_gg is None:
                        print(f"\n[{scene_name}] seed={seed_idx}: нет кандидатов (GraspGen) среди {args.products}")
                        if record:
                            _flush_episode(
                                env_gg, False,
                                save_video=args.save_video,
                                only_success=args.only_count_success,
                            )
                    else:
                        product_slug, target_actor = sampled_gg
                        print(f"\n--- seed={seed_idx} product={product_slug} target={target_actor} ---")
                        try:
                            ok_gg, dist_gg = solve_graspgen(
                                env_gg,
                                target_actor,
                                graspgen_client,
                                seed=seed_idx,
                                debug=args.debug,
                                vis=args.vis,
                                max_grasps=args.max_grasps,
                                basket_success_radius=args.basket_success_radius,
                                skip_reset=True,
                                initial_obs=obs_gg,
                            )
                        except Exception as e:
                            print(f"[GraspGen] Exception: {e}")
                            ok_gg, dist_gg = False, float("inf")
                        overall_gg.append(ok_gg)
                        by_prod_gg[product_slug].append(ok_gg)
                        print(f"  GraspGen: success={ok_gg}, dist={dist_gg:.4f}")
                        if record:
                            _flush_episode(
                                env_gg, ok_gg,
                                save_video=args.save_video,
                                only_success=args.only_count_success,
                            )

                if env_dm is not None:
                    obs_dm, _ = env_dm.reset(seed=seed_idx, options={"reconfigure": True})
                    rng_ep_dm = np.random.default_rng(int(seed_idx))
                    sampled_dm = sample_random_product_and_target(
                        env_dm.unwrapped, rng_ep_dm, args.products
                    )
                    if sampled_dm is None:
                        print(f"\n[{scene_name}] seed={seed_idx}: нет кандидатов (Dummy) среди {args.products}")
                        if record:
                            _flush_episode(
                                env_dm, False,
                                save_video=args.save_video,
                                only_success=args.only_count_success,
                            )
                    else:
                        use_slug, use_actor = sampled_dm
                        if sampled_gg is not None and sampled_dm != sampled_gg:
                            print(
                                f"  [warn] цель Dummy != GraspGen: {sampled_dm} vs {sampled_gg} "
                                f"(проверьте детерминизм reconfigure по seed)"
                            )
                        print(f"\n--- [Dummy] seed={seed_idx} product={use_slug} target={use_actor} ---")
                        try:
                            ok_dm, dist_dm = solve_dummy(
                                env_dm,
                                use_actor,
                                debug=args.debug,
                                seed=seed_idx,
                                vis=args.vis,
                                basket_success_radius=args.basket_success_radius,
                                skip_reset=True,
                            )
                        except Exception as e:
                            print(f"[Dummy] Exception: {e}")
                            ok_dm, dist_dm = False, float("inf")
                        overall_dm.append(ok_dm)
                        by_prod_dm[use_slug].append(ok_dm)
                        print(f"  Dummy:    success={ok_dm}, dist={dist_dm:.4f}")
                        if record:
                            _flush_episode(
                                env_dm, ok_dm,
                                save_video=args.save_video,
                                only_success=args.only_count_success,
                            )

                pbar.set_postfix(
                    gg=f"{np.mean(overall_gg):.2f}" if overall_gg else "—",
                    dm=f"{np.mean(overall_dm):.2f}" if overall_dm else "—",
                )

            if env_gg is not None:
                env_gg.close()
            if env_dm is not None:
                env_dm.close()

            n_gg = len(overall_gg)
            n_dm = len(overall_dm)
            by_product_out: Dict[str, Any] = {}
            for p in args.products:
                ng = len(by_prod_gg[p])
                nd = len(by_prod_dm[p])
                by_product_out[p] = {
                    "with": float(np.mean(by_prod_gg[p])) if ng else float("nan"),
                    "without": float(np.mean(by_prod_dm[p])) if nd else float("nan"),
                    "n_episodes_graspgen": ng,
                    "n_episodes_dummy": nd,
                    "n_episodes": max(ng, nd),
                }

            results[scene_name] = {
                "mode": "random_after_reset",
                "overall": {
                    "with": float(np.mean(overall_gg)) if n_gg else float("nan"),
                    "without": float(np.mean(overall_dm)) if n_dm else float("nan"),
                    "n_episodes": max(n_gg, n_dm),
                },
                "by_product": by_product_out,
            }
            print(
                f"\n  [{scene_name}] overall GraspGen SR="
                f"{results[scene_name]['overall']['with']:.3f}  Dummy SR="
                f"{results[scene_name]['overall']['without']:.3f}  (n={max(n_gg, n_dm)})"
            )

    else:
        # ---------- Legacy: цикл по продуктам + CSV на диске ----------
        for scene_dir in scene_dirs:
            scene_name = scene_dir.name
            results[scene_name] = {}

            for product_slug in args.products:
                front_instances = pick_front_row_instances(scene_dir, product_slug)
                if not front_instances:
                    print(f"  [{scene_name}] No front-row instances for '{product_slug}', skipping")
                    results[scene_name][product_slug] = {"with": 0.0, "without": 0.0, "skipped": True}
                    continue

                print(f"\n{'='*70}")
                print(f"  Scene={scene_name}  Product={product_slug}  FrontRow={len(front_instances)} instances")
                print(f"{'='*70}")

                targets_per_seed = []
                for seed_idx in range(args.num_traj):
                    rng = np.random.default_rng(seed_idx)
                    targets_per_seed.append(rng.choice(front_instances))

                graspgen_successes: List[bool] = []
                dummy_successes: List[bool] = []
                record_dir = str(scene_dir / "demos" / "benchmark")
                env_id = f"PickToBasketCont{product_slug.capitalize()}Env"

                if not args.skip_graspgen and graspgen_client is not None:
                    env_gg = _make_env(args, scene_dir, sensor_configs, obs_mode="pointcloud", env_id=env_id)
                    if record:
                        env_gg = _wrap_record(
                            env_gg, record_dir,
                            traj_name=f"graspgen_{product_slug}",
                            save_video=args.save_video,
                            video_fps=args.video_fps,
                        )

                    pbar_gg = tqdm(range(args.num_traj), desc=f"[GraspGen] {scene_name}/{product_slug}")
                    for seed_idx in pbar_gg:
                        target_instance = targets_per_seed[seed_idx]
                        print(f"\n--- [GraspGen] seed={seed_idx}, target={target_instance} ---")

                        try:
                            ok_gg, dist_gg = solve_graspgen(
                                env_gg,
                                target_instance,
                                graspgen_client,
                                seed=seed_idx,
                                debug=args.debug,
                                vis=args.vis,
                                max_grasps=args.max_grasps,
                                basket_success_radius=args.basket_success_radius,
                            )
                        except Exception as e:
                            print(f"[GraspGen] Exception: {e}")
                            ok_gg, dist_gg = False, float("inf")

                        graspgen_successes.append(ok_gg)
                        print(f"  GraspGen: success={ok_gg}, dist={dist_gg:.4f}")

                        if record:
                            _flush_episode(
                                env_gg, ok_gg,
                                save_video=args.save_video,
                                only_success=args.only_count_success,
                            )
                        pbar_gg.set_postfix(sr=f"{np.mean(graspgen_successes):.3f}")

                    env_gg.close()

                if not args.skip_dummy:
                    env_dm = _make_env(args, scene_dir, sensor_configs, obs_mode="pointcloud", env_id=env_id)
                    if record:
                        env_dm = _wrap_record(
                            env_dm, record_dir,
                            traj_name=f"dummy_{product_slug}",
                            save_video=args.save_video,
                            video_fps=args.video_fps,
                        )

                    pbar_dm = tqdm(range(args.num_traj), desc=f"[Dummy]    {scene_name}/{product_slug}")
                    for seed_idx in pbar_dm:
                        target_instance = targets_per_seed[seed_idx]
                        print(f"\n--- [Dummy] seed={seed_idx}, target={target_instance} ---")

                        try:
                            ok_dm, dist_dm = solve_dummy(
                                env_dm,
                                target_instance,
                                debug=args.debug,
                                seed=seed_idx,
                                vis=args.vis,
                                basket_success_radius=args.basket_success_radius,
                            )
                        except Exception as e:
                            print(f"[Dummy] Exception: {e}")
                            ok_dm, dist_dm = False, float("inf")

                        dummy_successes.append(ok_dm)
                        print(f"  Dummy:    success={ok_dm}, dist={dist_dm:.4f}")

                        if record:
                            _flush_episode(
                                env_dm, ok_dm,
                                save_video=args.save_video,
                                only_success=args.only_count_success,
                            )
                        pbar_dm.set_postfix(sr=f"{np.mean(dummy_successes):.3f}")

                    env_dm.close()

                rate_gg = float(np.mean(graspgen_successes)) if graspgen_successes else float("nan")
                rate_dm = float(np.mean(dummy_successes)) if dummy_successes else float("nan")
                results[scene_name][product_slug] = {"with": rate_gg, "without": rate_dm}
                print(f"\n  [{scene_name}/{product_slug}] GraspGen SR={rate_gg:.3f}  Dummy SR={rate_dm:.3f}")

    if graspgen_client is not None:
        graspgen_client.close()

    _print_results_summary(results)

    out_path = scenes_root / "benchmark_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
