#!/usr/bin/env python3
"""
Load GraspGen/ManiSkill-style JSON point clouds (cereals.json) and a set of grasp poses
(cereals_top5_grasps.json), then run Panda motion planning against a collision scene
constructed from the JSON data.

Important assumptions / conventions:
- `cereals.json` uses the same keys as `darkstore-synthesizer/scripts/save_view.py`:
  - object_info.pc, object_info.pc_color
  - scene_info.pc_color, scene_info.img_color
- Each grasp pose in `cereals_top5_grasps.json` is a 4x4 transform matrix in the same
  coordinate frame as the point clouds.
- The 4x4 transform is interpreted as the target TCP pose:
  - translation = tcp position
  - rotation matrix = tcp frame orientation (local +Z is used as approach direction)
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from mani_skill.utils.wrappers.record import RecordEpisode
from tqdm import tqdm
from transforms3d.quaternions import mat2quat


import sys

import sapien  # type: ignore

# Make sure imports resolve when running the script from this folder.
ROOT_DIR = Path(__file__).resolve().parents[1]  # darkstore-synthesizer/
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure dsynth env registrations are executed.
from dsynth.envs import *  # noqa: F401,F403

from dsynth.planning.motionplanner import FetchMotionPlanningSapienSolver, PandaArmMotionPlanningSolverV2
from dsynth.planning.utils import get_fcl_object_name
from dsynth.robots import *


def _to_numpy(x):
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def quat_wxyz_from_rot_matrix(R: np.ndarray) -> np.ndarray:
    # transforms3d.mat2quat returns [w, x, y, z]
    q_wxyz = mat2quat(R)
    return np.asarray(q_wxyz, dtype=np.float64)


def _axis_angle_rotation(axis: str, degrees: float) -> np.ndarray:
    angle = np.deg2rad(float(degrees))
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.asarray(
            [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
            dtype=np.float64,
        )
    if axis == "y":
        return np.asarray(
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
            dtype=np.float64,
        )
    if axis == "z":
        return np.asarray(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    raise ValueError(f"Unknown axis='{axis}'. Expected one of: x, y, z")


def _rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_rel = np.asarray(R_a, dtype=np.float64).T @ np.asarray(R_b, dtype=np.float64)
    tr = np.clip((np.trace(R_rel) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(tr)))


def _get_tcp_rotation_world(env) -> np.ndarray:
    tcp_pose_sp = env.unwrapped.agent.tcp.pose.sp
    tcp_T = _to_numpy(tcp_pose_sp.to_transformation_matrix()).reshape(4, 4)
    return np.asarray(tcp_T[:3, :3], dtype=np.float64)


def _calc_basket_target_pose_world(env) -> sapien.Pose:
    """
    Pick-to-basket fallback for envs without `calc_target_pose()`.
    Mirrors `pick_to_basket.py`: base_link * [0.3, 0.25, 0.14].
    """
    base_pose = env.unwrapped.agent.base_link.pose.sp
    basket_shift = sapien.Pose(p=[0.3, 0.25, 0.14])
    return base_pose * basket_shift


def _basket_center_world(env) -> np.ndarray:
    if hasattr(env.unwrapped, "calc_target_pose"):
        return _to_numpy(env.unwrapped.calc_target_pose().sp.p).reshape(3)
    return _to_numpy(_calc_basket_target_pose_world(env).p).reshape(3)


def _episode_success_by_basket_proximity(
    env,
    product_actor,
    radius: float,
) -> tuple[bool, float]:
    """Success iff distance(product pose, basket target) <= radius."""
    product_p = _to_numpy(product_actor.pose.sp.p).reshape(3)
    basket_p = _basket_center_world(env)
    dist = float(np.linalg.norm(product_p - basket_p))
    return dist <= radius, dist


def _align_grasp_rotation_with_tcp(
    R_grasp: np.ndarray,
    R_tcp: np.ndarray,
    flip_axis: str,
) -> np.ndarray:
    if flip_axis == "none":
        return R_grasp
    R_flip = _axis_angle_rotation(flip_axis, 180.0)
    R_candidate = R_grasp @ R_flip
    if _rotation_angle_deg(R_tcp, R_candidate) < _rotation_angle_deg(R_tcp, R_grasp):
        return R_candidate
    return R_grasp


def load_cereals_scene(scene_json_path: Path):
    with scene_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    object_pc = np.asarray(data["object_info"]["pc"], dtype=np.float32)  # (No, 3)
    # In our canonical format `scene_info.pc_color` stores point coordinates.
    scene_pc = np.asarray(data["scene_info"]["pc_color"], dtype=np.float32)  # (Ns, 3)

    return object_pc, scene_pc


def load_topk_grasps(topk_grasps_json_path: Path):
    with topk_grasps_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    poses = data["grasp_poses_obj_frame_topk"]
    # shape: (K, 4, 4)
    poses = np.asarray(poses, dtype=np.float64)
    return poses


@dataclass
class CollisionSceneFromPointCloud:
    obstacle_points: np.ndarray  # (N, 3) in simulator world frame
    object_bbox_min: np.ndarray  # (3,)
    object_bbox_max: np.ndarray  # (3,)
    object_center: np.ndarray  # (3,)

    @staticmethod
    def from_cereals_json(
        cereals_json_path: Path,
        *,
        R_obj_to_world: np.ndarray,
        t_obj_to_world: np.ndarray,
        pointclouds_already_in_world: bool = False,
        obstacle_downsample: int,
        exclude_object_from_obstacles: bool = True,
        exclude_bbox_margin: float = 0.01,
        rng_seed: int = 0,
    ) -> "CollisionSceneFromPointCloud":
        object_pc, scene_pc = load_cereals_scene(cereals_json_path)

        if pointclouds_already_in_world:
            object_pc_world = np.asarray(object_pc, dtype=np.float64)
            scene_pc_world = np.asarray(scene_pc, dtype=np.float64)
        else:
            R_obj_to_world = np.asarray(R_obj_to_world, dtype=np.float64).reshape(3, 3)
            t_obj_to_world = np.asarray(t_obj_to_world, dtype=np.float64).reshape(3)

            # Transform object/scene point clouds from object-frame to world-frame.
            # Points are stored as row-vectors => x_world = x_obj @ R^T + t
            object_pc_world = object_pc @ R_obj_to_world.T + t_obj_to_world.reshape(1, 3)
            scene_pc_world = scene_pc @ R_obj_to_world.T + t_obj_to_world.reshape(1, 3)

        obj_bbox_min = object_pc_world.min(axis=0)
        obj_bbox_max = object_pc_world.max(axis=0)
        obj_center = object_pc_world.mean(axis=0)

        if exclude_object_from_obstacles:
            # Keep everything outside the (expanded) object AABB.
            keep_mask = ~(
                (scene_pc_world[:, 0] >= (obj_bbox_min[0] - exclude_bbox_margin))
                & (scene_pc_world[:, 0] <= (obj_bbox_max[0] + exclude_bbox_margin))
                & (scene_pc_world[:, 1] >= (obj_bbox_min[1] - exclude_bbox_margin))
                & (scene_pc_world[:, 1] <= (obj_bbox_max[1] + exclude_bbox_margin))
                & (scene_pc_world[:, 2] >= (obj_bbox_min[2] - exclude_bbox_margin))
                & (scene_pc_world[:, 2] <= (obj_bbox_max[2] + exclude_bbox_margin))
            )
            obstacle_points = scene_pc_world[keep_mask]
        else:
            obstacle_points = scene_pc_world

        if obstacle_points.shape[0] == 0:
            raise RuntimeError(
                "Obstacle point cloud is empty after filtering. "
                "Try disabling exclude_object_from_obstacles or increasing margins."
            )

        rng = np.random.default_rng(rng_seed)
        if obstacle_points.shape[0] > obstacle_downsample:
            # Sample with replacement to avoid huge memory overhead of permutations.
            idx = rng.integers(0, obstacle_points.shape[0], size=obstacle_downsample, endpoint=False)
            obstacle_points = obstacle_points[idx]

        return CollisionSceneFromPointCloud(
            obstacle_points=obstacle_points.astype(np.float32),
            object_bbox_min=obj_bbox_min.astype(np.float32),
            object_bbox_max=obj_bbox_max.astype(np.float32),
            object_center=obj_center.astype(np.float32),
        )


def grasp_mat_to_tcp_pose(mat_4x4: np.ndarray) -> sapien.Pose:
    # mat_4x4: (4, 4)
    R = mat_4x4[:3, :3]
    p = mat_4x4[:3, 3]
    q_wxyz = quat_wxyz_from_rot_matrix(R)
    return sapien.Pose(p=p.tolist(), q=q_wxyz.tolist())


def set_actor_rgba(actor, rgba):
    """Best-effort visual highlight for DSynth product actors."""
    try:
        if actor is None or not hasattr(actor, "_objs") or len(actor._objs) == 0:
            return False
        rgba = [float(x) for x in rgba]
        render_comp = actor._objs[0].find_component_by_type(sapien.pysapien.render.RenderBodyComponent)
        if render_comp is None:
            return False
        # Render shapes may share the same material object(s)
        for rs in getattr(render_comp, "render_shapes", []):
            if hasattr(rs, "material") and hasattr(rs.material, "base_color"):
                rs.material.base_color = rgba
        return True
    except Exception:
        return False


# --- Предыдущая версия _prepare_base_and_lift (оставлена для справки) ---
# def _prepare_base_and_lift(
#     env,
#     planner: FetchMotionPlanningSapienSolver,
#     target_center_world: np.ndarray,
#     base_standoff: float,
#     lift_offset_local_z: float,
# ) -> int:
#     env_unwrapped = env.unwrapped
#     if not hasattr(env_unwrapped, "active_shelves") or len(env_unwrapped.active_shelves) == 0:
#         print("No active_shelves found. Skip base pre-positioning.")
#         return 0
#
#     actor_shelf_name = env_unwrapped.active_shelves[0][0]
#     shelf_pose = env_unwrapped.actors["fixtures"]["shelves"][actor_shelf_name].pose.sp
#     direction_to_shelf = np.asarray(env_unwrapped.directions_to_shelf[0], dtype=np.float64)
#     origin = shelf_pose.p - float(base_standoff) * direction_to_shelf
#
#     base_pose = env_unwrapped.agent.base_link.pose.sp
#     view_to_target = np.asarray(target_center_world, dtype=np.float64) - np.asarray(base_pose.p, dtype=np.float64)
#     view_to_target[2] = 0.0
#
#     tcp_pose = env_unwrapped.agent.tcp.pose.sp
#     lift_ee_pos = np.asarray(tcp_pose.p, dtype=np.float64).copy()
#     lift_ee_pos[2] = float(target_center_world[2])
#     lift_ee_pose = sapien.Pose(p=lift_ee_pos.tolist(), q=np.asarray(tcp_pose.q, dtype=np.float64).tolist())
#     lift_ee_pose = lift_ee_pose * sapien.Pose(p=[0.0, 0.0, float(lift_offset_local_z)])
#
#     res = planner.static_manipulation(lift_ee_pose, n_init_qpos=80, disable_lift_joint=False)
#     if res == -1:
#         return -1
#     res = planner.drive_base(origin)
#     if res == -1:
#         print("drive")
#         return -1
#
#     planner.planner.update_from_simulation()
#     res = planner.rotate_base_z(view_to_target)
#     if res == -1:
#         print("rotate fail")
#         return -1
#     planner.planner.update_from_simulation()
#     return 0


def _prepare_base_and_lift(
    env,
    planner: FetchMotionPlanningSapienSolver,
    target_center_world: np.ndarray,
    base_standoff: float,
    lift_offset_local_z: float,
) -> int:
    """
    Предпозиционирование базы и подъём руки перед grasp planning.

    Порядок:
    1) Повернуть базу в сторону товара (горизонтальный вектор base → target).
    2) Проехать *вдоль фронта полки* (касательная t перпендикулярна directions_to_shelf в плоскости XY),
       чтобы выровняться по «полке» с товаром (как правый/левый угол), не подъезжая лицом к полке —
       смещаем только компоненту вдоль t.
    3) Повернуть базу «прямо» к полке (ориентация по directions_to_shelf).
    4) Поднять TCP на высоту товара (static_manipulation).
    5) Подъехать вперёд к точке напротив товара: ref - standoff*n + lateral*t (как раньше origin, но со сдвигом по полке).
    """
    env_unwrapped = env.unwrapped
    if not hasattr(env_unwrapped, "active_shelves") or len(env_unwrapped.active_shelves) == 0:
        print("No active_shelves found. Skip base pre-positioning.")
        return 0

    target_center_world = np.asarray(target_center_world, dtype=np.float64).reshape(3)
    actor_shelf_name = env_unwrapped.active_shelves[0][0]
    shelf_pose = env_unwrapped.actors["fixtures"]["shelves"][actor_shelf_name].pose.sp
    direction_to_shelf = np.asarray(env_unwrapped.directions_to_shelf[0], dtype=np.float64).reshape(3)
    direction_to_shelf[2] = 0.0
    n_norm = np.linalg.norm(direction_to_shelf)
    if n_norm < 1e-6:
        print("directions_to_shelf degenerate; skip base pre-positioning.")
        return 0
    n = direction_to_shelf / n_norm
    # Единичная касательная к фронту полки в плоскости пола (вдоль полки, «вправо/влево»).
    t = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float64), n)
    t[2] = 0.0
    t_norm = np.linalg.norm(t)
    if t_norm < 1e-6:
        print("Shelf tangent degenerate; skip base pre-positioning.")
        return 0
    t = t / t_norm

    ref = np.asarray(shelf_pose.p, dtype=np.float64).reshape(3)
    # Латеральная координата товара вдоль полки (скаляр вдоль t).
    lat = float(np.dot(target_center_world[:2] - ref[:2], t[:2]))

    base_pose = env_unwrapped.agent.base_link.pose.sp
    base_p = np.asarray(base_pose.p, dtype=np.float64).reshape(3)

    # 1) Повернуться к товару.
    view_to_target = target_center_world - base_p
    view_to_target[2] = 0.0
    if np.linalg.norm(view_to_target) > 1e-4:
        res = planner.rotate_base_z(view_to_target)
        if res == -1:
            print("rotate toward product failed")
            return -1
        planner.planner.update_from_simulation()

    base_pose = env_unwrapped.agent.base_link.pose.sp
    base_p = np.asarray(base_pose.p, dtype=np.float64).reshape(3)
    lat_cur = float(np.dot(base_p[:2] - ref[:2], t[:2]))

    # 2) Съехать вдоль полки к той же «полосе», что и товар (без сближения с полкой по нормали).
    delta_lat = lat - lat_cur
    p_parallel = base_p + delta_lat * t
    p_parallel[2] = base_p[2]
    if abs(delta_lat) > 1e-3:
        res = planner.drive_base(p_parallel)
        if res == -1:
            print("parallel drive along shelf failed")
            return -1
        planner.planner.update_from_simulation()

    # 3) Встать «прямо» к полке.
    face_shelf = n.copy()
    face_shelf[2] = 0.0
    if np.linalg.norm(face_shelf) > 1e-4:
        res = planner.rotate_base_z(face_shelf)
        if res == -1:
            print("rotate to face shelf failed")
            return -1
        planner.planner.update_from_simulation()

    # 4) Поднять руку на уровень товара.
    tcp_pose = env_unwrapped.agent.tcp.pose.sp
    lift_ee_pos = np.asarray(tcp_pose.p, dtype=np.float64).copy()
    lift_ee_pos[2] = float(target_center_world[2])
    lift_ee_pose = sapien.Pose(p=lift_ee_pos.tolist(), q=np.asarray(tcp_pose.q, dtype=np.float64).tolist())
    lift_ee_pose = lift_ee_pose * sapien.Pose(p=[0.0, 0.0, float(lift_offset_local_z)])

    res = planner.static_manipulation(lift_ee_pose, n_init_qpos=80, disable_lift_joint=False)
    if res == -1:
        return -1
    planner.planner.update_from_simulation()

    # 5) Подъехать к полке напротив товара (standoff по нормали n + сдвиг lat вдоль полки).
    base_pose = env_unwrapped.agent.base_link.pose.sp
    base_z = float(np.asarray(base_pose.p, dtype=np.float64).reshape(3)[2])
    goal = ref - float(base_standoff) * n + lat * t
    goal[2] = base_z

    res = planner.drive_base(goal)
    if res == -1:
        print("final approach to shelf failed")
        return -1
    planner.planner.update_from_simulation()
    return 0


def plan_grasps_from_json(
    *,
    env,
    planner: FetchMotionPlanningSapienSolver,
    collision_scene: CollisionSceneFromPointCloud,
    grasp_poses_obj_frame: np.ndarray,
    T_obj_to_world: np.ndarray,
    target_actor=None,
    grasps_already_in_world: bool = False,
    max_grasps: int,
    approach_offset: float,
    approach_axis: str = "z",  # local axis used as +approach direction
    grasp_rot_axis: str = "z",
    grasp_rot_deg: float = 90.0,
    grasp_forward_offset: float = 0.015,
    grasp_forward_axis: str = "same_as_approach",
    auto_flip_180_axis: str = "z",
    dry_run: bool = False,
    vis: bool = False,
    pause_on_failure: bool = True,
    stop_on_first_success: bool = True,
    add_pointcloud_collisions: bool = True,
    base_standoff: float = 1.4,
    lift_offset_local_z: float = 0.4,
    post_grasp_backoff: float = 0.1,
    place_to_basket: bool = True,
    place_offset_x: float = -0.03,
    place_offset_z: float = -0.35,
    visualize_only: bool = False,
    highlight_actor=None,
    highlight_color: tuple[float, float, float, float] = (1.0, 0.1, 1.0, 0.6),
):
    T_obj_to_world = np.asarray(T_obj_to_world, dtype=np.float64).reshape(4, 4)
    # Move planner collisions into mplib.
    #planner.clear_collisions()
    if add_pointcloud_collisions and collision_scene.obstacle_points.shape[0] > 0:
        planner.add_collision_pts(collision_scene.obstacle_points, name="cereals_scene_pcd")

    if not dry_run and not visualize_only:
        # Use current end-effector state as "open" baseline.
        planner.open_gripper()

    prep_res = _prepare_base_and_lift(
        env=env,
        planner=planner,
        target_center_world=collision_scene.object_center,
        base_standoff=base_standoff,
        lift_offset_local_z=lift_offset_local_z,
    )
    if prep_res == -1:
        return [False]

    axis_map = {"x": 0, "y": 1, "z": 2}
    if approach_axis not in axis_map:
        raise ValueError(f"Unknown approach_axis='{approach_axis}'. Use one of {list(axis_map.keys())}.")
    if grasp_rot_axis not in axis_map:
        raise ValueError(f"Unknown grasp_rot_axis='{grasp_rot_axis}'. Use one of {list(axis_map.keys())}.")
    if grasp_forward_axis != "same_as_approach" and grasp_forward_axis not in axis_map:
        raise ValueError(
            "Unknown grasp_forward_axis='{}'. Use one of {} or 'same_as_approach'.".format(
                grasp_forward_axis, list(axis_map.keys())
            )
        )
    if auto_flip_180_axis != "none" and auto_flip_180_axis not in axis_map:
        raise ValueError(
            "Unknown auto_flip_180_axis='{}'. Use one of {} or 'none'.".format(
                auto_flip_180_axis, list(axis_map.keys())
            )
        )
    axis_idx = axis_map[approach_axis]
    forward_axis_idx = axis_idx if grasp_forward_axis == "same_as_approach" else axis_map[grasp_forward_axis]
    grasp_rot_correction = _axis_angle_rotation(grasp_rot_axis, grasp_rot_deg)

    successes = []
    highlight_done = False
    for k in tqdm(range(min(max_grasps, len(grasp_poses_obj_frame))), desc="grasps"):
        mat_obj = grasp_poses_obj_frame[k]
        mat_world = mat_obj if grasps_already_in_world else (T_obj_to_world @ mat_obj)
        mat_world = mat_world.copy()

        # Correct potential 90-degree frame mismatch of predicted grasp orientation.
        # Right multiplication => rotation in local TCP frame.
        mat_world[:3, :3] = mat_world[:3, :3] @ grasp_rot_correction
        mat_world[:3, :3] = _align_grasp_rotation_with_tcp(
            R_grasp=mat_world[:3, :3],
            R_tcp=_get_tcp_rotation_world(env),
            flip_axis=auto_flip_180_axis,
        )

        R = mat_world[:3, :3]
        approach_dir_world = R[:, axis_idx]
        forward_dir_world = R[:, forward_axis_idx]

        # Bring TCP slightly closer to the object before planning.
        # Positive offset moves in +forward axis direction.
        mat_world[:3, 3] = mat_world[:3, 3] + forward_dir_world * float(grasp_forward_offset)
        tcp_pose = grasp_mat_to_tcp_pose(mat_world)

        # Pre-grasp: move back along the local approach axis.
        pre_p = _to_numpy(tcp_pose.p) - approach_dir_world * float(approach_offset)
        pre_pose = sapien.Pose(p=pre_p.tolist(), q=tcp_pose.q.tolist())

        if vis:
            print(f"[{k}] tcp p={np.asarray(tcp_pose.p).tolist()} pre_p={pre_p.tolist()}")
        print("visualize_only", visualize_only)
        if visualize_only:
            # Only set visual target grasp pose (no IK / no collision planning).
            if getattr(planner, "grasp_pose_visual", None) is not None:
                #planner.grasp_pose_visual.set_pose(tcp_pose)
                planner.grasp_pose_visual.set_pose(pre_pose)

            if not highlight_done and highlight_actor is not None:
                set_actor_rgba(highlight_actor, highlight_color)
                highlight_done = True

            try:
                env.unwrapped.render_human()
            except Exception:
                try:
                    planner.render_wait()
                except Exception:
                    pass

            if pause_on_failure:
                ans = input(
                    f"[{k}] Visualize-only. Press Enter for next, or 'q' to quit: "
                ).strip().lower()
                if ans == "q":
                    break

            successes.append(True)
            # In visualize-only mode we want to inspect all candidate grasps.
            if stop_on_first_success and not visualize_only:
                break
            continue

        planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
                get_fcl_object_name(target_actor), True
            )
        res_pre = planner.static_manipulation(pre_pose, n_init_qpos=400, disable_lift_joint=False)
        if res_pre == -1:
            print("pre-grasp failed")
            successes.append(False)
            if pause_on_failure:
                ans = input(f"[{k}] Pre-grasp failed. Press Enter to continue, or 'q' to quit: ").strip().lower()
                if ans == "q":
                    break
            continue

        if target_actor is not None:
            planner.planner.planning_world.get_allowed_collision_matrix().set_default_entry(
                get_fcl_object_name(target_actor), True
            )
        res = planner.static_manipulation(tcp_pose, n_init_qpos=400, disable_lift_joint=False)
        if res == -1:
            print("grasp failed")
            successes.append(False)
            if pause_on_failure:
                ans = input(f"[{k}] Grasp failed. Press Enter to continue, or 'q' to quit: ").strip().lower()
                if ans == "q":
                    break
            continue

        if not dry_run:
            planner.close_gripper()
            if target_actor is not None:
                # Attach the picked product so a subsequent base backoff moves the item together
                # with the robot (same idea as in pick_to_cart.py).
                kwargs = {
                    "name": get_fcl_object_name(target_actor),
                    "art_name": "scene-0_ds_fetch_basket_1",
                    "link_id": planner.planner.move_group_link_id,
                }
                planner.planner.planning_world.attach_object(**kwargs)
            planner.planner.update_from_simulation()
            if post_grasp_backoff != 0:
                res_backoff = planner.move_forward_delta(delta=-float(post_grasp_backoff))
                if res_backoff == -1:
                    successes.append(False)
                    if pause_on_failure:
                        ans = input(
                            f"[{k}] Post-grasp backoff failed. Press Enter to continue, or 'q' to quit: "
                        ).strip().lower()
                        if ans == "q":
                            break
                    continue
                planner.planner.update_from_simulation()
            if place_to_basket:
                if hasattr(env.unwrapped, "calc_target_pose"):
                    goal_center = env.unwrapped.calc_target_pose().sp.p
                else:
                    goal_center = _calc_basket_target_pose_world(env).p
                goal_approaching = np.array([0.0, 0.0, -1.0], dtype=np.float64)
                goal_closing = -env.unwrapped.agent.base_link.pose.sp.to_transformation_matrix()[:3, 1]
                goal_pose = env.unwrapped.agent.build_grasp_pose(goal_approaching, goal_closing, goal_center)
                goal_pose = goal_pose * sapien.Pose(p=[float(place_offset_x), 0.0, float(place_offset_z)])
                res_place = planner.static_manipulation(goal_pose, n_init_qpos=100, disable_lift_joint=False)
                if res_place == -1:
                    successes.append(False)
                    if pause_on_failure:
                        ans = input(
                            f"[{k}] Place-to-basket failed. Press Enter to continue, or 'q' to quit: "
                        ).strip().lower()
                        if ans == "q":
                            break
                    continue
                planner.planner.update_from_simulation()
                planner.open_gripper()
                planner.planner.update_from_simulation()
                planner.idle_steps(t=10)
        successes.append(True)

        if stop_on_first_success:
            break

    return successes


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cereals-json", type=str, default="cereals.json", help="Path to cereals.json")
    p.add_argument(
        "--topk-grasps-json",
        type=str,
        default="cereals_top5_grasps.json",
        help="Path to cereals_top5_grasps.json",
    )
    p.add_argument("--env-id", type=str, default="DarkstoreContinuousBaseEnv", help="ManiSkill env id to use")
    p.add_argument("--robot-uids", type=str, default="ds_fetch_basket", help="Robot UID for the env")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-traj", type=int, default=50, help="Number of episodes to run for success-rate measurement.")
    p.add_argument("--vis", action="store_true", help="Open GUI")
    p.add_argument(
        "--hold-viewer",
        action="store_true",
        help="Keep simulation window open until Ctrl+C.",
    )
    p.add_argument("--max-grasps", type=int, default=5)
    p.add_argument("--approach-offset", type=float, default=0, help="Distance for pre-grasp offset")
    p.add_argument("--approach-axis", type=str, default="y", choices=["x", "y", "z"], help="Local axis for approach")
    p.add_argument(
        "--grasp-rot-axis",
        type=str,
        default="z",
        choices=["x", "y", "z"],
        help="Local axis used to rotate grasp orientation correction.",
    )
    p.add_argument(
        "--grasp-rot-deg",
        type=float,
        default=-90.0,
        help="Grasp orientation correction angle in degrees (local frame).",
    )
    p.add_argument(
        "--grasp-forward-offset",
        type=float,
        default=0.2,
        help="Translate grasp TCP along +grasp-forward-axis.",
    )
    p.add_argument(
        "--grasp-forward-axis",
        type=str,
        default="same_as_approach",
        choices=["x", "y", "z", "same_as_approach"],
        help="Local axis used for grasp TCP translation.",
    )
    p.add_argument(
        "--auto-flip-180-axis",
        type=str,
        default="z",
        choices=["none", "x", "y", "z"],
        help="Auto-resolve 180-degree grasp ambiguity using current TCP orientation.",
    )
    p.add_argument(
        "--obstacle-downsample",
        type=int,
        default=8000,
        help="Max number of collision points to feed to mplib",
    )
    p.add_argument(
        "--include-object-in-obstacles",
        action="store_true",
        help="By default we exclude the object AABB from obstacles so the robot can reach the grasp pose.",
    )
    p.add_argument("--exclude-bbox-margin", type=float, default=0.0001)
    p.add_argument("--dry-run", action="store_true", help="Plan without executing robot actions")
    p.add_argument(
        "--grasps-already-in-world",
        action="store_true",
        help=(
            "Treat 4x4 grasp transforms from `grasp_poses_obj_frame_topk` as already expressed in "
            "the simulator world frame. If set, we skip `T_obj_to_world @ mat_obj`."
        ),
    )
    p.add_argument(
        "--vis-only",
        action="store_true",
        help="Do not run IK/motion planning; only set grasp pose visual + render frames.",
    )
    p.add_argument(
        "--with-pointcloud-collisions",
        action="store_true",
        help=(
            "If set, adds `scene_info` pointcloud from cereals.json as extra collision points for mplib."
            " Usually for fetch robots this should be OFF because the scene alignment might be imperfect."
        ),
    )
    p.add_argument(
        "--base-standoff",
        type=float,
        default=0.95,
        help="Distance to shelf before manipulation.",
    )
    p.add_argument(
        "--lift-offset-local-z",
        type=float,
        default=0,
        help="Lift offset after setting TCP z to object level.",
    )
    p.add_argument(
        "--post-grasp-backoff",
        type=float,
        default=0.3,
        help="How far to move the base backwards after grasp (meters).",
    )
    p.add_argument(
        "--no-place-to-basket",
        action="store_true",
        help="Disable place-to-basket stage after grasp and backoff.",
    )
    p.add_argument("--place-offset-x", type=float, default=-0.03, help="Local x offset for place pose.")
    p.add_argument("--place-offset-z", type=float, default=-0.35, help="Local z offset for place pose.")
    p.add_argument(
        "--basket-success-radius",
        type=float,
        default=0.25,
        help="Episode success: product must be within this distance (m) of basket target pose.",
    )
    p.add_argument(
        "--cereals-pointclouds-already-in-world",
        action="store_true",
        help=(
            "Treat `cereals.json` pointclouds (`object_info.pc`, `scene_info.pc_color`) as already in world frame. "
            "If set, we skip applying T_obj_to_world to point clouds."
        ),
    )
    p.add_argument(
        "--anchor-product-name-contains",
        type=str,
        default="",
        help=(
            "For DarkstoreContinuousBaseEnv: substring to match product actor name in "
            "`env.unwrapped.actors['products']` (default: first product)."
        ),
    )
    p.add_argument(
        "--no-pause-on-failure",
        action="store_true",
        help="Disable waiting for Enter after a failed grasp attempt.",
    )
    p.add_argument(
        "--no-stop-on-first-success",
        action="store_true",
        help="Continue trying other grasps after the first success.",
    )
    p.add_argument(
        "--scene-dir",
        type=str,
        default=None,
        help="Darkstore scene config dir (also used for demos/motionplanning output). Default: demo_envs/pick_to_basket under darkstore-synthesizer.",
    )
    p.add_argument(
        "--record-dir",
        type=str,
        default=None,
        help="Directory for saving demos (h5/videos). Default: <scene-dir>/demos.",
    )
    p.add_argument(
        "--save-video",
        action="store_true",
        help="Save MP videos next to trajectory .h5 (same pattern as run_mp.py).",
    )
    p.add_argument(
        "--traj-name",
        type=str,
        default=None,
        help="Trajectory .h5 base name (default: timestamp).",
    )
    p.add_argument(
        "--shader",
        type=str,
        default="default",
        help="Shader pack for viewer/human render/sensors (same as run_mp.py).",
    )
   
    return p.parse_args()


def main():
    args = parse_args()
    cereals_json_path = Path(args.cereals_json)
    topk_grasps_json_path = Path(args.topk_grasps_json)

    if not cereals_json_path.exists():
        raise FileNotFoundError(cereals_json_path)
    if not topk_grasps_json_path.exists():
        raise FileNotFoundError(topk_grasps_json_path)

    # Load point clouds and grasps (in JSON coordinates).
    object_pc, _scene_pc = load_cereals_scene(cereals_json_path)
    grasp_mats = load_topk_grasps(topk_grasps_json_path)

    json_obj_center = object_pc.mean(axis=0).astype(np.float32)

    scene_dir = args.scene_dir
    if scene_dir is None:
        scene_dir = str(ROOT_DIR / "demo_envs" / "pick_to_basket")
    record_dir = args.record_dir if args.record_dir is not None else osp.join(scene_dir, "demos")
    os.makedirs(osp.join(record_dir, "motionplanning"), exist_ok=True)

    # RecordEpisode captures frames via env.render(); with render_mode="human" render() may not
    # return an HxWx3 array -> flush_video crashes (imageio/ffmpeg). Use rgb_array when recording video.
    render_mode = "rgb_array" if args.save_video else ("human" if args.vis else "rgb_array")

    # Create env and align JSON coordinates into env coordinates by object center translation.
    env = gym.make(
        args.env_id,
        robot_uids=args.robot_uids,
        config_dir_path=scene_dir,
        num_envs=1,
        control_mode="pd_joint_pos",
        obs_mode="none",
        render_mode=render_mode,
        enable_shadow=True,
        viewer_camera_configs={"shader_pack": args.shader},
        human_render_camera_configs={"shader_pack": args.shader},
        sensor_configs={"shader_pack": args.shader},
        parallel_in_single_scene=False,
    )
    traj_name = args.traj_name if args.traj_name else time.strftime("%Y%m%d_%H%M%S")
    env = RecordEpisode(
        env,
        output_dir=osp.join(record_dir, "motionplanning"),
        trajectory_name=traj_name,
        save_video=args.save_video,
        source_type="motionplanning",
        source_desc="cereals JSON grasp motion planning (run_cereals_json_motion_planning)",
        video_fps=30,
        record_reward=False,
        save_on_reset=False,
    )
    output_h5_path = env._h5_file.filename
    print(f"Trajectory recording: {output_h5_path}")
    if args.save_video and args.vis:
        print("Note: --save-video forces render_mode=rgb_array (needed for video frames). Use render_human in GUI as needed.")

    try:
        episode_successes = []
        saved_successful_trajectories = 0
        for ep_idx in range(args.num_traj):
            cur_seed = args.seed + ep_idx
            print(f"\n=== Episode {ep_idx + 1}/{args.num_traj}, seed={cur_seed} ===")
            #env.reset(seed=cur_seed, options={"reconfigure": True})
            env.reset(seed=0,options={"reconfigure": True})

            # Anchor: align JSON coordinates to sim world frame.
            anchor_center = None
            anchor_R = None
            anchor_actor = None
            if hasattr(env.unwrapped, "cube"):
                anchor_center = _to_numpy(env.unwrapped.cube.pose.sp.p).reshape(3)
            elif hasattr(env.unwrapped, "target_pose"):
                anchor_center = _to_numpy(env.unwrapped.target_pose[0].sp.p).reshape(3)
            else:
                products = getattr(env.unwrapped, "actors", {}).get("products", {})
                if isinstance(products, dict) and len(products) > 0:
                    anchor_product_name_contains = (args.anchor_product_name_contains or "").strip()
                    keys = sorted(list(products.keys()))
                    chosen_key: Optional[str] = None
                    if anchor_product_name_contains:
                        for k in keys:
                            if anchor_product_name_contains in k:
                                chosen_key = k
                                break
                    else:
                        chosen_key = keys[0]

                    if chosen_key is None:
                        raise RuntimeError(
                            "Cannot find product actor for anchor. "
                            f"Substring='{anchor_product_name_contains}'. Available product actors: {keys[:10]}..."
                        )

                    anchor_pose_sp = products[chosen_key].pose.sp
                    anchor_center = _to_numpy(anchor_pose_sp.p).reshape(3)
                    anchor_T = _to_numpy(anchor_pose_sp.to_transformation_matrix()).reshape(4, 4)
                    anchor_R = anchor_T[:3, :3]
                    anchor_actor = products[chosen_key]
                    print(f"Anchor product actor: {chosen_key}. anchor_center={anchor_center.tolist()}")

            if anchor_center is None:
                raise RuntimeError("Cannot find an anchor position in env for JSON alignment.")

            if anchor_R is None:
                anchor_R = np.eye(3, dtype=np.float64)
            anchor_R = np.asarray(anchor_R, dtype=np.float64).reshape(3, 3)
            t_obj_to_world = anchor_center.astype(np.float64) - (anchor_R @ json_obj_center.astype(np.float64))
            T_obj_to_world = np.eye(4, dtype=np.float64)
            T_obj_to_world[:3, :3] = anchor_R
            T_obj_to_world[:3, 3] = t_obj_to_world
            print(f"T_obj_to_world translation={t_obj_to_world.tolist()}")

            collision_scene = CollisionSceneFromPointCloud.from_cereals_json(
                cereals_json_path,
                R_obj_to_world=anchor_R,
                t_obj_to_world=t_obj_to_world,
                obstacle_downsample=args.obstacle_downsample,
                exclude_object_from_obstacles=not args.include_object_in_obstacles,
                exclude_bbox_margin=args.exclude_bbox_margin,
                rng_seed=cur_seed,
                pointclouds_already_in_world=args.cereals_pointclouds_already_in_world,
            )

            planner = FetchMotionPlanningSapienSolver(
                env,
                debug=True,
                vis=args.vis,
                base_pose=env.unwrapped.agent.robot.pose,
                visualize_target_grasp_pose=args.vis,
                print_env_info=False,
                joint_vel_limits=0.9,
                joint_acc_limits=0.9,
            )

            successes = plan_grasps_from_json(
                env=env,
                planner=planner,
                collision_scene=collision_scene,
                grasp_poses_obj_frame=grasp_mats,
                T_obj_to_world=T_obj_to_world,
                target_actor=anchor_actor,
                grasps_already_in_world=args.grasps_already_in_world,
                max_grasps=args.max_grasps,
                approach_offset=args.approach_offset,
                approach_axis=args.approach_axis,
                grasp_rot_axis=args.grasp_rot_axis,
                grasp_rot_deg=args.grasp_rot_deg,
                grasp_forward_offset=args.grasp_forward_offset,
                grasp_forward_axis=args.grasp_forward_axis,
                auto_flip_180_axis=args.auto_flip_180_axis,
                dry_run=args.dry_run,
                vis=args.vis,
                pause_on_failure=not args.no_pause_on_failure,
                stop_on_first_success=not args.no_stop_on_first_success,
                add_pointcloud_collisions=args.with_pointcloud_collisions,
                base_standoff=args.base_standoff,
                lift_offset_local_z=args.lift_offset_local_z,
                post_grasp_backoff=args.post_grasp_backoff,
                place_to_basket=not args.no_place_to_basket,
                place_offset_x=args.place_offset_x,
                place_offset_z=args.place_offset_z,
                visualize_only=args.vis_only,
                highlight_actor=anchor_actor if args.vis_only else None,
            )

            ok = sum(1 for x in successes if x)
            if anchor_actor is not None:
                episode_success, dist_to_basket = _episode_success_by_basket_proximity(
                    env, anchor_actor, args.basket_success_radius
                )
            else:
                episode_success, dist_to_basket = False, float("nan")
            episode_successes.append(episode_success)
            print(
                f"Episode result: successful grasps={ok}/{len(successes)}, "
                f"dist_to_basket={dist_to_basket:.4f}m (threshold={args.basket_success_radius}m), "
                f"episode_success={episode_success}, success_rate={np.mean(episode_successes):.3f}"
            )
            env.flush_trajectory(save=episode_success)
            if args.save_video:
                try:
                    env.flush_video(save=episode_success)
                except (ValueError, OSError) as e:
                    print(f"Warning: flush_video failed ({e}); skipping video for this episode.")
            if episode_success:
                saved_successful_trajectories += 1

        final_success_rate = np.mean(episode_successes) if len(episode_successes) > 0 else 0.0
        print(
            f"\nFinished {len(episode_successes)} episode(s). "
            f"Success rate={final_success_rate:.3f} ({sum(episode_successes)}/{len(episode_successes)})"
        )
        print(
            f"Saved successful trajectories to h5: {saved_successful_trajectories}/"
            f"{len(episode_successes)} (saved_success_rate={final_success_rate:.3f})"
        )
        print(f"Saved trajectory bundle: {output_h5_path}")
        if args.vis and args.hold_viewer:
            print("Viewer hold enabled. Press Ctrl+C to close.")
            try:
                while True:
                    env.unwrapped.render_human()
            except KeyboardInterrupt:
                pass
    finally:
        env.close()


if __name__ == "__main__":
    main()

# usage python ./scripts/run_cereals_json_motion_planning.py   --cereals-json ../monster_demopick.json   --topk-grasps-json /home/vadims077/GraspSkil/monster_demopick_top5_grasps.json   --robot-uids ds_fetch_basket   --anchor-product-name-contains "[ENV#0]_food.ENERGY_DRINKS.MonsterEnergyDrink:0:2:0:0"   --vis   --max-grasps 5    --grasps-already-in-world --cereals-pointclouds-already-in-world   --grasp-rot-axis z --grasp-rot-deg -90 --grasp-forward-axis z --grasp-forward-offset 0.2 --hold-viewer 