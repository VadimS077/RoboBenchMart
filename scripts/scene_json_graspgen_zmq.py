#!/usr/bin/env python3
"""
Сцена → GraspGen ZMQ (infer_scene с полной сценой для коллизий), выход: JSON с top-K хватами
для run_cereals_json_motion_planning (--topk-grasps-json).

Источник сцены (ровно один):

  * **Захват из ManiSkill** — та же логика, что ``save_view.py`` (pointcloud + сегментация объекта),
    без обязательного сохранения JSON.
  * ``--scene-json`` — готовый JSON от save_view.
  * ``--pc-npz`` — готовое сжатое облако (object_xyz + scene_xyz), см. ``--save-pc-npz``.

Примеры:

  # Захват сцены, сохранить облака в NPZ, GraspGen, сохранить хваты в JSON
  python scripts/scene_json_graspgen_zmq.py --capture --scene-dir demo_envs/pick_to_basket \\
      --save-pc-npz ./run_scene.npz --out-json ./grasps_topk.json \\
      --host localhost --port 5556

  # Только JSON сцены (как раньше)
  python scripts/scene_json_graspgen_zmq.py --scene-json scene.json --out-json ./grasps_topk.json

  # Ранее сохранённый NPZ
  python scripts/scene_json_graspgen_zmq.py --pc-npz ./run_scene.npz --out-json ./grasps_topk.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

# darkstore-synthesizer/
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Репозиторий GraspGen (сосед с darkstore-synthesizer)
_GRASPGEN_ROOT = Path(os.environ.get("GRASPGEN_ROOT", "")).expanduser() if os.environ.get("GRASPGEN_ROOT") else None
if _GRASPGEN_ROOT and not (_GRASPGEN_ROOT / "grasp_gen").is_dir():
    _GRASPGEN_ROOT = None
if _GRASPGEN_ROOT is None:
    _GRASPGEN_ROOT = ROOT_DIR.parent / "GraspGen"
if str(_GRASPGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GRASPGEN_ROOT))

import gymnasium as gym  # noqa: E402

from grasp_gen.serving.zmq_client import GraspGenClient  # noqa: E402

# Регистрации окружений dsynth (как в save_view.py)
from dsynth.envs import *  # noqa: F401, F403, E402
from dsynth.robots import *  # noqa: F401, F403, E402


def _resolve_selected_id(env, selected_id_arg: Any) -> Optional[int]:
    """Логика как в save_view: цифра / имя из segmentation_id_map / подстрока в obj.name."""
    if selected_id_arg is None:
        return None
    if isinstance(selected_id_arg, str) and selected_id_arg.isdigit():
        return int(selected_id_arg)
    if isinstance(selected_id_arg, int):
        return selected_id_arg
    seg_map = getattr(env.unwrapped, "segmentation_id_map", None)
    if seg_map is None:
        return None
    reverse_name_to_id = {obj.name: obj_id for obj_id, obj in seg_map.items()}
    if selected_id_arg in reverse_name_to_id:
        return reverse_name_to_id[selected_id_arg]
    for obj_id, obj in seg_map.items():
        try:
            if selected_id_arg in obj.name:
                return obj_id
        except Exception:
            continue
    return None


def capture_pointcloud_from_maniskill(
    *,
    env_id: str,
    robot_uids: str,
    scene_dir: str | Path,
    num_envs: int,
    cam_width: int,
    cam_height: int,
    selected_id_arg: str,
    shader: str,
    gui: bool,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Как ``save_view.save_pointcloud_from_maniskill``: одна симуляция, obs pointcloud,
    маска по сегментации для объекта. Возвращает:
      object_xyz (No,3), object_rgb (No,3) uint8, scene_xyz (Ns,3), scene_rgb (Ns,3) uint8
    """
    sensor_configs = {"width": cam_width, "height": cam_height}
    parallel_in_single_scene = num_envs > 1 and gui
    env = gym.make(
        env_id,
        robot_uids=robot_uids,
        config_dir_path=str(scene_dir),
        num_envs=num_envs,
        viewer_camera_configs={"shader_pack": shader},
        human_render_camera_configs={"shader_pack": shader},
        render_mode="human" if gui else "rgb_array",
        control_mode=None,
        enable_shadow=True,
        sim_config={"spacing": 20},
        obs_mode="pointcloud",
        sim_backend="auto",
        parallel_in_single_scene=parallel_in_single_scene,
        sensor_configs=sensor_configs,
    )
    try:
        obs, _ = env.reset(seed=seed, options={"reconfigure": True})
        xyz = obs["pointcloud"]["xyzw"][0, ..., :3].cpu().numpy()
        rgb = obs["pointcloud"]["rgb"][0].cpu().numpy()
        seg_raw = obs["pointcloud"]["segmentation"][0].cpu().numpy()

        selected_id = _resolve_selected_id(env, selected_id_arg)
        seg_map = getattr(env.unwrapped, "segmentation_id_map", None)
        if seg_map is not None:
            print({obj_id: obj.name for obj_id, obj in seg_map.items()})

        if seg_raw is None:
            mask_obj = np.ones(len(xyz), dtype=bool)
        else:
            mask_obj = (seg_raw == selected_id) if selected_id is not None else np.ones(len(xyz), dtype=bool)

        mask_obj = mask_obj.reshape(-1)
        xyz_obj = xyz[mask_obj]
        rgb_obj = rgb[mask_obj]

        return (
            xyz_obj.astype(np.float32),
            rgb_obj.astype(np.uint8),
            xyz.astype(np.float32),
            rgb.astype(np.uint8),
        )
    finally:
        env.close()


def save_pointcloud_npz(
    path: Path,
    object_xyz: np.ndarray,
    scene_xyz: np.ndarray,
    object_rgb: Optional[np.ndarray] = None,
    scene_rgb: Optional[np.ndarray] = None,
    meta: Optional[dict[str, Any]] = None,
) -> None:
    """Сохранить облака в ``.npz`` (компактнее и быстрее, чем JSON)."""
    path = Path(path)
    payload: dict[str, np.ndarray] = {
        "object_xyz": np.asarray(object_xyz, dtype=np.float32),
        "scene_xyz": np.asarray(scene_xyz, dtype=np.float32),
    }
    if object_rgb is not None:
        payload["object_rgb"] = np.asarray(object_rgb)
    if scene_rgb is not None:
        payload["scene_rgb"] = np.asarray(scene_rgb)
    np.savez_compressed(path, **payload)
    if meta:
        meta_path = path.with_suffix(".meta.json")
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def load_pointcloud_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    obj = np.asarray(z["object_xyz"], dtype=np.float32).reshape(-1, 3)
    scene = np.asarray(z["scene_xyz"], dtype=np.float32).reshape(-1, 3)
    return obj, scene


def load_save_view_scene_json(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    obj = np.asarray(data["object_info"]["pc"], dtype=np.float32)
    scene_info = data.get("scene_info") or {}
    if "pc_color" in scene_info:
        scene = np.asarray(scene_info["pc_color"], dtype=np.float32)
    elif "full_pc" in scene_info:
        scene = np.asarray(scene_info["full_pc"], dtype=np.float32)
    else:
        raise KeyError("scene_info must contain 'pc_color' or 'full_pc' (see save_view.py)")

    if scene.ndim == 3:
        scene = scene[0]
    if obj.ndim == 3:
        obj = obj[0]

    return obj.reshape(-1, 3), scene.reshape(-1, 3)


def parse_args():
    p = argparse.ArgumentParser(
        description="ManiSkill capture / JSON / NPZ → GraspGen ZMQ infer_scene → top-K grasps JSON"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--capture",
        action="store_true",
        help="Захватить сцену из ManiSkill (логика как save_view.py save)",
    )
    src.add_argument("--scene-json", type=Path, help="JSON от save_view")
    src.add_argument("--pc-npz", type=Path, help="Загрузить object_xyz / scene_xyz из .npz")

    p.add_argument(
        "--scene-dir",
        type=Path,
        default=None,
        help="Каталог конфига сцены (обязателен для --capture)",
    )
    p.add_argument(
        "--save-pc-npz",
        type=Path,
        default=None,
        help="После захвата сохранить облака в этот .npz (+ опционально .meta.json)",
    )
    p.add_argument(
        "--save-scene-json",
        type=Path,
        default=None,
        help="После захвата сохранить сцену в JSON в формате save_view (для --cereals-json в run_cereals_json_motion_planning)",
    )
    p.add_argument("--env-id", type=str, default="DarkstoreContinuousBaseEnv")
    p.add_argument("--robot-uids", type=str, default="ds_fetch_basket")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--cam-width", type=int, default=1024)
    p.add_argument("--cam-height", type=int, default=1024)
    p.add_argument(
        "--selected-id",
        type=str,
        default="[ENV#0]_food.CRACKERS_COOKIES.OreoLemonCremeSandwichCookies:0:1:0:0",
        help="Сегментация цели (как --selected-id в save_view)",
    )
    p.add_argument(
        "--shader",
        type=str,
        default="default",
        choices=["rt", "rt-fast", "rt-med", "default", "minimal"],
    )
    p.add_argument("--gui", action="store_true", default=True, help="Окно симулятора при захвате")
    p.add_argument("--no-gui", action="store_true", help="Без окна (rgb_array)")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--out-json", type=Path, default=None, help="Куда сохранить top-K хватов")
    p.add_argument("--host", type=str, default="localhost")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--no-filter-collisions", action="store_true")
    p.add_argument("--collision-threshold", type=float, default=0.02)
    p.add_argument("--max-scene-points", type=int, default=8192)
    p.add_argument("--output-topk", type=int, default=100)
    p.add_argument("--num-grasps", type=int, default=200)
    p.add_argument("--grasp-threshold", type=float, default=-1.0)
    p.add_argument("--topk-num-grasps", type=int, default=-1)
    p.add_argument("--no-remove-outliers", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    gui = args.gui and not args.no_gui

    source_desc: str
    meta_capture: dict[str, Any] = {}

    if args.capture:
        if args.scene_dir is None:
            print("--capture requires --scene-dir", file=sys.stderr)
            return 1
        scene_dir = args.scene_dir.resolve()
        if not scene_dir.is_dir():
            print(f"Not a directory: {scene_dir}", file=sys.stderr)
            return 1

        print(f"Capturing point cloud (env={args.env_id}, scene_dir={scene_dir}) ...")
        obj_xyz, obj_rgb, scene_xyz, scene_rgb = capture_pointcloud_from_maniskill(
            env_id=args.env_id,
            robot_uids=args.robot_uids,
            scene_dir=scene_dir,
            num_envs=args.num_envs,
            cam_width=args.cam_width,
            cam_height=args.cam_height,
            selected_id_arg=args.selected_id,
            shader=args.shader,
            gui=gui,
            seed=args.seed,
        )
        source_desc = f"capture:{scene_dir}"
        meta_capture = {
            "env_id": args.env_id,
            "robot_uids": args.robot_uids,
            "scene_dir": str(scene_dir),
            "selected_id": args.selected_id,
            "seed": args.seed,
        }
        if args.save_pc_npz:
            sp = args.save_pc_npz.resolve()
            save_pointcloud_npz(
                sp,
                obj_xyz,
                scene_xyz,
                object_rgb=obj_rgb,
                scene_rgb=scene_rgb,
                meta=meta_capture,
            )
            print(f"Saved point clouds -> {sp} (+ optional .meta.json)")
        if args.save_scene_json:
            jp = args.save_scene_json.resolve()
            data = {
                "object_info": {
                    "pc": obj_xyz.tolist(),
                    "pc_color": obj_rgb.tolist(),
                },
                "scene_info": {
                    "pc_color": scene_xyz.tolist(),
                    "img_color": scene_rgb.tolist(),
                },
                "grasp_info": {"grasp_poses": [], "grasp_conf": []},
            }
            with jp.open("w", encoding="utf-8") as f:
                json.dump(data, f)
            print(f"Saved scene JSON (save_view format) -> {jp}")
    elif args.pc_npz is not None:
        pc_path = args.pc_npz.resolve()
        if not pc_path.is_file():
            print(f"Not found: {pc_path}", file=sys.stderr)
            return 1
        obj_xyz, scene_xyz = load_pointcloud_npz(pc_path)
        source_desc = str(pc_path)
    else:
        scene_path = args.scene_json.resolve()
        if not scene_path.is_file():
            print(f"File not found: {scene_path}", file=sys.stderr)
            return 1
        obj_xyz, scene_xyz = load_save_view_scene_json(scene_path)
        source_desc = str(scene_path)

    obj_pc, scene_pc = obj_xyz, scene_xyz
    print(f"Object points: {len(obj_pc)}, scene points: {len(scene_pc)}  [{source_desc}]")

    out_path = args.out_json
    if out_path is None:
        stem = "grasps_zmq_topk"
        if args.capture and args.scene_dir:
            stem = f"{Path(args.scene_dir).name}_zmq_topk_grasps"
        elif args.scene_json:
            stem = f"{Path(args.scene_json).stem}_zmq_topk_grasps"
        elif args.pc_npz:
            stem = f"{Path(args.pc_npz).stem}_zmq_topk_grasps"
        out_path = Path.cwd() / f"{stem}.json"
    else:
        out_path = out_path.resolve()

    t0 = time.monotonic()
    with GraspGenClient(host=args.host, port=args.port) as client:
        meta_srv = client.server_metadata or {}
        grasps, conf, meta = client.infer_scene(
            obj_pc,
            None if args.no_filter_collisions else scene_pc,
            filter_collisions=not args.no_filter_collisions,
            grasp_threshold=args.grasp_threshold,
            num_grasps=args.num_grasps,
            topk_num_grasps=args.topk_num_grasps,
            remove_outliers=not args.no_remove_outliers,
            collision_threshold=args.collision_threshold,
            max_scene_points=args.max_scene_points,
            output_topk=args.output_topk,
        )

    elapsed = time.monotonic() - t0
    gripper_name = meta_srv.get("gripper_name", "unknown")

    k = int(len(grasps))
    payload = {
        "source": source_desc,
        "capture_meta": meta_capture if meta_capture else None,
        "gripper_name": gripper_name,
        "server_metadata": meta_srv,
        "zmq_meta": meta,
        "collision_filter": not args.no_filter_collisions,
        "collision_threshold_m": args.collision_threshold if not args.no_filter_collisions else None,
        "topk": k,
        "round_trip_s": elapsed,
        "grasp_conf_topk": conf.tolist() if k else [],
        "grasp_poses_obj_frame_topk": grasps.tolist() if k else [],
    }
    # Совместимость с прошлым полем и с run_cereals (если передали JSON путь)
    if args.scene_json and not args.capture:
        payload["source_scene_json"] = str(args.scene_json.resolve())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {k} grasps -> {out_path}")
    if k:
        print(f"  confidence: {float(conf.min()):.4f} .. {float(conf.max()):.4f}")
    return 0 if k > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
