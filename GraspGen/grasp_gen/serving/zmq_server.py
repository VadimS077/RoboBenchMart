# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import time
import logging
from typing import Optional

import numpy as np
import trimesh.transformations as tra
import zmq
import msgpack
import msgpack_numpy

msgpack_numpy.patch()

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.robot import get_gripper_info
from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps

logger = logging.getLogger(__name__)


class GraspGenZMQServer:
    """ZMQ server that wraps GraspGenSampler for remote grasp inference.

    Protocol (msgpack over ZMQ REP socket):
        Request:  {"action": "infer", "point_cloud": ndarray(N,3), ...params}
                  {"action": "metadata"}
                  {"action": "health"}
        Response: msgpack-encoded dict with results or error.
    """

    def __init__(
        self,
        gripper_config: str,
        host: str = "0.0.0.0",
        port: int = 5556,
    ) -> None:
        self._host = host
        self._port = port
        self._gripper_config = gripper_config

        logger.info("Loading gripper config from %s", gripper_config)
        self._cfg = load_grasp_cfg(gripper_config)
        self._gripper_name = self._cfg.data.gripper_name
        self._model_name = self._cfg.eval.model_name

        logger.info(
            "Initializing GraspGenSampler (model=%s, gripper=%s)",
            self._model_name,
            self._gripper_name,
        )
        self._sampler = GraspGenSampler(self._cfg)
        logger.info("Model loaded and ready for inference")

        self._metadata = {
            "gripper_name": self._gripper_name,
            "model_name": self._model_name,
            "gripper_config": gripper_config,
        }
        self._gripper_collision_mesh = None  # lazy: trimesh, for infer_scene collision filter

    def serve_forever(self) -> None:
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REP)
        bind_addr = f"tcp://{self._host}:{self._port}"
        socket.bind(bind_addr)
        logger.info("GraspGen ZMQ server listening on %s", bind_addr)

        try:
            while True:
                raw = socket.recv()
                try:
                    request = msgpack.unpackb(raw, raw=False)
                    response = self._handle(request)
                except Exception as exc:
                    logger.exception("Error handling request")
                    response = {"error": str(exc)}
                socket.send(msgpack.packb(response, use_bin_type=True))
        except KeyboardInterrupt:
            logger.info("Shutting down server")
        finally:
            socket.close()
            ctx.term()

    def _handle(self, request: dict) -> dict:
        action = request.get("action")
        if action == "health":
            return {"status": "ok"}
        if action == "metadata":
            return self._metadata
        if action == "infer":
            return self._handle_infer(request)
        if action == "infer_scene":
            return self._handle_infer_scene(request)
        return {"error": f"Unknown action: {action}"}

    def _get_gripper_collision_mesh(self):
        if self._gripper_collision_mesh is None:
            gi = get_gripper_info(self._gripper_name)
            self._gripper_collision_mesh = gi.collision_mesh
            logger.info(
                "Loaded gripper collision mesh for scene filtering: %s (%d verts)",
                self._gripper_name,
                len(self._gripper_collision_mesh.vertices),
            )
        return self._gripper_collision_mesh

    def _handle_infer(self, request: dict) -> dict:
        point_cloud = request.get("point_cloud")
        if point_cloud is None:
            return {"error": "Missing required field 'point_cloud'"}

        point_cloud = np.asarray(point_cloud, dtype=np.float32)
        if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
            return {
                "error": f"point_cloud must be (N, 3), got {point_cloud.shape}"
            }

        params = {
            "grasp_threshold": float(request.get("grasp_threshold", -1.0)),
            "num_grasps": int(request.get("num_grasps", 200)),
            "topk_num_grasps": int(request.get("topk_num_grasps", -1)),
            "min_grasps": int(request.get("min_grasps", 40)),
            "max_tries": int(request.get("max_tries", 6)),
            "remove_outliers": bool(request.get("remove_outliers", True)),
        }

        t0 = time.monotonic()
        grasps, grasp_conf = GraspGenSampler.run_inference(
            point_cloud, self._sampler, **params
        )
        infer_ms = (time.monotonic() - t0) * 1000

        if len(grasps) == 0:
            return {
                "grasps": np.empty((0, 4, 4), dtype=np.float32),
                "confidences": np.empty((0,), dtype=np.float32),
                "num_grasps": 0,
                "timing": {"infer_ms": infer_ms},
            }

        grasps_np = grasps.cpu().numpy().astype(np.float32)
        conf_np = grasp_conf.cpu().numpy().astype(np.float32)

        logger.info(
            "Inferred %d grasps in %.1f ms (conf range %.3f - %.3f)",
            len(grasps_np),
            infer_ms,
            conf_np.min(),
            conf_np.max(),
        )

        return {
            "grasps": grasps_np,
            "confidences": conf_np,
            "num_grasps": len(grasps_np),
            "timing": {"infer_ms": infer_ms},
        }

    def _handle_infer_scene(self, request: dict) -> dict:
        """
        Object point cloud -> GraspGen inference, optional scene PC collision filter
        (same idea as scripts/demo_scene_pc.py): grasps returned in the object/world
        frame of the input points.
        """
        obj_pc = request.get("object_point_cloud")
        if obj_pc is None:
            return {"error": "Missing required field 'object_point_cloud'"}

        obj_pc = np.asarray(obj_pc, dtype=np.float32)
        if obj_pc.ndim != 2 or obj_pc.shape[1] != 3:
            return {"error": f"object_point_cloud must be (N, 3), got {obj_pc.shape}"}

        scene_pc = request.get("scene_point_cloud")
        filter_collisions = bool(request.get("filter_collisions", True))
        if filter_collisions:
            if scene_pc is None:
                return {
                    "error": "filter_collisions=True requires 'scene_point_cloud' (full scene, same frame as object)"
                }
            scene_pc = np.asarray(scene_pc, dtype=np.float32)
            if scene_pc.ndim != 2 or scene_pc.shape[1] != 3:
                return {
                    "error": f"scene_point_cloud must be (M, 3), got {scene_pc.shape}"
                }

        params = {
            "grasp_threshold": float(request.get("grasp_threshold", -1.0)),
            "num_grasps": int(request.get("num_grasps", 200)),
            "topk_num_grasps": int(request.get("topk_num_grasps", -1)),
            "min_grasps": int(request.get("min_grasps", 40)),
            "max_tries": int(request.get("max_tries", 6)),
            "remove_outliers": bool(request.get("remove_outliers", True)),
        }
        collision_threshold = float(request.get("collision_threshold", 0.02))
        max_scene_points = int(request.get("max_scene_points", 8192))
        output_topk = int(request.get("output_topk", 100))
        if output_topk < 1:
            output_topk = 100

        # --- Approach direction filter ---
        # approach_direction (3,) — preferred approach direction in the same frame as point clouds.
        #   Usually: direction from robot to the object (e.g. directions_to_shelf).
        # robot_position (3,) — alternative: compute per-grasp direction (grasp_center → robot) and
        #   filter grasps whose approach axis points away from the robot.
        # approach_axis_index: which column of grasp R is the approach axis (0=x, 1=y, 2=z). Default 2.
        # approach_cos_threshold: minimum cosine(approach, preferred). Default 0 (hemisphere).
        approach_dir_raw = request.get("approach_direction")
        robot_pos_raw = request.get("robot_position")
        approach_axis_index = int(request.get("approach_axis_index", 2))
        approach_cos_threshold = float(request.get("approach_cos_threshold", 0.0))
        filter_approach = approach_dir_raw is not None or robot_pos_raw is not None
        approach_dir = None
        robot_pos = None
        if approach_dir_raw is not None:
            approach_dir = np.asarray(approach_dir_raw, dtype=np.float64).reshape(3)
            n = np.linalg.norm(approach_dir)
            if n > 1e-8:
                approach_dir = approach_dir / n
            else:
                filter_approach = False
        if robot_pos_raw is not None:
            robot_pos = np.asarray(robot_pos_raw, dtype=np.float64).reshape(3)

        t_infer0 = time.monotonic()
        print(f"obj_pc: {obj_pc.shape}")
        grasps, grasp_conf = GraspGenSampler.run_inference(
            obj_pc, self._sampler, **params
        )
        print(f"grasps: {grasps.shape}")
        infer_ms = (time.monotonic() - t_infer0) * 1000

        if len(grasps) == 0:
            return {
                "grasps": np.empty((0, 4, 4), dtype=np.float32),
                "confidences": np.empty((0,), dtype=np.float32),
                "num_grasps": 0,
                "num_grasps_collision_free": 0,
                "filter_collisions": filter_collisions,
                "timing": {"infer_ms": infer_ms, "collision_ms": 0.0},
            }

        grasps_np = grasps.cpu().numpy().astype(np.float64)
        conf_np = grasp_conf.cpu().numpy().astype(np.float64)
        grasps_np[:, 3, 3] = 1.0

        # Mean of object cloud (same frame as model output); matches demo_scene_pc centering.
        obj_mean = obj_pc.astype(np.float64).mean(axis=0)
        T_center = tra.translation_matrix(-obj_mean)

        collision_ms = 0.0
        num_before = len(grasps_np)

        if filter_collisions:
            t_col0 = time.monotonic()
            grasps_c = np.stack([T_center @ np.asarray(g) for g in grasps_np], axis=0)
            scene_c = tra.transform_points(scene_pc.astype(np.float64), T_center)

            if len(scene_c) > max_scene_points:
                rng = np.random.default_rng(0)
                idx = rng.choice(len(scene_c), max_scene_points, replace=False)
                scene_c = scene_c[idx]

            mesh = self._get_gripper_collision_mesh()
            collision_mask = filter_colliding_grasps(
                scene_pc=scene_c.astype(np.float32),
                grasp_poses=grasps_c.astype(np.float64),
                gripper_collision_mesh=mesh,
                collision_threshold=collision_threshold,
            )
            grasps_c_free = grasps_c[collision_mask]
            conf_free = conf_np[collision_mask]
            collision_ms = (time.monotonic() - t_col0) * 1000

            T_inv = tra.inverse_matrix(T_center)
            grasps_out = np.stack(
                [T_inv @ np.asarray(g) for g in grasps_c_free], axis=0
            ).astype(np.float32)
            conf_out = conf_free.astype(np.float32)
        else:
            grasps_out = grasps_np.astype(np.float32)
            conf_out = conf_np.astype(np.float32)

        # --- Approach direction filter ---
        num_before_approach = len(grasps_out)
        if filter_approach and len(grasps_out) > 0:
            keep = np.ones(len(grasps_out), dtype=bool)
            for i, g in enumerate(grasps_out):
                grasp_approach = g[:3, approach_axis_index].astype(np.float64)
                grasp_approach /= max(np.linalg.norm(grasp_approach), 1e-12)
                if approach_dir is not None:
                    preferred = approach_dir
                elif robot_pos is not None:
                    v = robot_pos - g[:3, 3].astype(np.float64)
                    n = np.linalg.norm(v)
                    preferred = v / max(n, 1e-12)
                else:
                    continue
                cos = float(np.dot(grasp_approach, preferred))
                if cos < approach_cos_threshold:
                    keep[i] = False
            grasps_out = grasps_out[keep]
            conf_out = conf_out[keep]
            logger.info(
                "Approach direction filter: %d -> %d grasps (cos_thresh=%.2f)",
                num_before_approach,
                len(grasps_out),
                approach_cos_threshold,
            )

        # Sort by confidence and keep top-K
        if len(conf_out) > 0:
            order = np.argsort(conf_out)[::-1]
            grasps_out = grasps_out[order]
            conf_out = conf_out[order]
            if len(grasps_out) > output_topk:
                grasps_out = grasps_out[:output_topk]
                conf_out = conf_out[:output_topk]

        logger.info(
            "infer_scene: %d raw -> %d after collision -> %d after approach -> %d topk  (%.1f+%.1f ms)",
            num_before,
            num_before_approach,
            len(grasps_out),
            min(len(grasps_out), output_topk),
            infer_ms,
            collision_ms,
        )

        return {
            "grasps": grasps_out,
            "confidences": conf_out,
            "num_grasps": len(grasps_out),
            "num_grasps_raw": int(num_before),
            "num_grasps_after_collision": int(num_before_approach),
            "num_grasps_after_approach": int(len(grasps_out)),
            "filter_collisions": filter_collisions,
            "filter_approach": filter_approach,
            "collision_threshold_m": collision_threshold if filter_collisions else None,
            "timing": {"infer_ms": infer_ms, "collision_ms": collision_ms},
        }
