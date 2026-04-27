#!/usr/bin/env python3
"""
Печать сопоставления: плоский action (k,) и qpos/observation.state (n,).

Запуск (как в benchmark):
  cd RoboBenchMart && conda activate dsynth
  python scripts/print_action_state_layout.py \\
    --config-dir generated_envs/ds_small_scene_0

Нельзя вставлять в `python - <<'PY'` обрывки вроде `?` из IPython (SyntaxError) и
строки, слипшиеся с `PY` — используйте этот файл.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dsynth.envs import *  # noqa: F401,F403
from dsynth.robots import *  # noqa: F401,F403


def _to_np(x) -> np.ndarray:
    if hasattr(x, "cpu"):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _action_labels_for_uid(uid: str, ctrl) -> str:
    c = ctrl.config
    if hasattr(c, "joint_names") and c.joint_names:
        return ", ".join(c.joint_names)
    if uid == "base":
        return "vel_forward, vel_yaw ( PDBaseForwardVel; не имена суставов )"
    return "?"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config-dir",
        type=Path,
        default=_ROOT / "generated_envs/ds_small_scene_0",
        help="Каталог сцены (config_dir_path в gym.make).",
    )
    p.add_argument("--env-id", type=str, default="PickToBasketContActorEvalEnv")
    p.add_argument("--robot-uids", type=str, default="ds_fetch_basket")
    p.add_argument(
        "--obs-mode",
        type=str,
        default="state",
        help="state — минимальный obs (для qpos), без тяжёлых сенсоров.",
    )
    args = p.parse_args()

    if not args.config_dir.is_dir():
        raise SystemExit(f"Нет каталога: {args.config_dir}")

    env = gym.make(
        args.env_id,
        config_dir_path=str(args.config_dir),
        num_envs=1,
        robot_uids=args.robot_uids,
        control_mode="pd_joint_pos",
        obs_mode=args.obs_mode,
        render_mode=None,
        sim_backend="auto",
    )
    try:
        obs, _ = env.reset(seed=0, options={"reconfigure": True})
        u = env.unwrapped
        ag = u.agent
        comb = ag.controller

        print("=== action (flat) ===\n")
        a_space = comb.action_space
        print(f"shape: {a_space.shape}, space: {a_space}\n")
        for uid, controller in comb.controllers.items():
            a0, a1 = comb.action_mapping[uid]
            n = a1 - a0
            jlab = _action_labels_for_uid(uid, controller)
            print(f"  [{a0:2d}:{a1:2d}]  n={n}  uid={uid!r}  {jlab}")

        print("\n=== state / qpos (per active joint) ===\n")
        if isinstance(obs, dict) and "agent" in obs and "qpos" in obs["agent"]:
            q = _to_np(obs["agent"]["qpos"])
        else:
            # fallback: прямая поза артикуляции
            q = _to_np(ag.robot.get_qpos())
        if q.ndim == 2:
            q = q[0]
        anames = [j.name for j in ag.robot.active_joints]
        if q.shape[0] != len(anames):
            print(
                f"qpos len {q.shape[0]} != active_joints {len(anames)} — проверьте obs/робота."
            )
        for i, name in enumerate(anames):
            v = q[i] if i < q.shape[0] else float("nan")
            print(f"  state[{i:2d}]  {name!s:40s}  {float(v):.4f}")
        print(f"\n( observation.state в LeRobot = эти {q.shape[0]} qpos, см. convert_to_lerobot )")
    finally:
        env.close()


if __name__ == "__main__":
    main()
