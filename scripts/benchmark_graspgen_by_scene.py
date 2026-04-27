#!/usr/bin/env python3
"""
Sequential wrapper around benchmark_graspgen.py:
run scene1 -> scene2 -> scene3 with per-scene product lists.

Example:
  python scripts/benchmark_graspgen_by_scene.py \
      --scenes-root generated_envs \
      --scene-products first:avias,salt \
      --scene-products second:pepsi,milk \
      --scene-products third:oreo,monster,vanish \
      --num-traj 100 \
      --only-count-success \
      --save-traj
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parents[1]
BENCH_SCRIPT = ROOT_DIR / "scripts" / "benchmark_graspgen.py"

# Friendly aliases requested by user.
SCENE_ALIAS_TO_DIR = {
    "first": "ds_small_scene_1",
    "second": "ds_small_scene_2",
    "third": "ds_small_scene_3",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run benchmark_graspgen.py sequentially for scene-specific product pools."
    )
    p.add_argument(
        "--scenes-root",
        type=Path,
        default=ROOT_DIR / "generated_envs",
        help="Directory containing ds_small_scene_* folders",
    )
    p.add_argument(
        "--scene-products",
        action="append",
        required=True,
        help=(
            "Scene->products mapping. Format: <scene_key>:<p1,p2,...>. "
            "scene_key can be first/second/third or scene dir name "
            "(e.g. ds_small_scene_2:pepsi,milk). Repeat flag for each scene."
        ),
    )
    p.add_argument(
        "--scene-order",
        nargs="+",
        default=["first", "second", "third"],
        help=(
            "Execution order (default: first second third). "
            "Keys can be aliases (first/second/third) or scene dir names."
        ),
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for merged wrapper summary JSON.",
    )

    # Anything unknown gets forwarded to benchmark_graspgen.py
    args, forward = p.parse_known_args()
    args.forward_args = forward
    return args


def _resolve_scene_key(scene_key: str) -> str:
    key = scene_key.strip()
    return SCENE_ALIAS_TO_DIR.get(key, key)


def _parse_scene_products(entries: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for item in entries:
        if ":" not in item:
            raise ValueError(
                f"Invalid --scene-products value {item!r}. Expected '<scene>:<p1,p2,...>'."
            )
        scene_key, raw_products = item.split(":", 1)
        scene_name = _resolve_scene_key(scene_key.strip())
        products = [p.strip() for p in raw_products.split(",") if p.strip()]
        if not products:
            raise ValueError(
                f"No products provided in {item!r}. Expected at least one product slug."
            )
        out[scene_name] = products
    return out


def _run_one_scene(
    scenes_root: Path,
    scene_name: str,
    products: List[str],
    forward_args: List[str],
) -> Dict[str, object]:
    src_scene = scenes_root / scene_name
    if not src_scene.is_dir():
        raise FileNotFoundError(f"Scene directory not found: {src_scene}")

    cmd = [
        sys.executable,
        str(BENCH_SCRIPT),
        "--scenes-root",
        str(scenes_root),
        "--scene-names",
        scene_name,
        "--products",
        *products,
        *forward_args,
    ]
    print("\n" + "=" * 80)
    print(f"[Wrapper] Scene: {scene_name}")
    print(f"[Wrapper] Products: {products}")
    print(f"[Wrapper] Command: {' '.join(cmd)}")
    print("=" * 80)

    run = subprocess.run(cmd, check=False)

    per_scene_result_path = scenes_root / "benchmark_results.json"
    if not per_scene_result_path.exists():
        raise FileNotFoundError(
            f"benchmark_results.json was not created for scene {scene_name}"
        )
    payload = json.loads(per_scene_result_path.read_text(encoding="utf-8"))
    scene_payload = payload.get(scene_name)

    # Some runs finish and write valid results but crash during native teardown
    # (renderer/physics finalization). In that case keep going.
    if run.returncode != 0:
        if scene_payload is not None:
            print(
                f"[Wrapper] WARNING: benchmark exited with code {run.returncode} "
                f"for scene {scene_name}, but results exist. Continuing."
            )
        else:
            raise subprocess.CalledProcessError(run.returncode, cmd)

    return scene_payload or {}


def main() -> None:
    args = parse_args()
    scenes_root = args.scenes_root.resolve()
    if not scenes_root.is_dir():
        raise FileNotFoundError(f"scenes-root not found: {scenes_root}")

    scene_products = _parse_scene_products(args.scene_products)
    run_order = [_resolve_scene_key(k) for k in args.scene_order]

    missing = [s for s in run_order if s not in scene_products]
    if missing:
        raise ValueError(
            "Missing --scene-products for scenes in --scene-order: " + ", ".join(missing)
        )

    merged: Dict[str, object] = {}
    for scene_name in run_order:
        merged[scene_name] = _run_one_scene(
            scenes_root=scenes_root,
            scene_name=scene_name,
            products=scene_products[scene_name],
            forward_args=args.forward_args,
        )

    if args.output_json is not None:
        out_path = args.output_json.resolve()
    else:
        out_path = scenes_root / "benchmark_results_sequential_by_scene.json"
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Wrapper] Saved merged results -> {out_path}")


if __name__ == "__main__":
    main()
