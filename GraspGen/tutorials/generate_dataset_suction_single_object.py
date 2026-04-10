# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Tutorial: Generate Dataset for Single Object Suction Grasp Training

This script demonstrates how to generate a dataset for training a generator and discriminator
model for a single object using a suction cup gripper. The dataset follows the GraspGen
convention and can be used for training neural networks to predict grasp poses.

Usage:
    python generate_dataset_suction_single_object.py \
        --object_path /path/to/object.obj \
        --output_dir /results \
        --num_grasps 2000
"""

import os
import sys
import json
import argparse
import numpy as np
import trimesh
import trimesh.transformations as tra
from pathlib import Path

# Add the parent directory to the path to import grasp_gen modules
sys.path.append(str(Path(__file__).parent.parent))

from grasp_gen.dataset.suction import SuctionCupArray


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate dataset for single object suction grasp training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--object_path",
        type=str,
        required=True,
        help="Path to the object mesh file (obj, stl, or ply)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/results/tutorial",
        help="Directory to save the tutorial datasets",
    )
    parser.add_argument(
        "--num_grasps",
        type=int,
        default=2000,
        help="Total number of grasps to generate (positive and negative)",
    )
    parser.add_argument(
        "--object_scale",
        type=float,
        default=1.0,
        help="Scale factor to apply to the object mesh",
    )
    parser.add_argument(
        "--gripper_config",
        type=str,
        default="config/grippers/single_suction_cup_30mm.yaml",
        help="Path to gripper configuration file",
    )
    parser.add_argument(
        "--num_disturbances",
        type=int,
        default=10,
        help="Number of random disturbance samples for evaluation",
    )
    parser.add_argument(
        "--qp_solver",
        type=str,
        default="clarabel",
        choices=("clarabel", "cvxopt", "daqp", "ecos", "osqp", "scs"),
        help="QP solver to use for wrench resistance evaluation",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no_visualization", action="store_true", help="Disable visualization"
    )

    return parser.parse_args()


def setup_directories(output_dir, object_basename):
    """Create the required directory structure for the tutorial datasets."""
    # Create tutorial object dataset directory
    tutorial_object_dir = Path(output_dir) / "tutorial_object_dataset"
    tutorial_object_dir.mkdir(parents=True, exist_ok=True)

    # Create tutorial grasp dataset directory
    tutorial_grasp_dir = Path(output_dir) / "tutorial_grasp_dataset"
    tutorial_grasp_dir.mkdir(parents=True, exist_ok=True)

    return tutorial_object_dir, tutorial_grasp_dir


def copy_object_to_dataset(
    object_path, tutorial_object_dir, object_basename, scale=1.0
):
    """Copy and scale the object mesh to the tutorial dataset directory."""
    # Load the object mesh
    obj_mesh = trimesh.load(object_path)

    # Apply scale if needed
    if scale != 1.0:
        obj_mesh.apply_scale(scale)

    # Save to tutorial object dataset
    output_object_path = tutorial_object_dir / f"{object_basename}.obj"
    obj_mesh.export(str(output_object_path))

    print(f"Object saved to: {output_object_path}")
    return str(output_object_path), obj_mesh


def generate_grasps(obj_mesh, suction_gripper, num_grasps, num_disturbances, qp_solver):
    """Generate and evaluate grasps for the object."""
    print(f"Generating {num_grasps} grasps...")

    # Sample grasps
    points_on_surface, approach_vectors, grasp_transforms = (
        suction_gripper.sample_grasps(obj_mesh=obj_mesh, num_grasps=num_grasps)
    )

    # Evaluate grasps
    print("Evaluating grasps...")
    points, approach_vectors, contact_transforms, sealed, success, in_collision = (
        suction_gripper.evaluate_grasps(
            obj_mesh=obj_mesh,
            points_on_surface=points_on_surface,
            approach_vectors=approach_vectors,
            grasp_transforms=grasp_transforms,
            num_disturbances=num_disturbances,
            qp_solver=qp_solver,
            tqdm_disable=False,
        )
    )

    return points, approach_vectors, contact_transforms, sealed, success, in_collision


def create_grasp_dataset_json(
    object_file,
    object_scale,
    contact_transforms,
    sealed,
    success,
    in_collision,
    tutorial_grasp_dir,
    object_basename,
    obj_center_mass,
):
    """Create the grasp dataset JSON file following GraspGen convention."""

    # Assert that obj_center_mass is a 3D vector
    assert (
        len(obj_center_mass) == 3
    ), f"obj_center_mass should be length 3, got {len(obj_center_mass)}"

    # Determine successful grasps (sealed, high success rate, and collision-free)
    # We'll use a threshold for success rate to create a mix of positive and negative grasps
    success_threshold = 0.5  # Grasps with >50% success rate are considered positive

    # Create object_in_gripper mask
    object_in_gripper = np.logical_and.reduce(
        [
            sealed,  # Must be sealed
            # ~in_collision  # Must not be in collision; TODO - check if the collision mesh is specified correctly
        ]
    )

    # Transform grasps back to original object frame (add back center of mass)
    output_transforms = np.copy(contact_transforms)
    output_transforms[:, :3, 3] += obj_center_mass

    # Create the dataset dictionary
    dataset_dict = {
        "object": {
            "file": object_file,  # Relative path to object in tutorial dataset
            "scale": object_scale,
        },
        "grasps": {
            "transforms": output_transforms.tolist(),
            "object_in_gripper": object_in_gripper.tolist(),
        },
    }

    # Save to JSON file
    output_json_path = tutorial_grasp_dir / f"{object_basename}_grasps.json"
    with open(output_json_path, "w") as f:
        json.dump(dataset_dict, f, indent=2)

    print(f"Grasp dataset saved to: {output_json_path}")
    print(f"Total grasps: {len(object_in_gripper)}")
    print(f"Positive grasps: {sum(object_in_gripper)}")
    print(f"Negative grasps: {sum(~object_in_gripper)}")

    return str(output_json_path)


def create_splits_file(tutorial_object_dir, tutorial_grasp_dir, object_basename):
    """Create train.txt and valid.txt files for the tutorial dataset."""

    # Create train.txt (include the object)
    train_txt_path = tutorial_object_dir / "train.txt"
    with open(train_txt_path, "w") as f:
        f.write(f"{object_basename}.obj\n")

    # Create valid.txt (include the same object for validation)
    valid_txt_path = tutorial_object_dir / "valid.txt"
    with open(valid_txt_path, "w") as f:
        f.write(f"{object_basename}.obj\n")

    print(f"Split files created:")
    print(f"  Train: {train_txt_path}")
    print(f"  Valid: {valid_txt_path}")


def main():
    """Main function to generate the tutorial dataset."""
    args = parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.random_seed)

    # Validate inputs
    if not os.path.exists(args.object_path):
        raise FileNotFoundError(f"Object file not found: {args.object_path}")

    if not os.path.exists(args.gripper_config):
        raise FileNotFoundError(f"Gripper config not found: {args.gripper_config}")

    # Get object basename
    object_basename = Path(args.object_path).stem

    print(f"Generating dataset for object: {object_basename}")
    print(f"Object path: {args.object_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of grasps: {args.num_grasps}")
    print(f"Object scale: {args.object_scale}")

    # Setup directories
    tutorial_object_dir, tutorial_grasp_dir = setup_directories(
        args.output_dir, object_basename
    )
    # Copy object to dataset
    object_file, obj_mesh = copy_object_to_dataset(
        args.object_path, tutorial_object_dir, object_basename, args.object_scale
    )

    # Make object_file a relative path with respect to tutorial_object_dir
    object_file = f"{object_basename}.obj"

    # Center the object at origin
    obj_center_mass = obj_mesh.center_mass
    obj_mesh.apply_translation(-obj_center_mass)
    print(f"Object centered at origin. Original CoM: {obj_center_mass}")

    # Initialize suction gripper
    print(f"Loading gripper configuration: {args.gripper_config}")
    suction_gripper = SuctionCupArray.from_file(fname=args.gripper_config)

    # Generate grasps
    points, approach_vectors, contact_transforms, sealed, success, in_collision = (
        generate_grasps(
            obj_mesh=obj_mesh,
            suction_gripper=suction_gripper,
            num_grasps=args.num_grasps,
            num_disturbances=args.num_disturbances,
            qp_solver=args.qp_solver,
        )
    )

    # Create grasp dataset JSON
    grasp_json_path = create_grasp_dataset_json(
        object_file=object_file,
        object_scale=args.object_scale,
        contact_transforms=contact_transforms,
        sealed=sealed,
        success=success,
        in_collision=in_collision,
        tutorial_grasp_dir=tutorial_grasp_dir,
        object_basename=object_basename,
        obj_center_mass=obj_center_mass,
    )

    # Create split files
    create_splits_file(tutorial_object_dir, tutorial_grasp_dir, object_basename)

    # Optional visualization
    if not args.no_visualization:
        print("\nVisualizing results...")
        from grasp_gen.utils.meshcat_utils import (
            create_visualizer,
            visualize_mesh,
            visualize_pointcloud,
            visualize_grasp,
        )

        vis = create_visualizer()

        # Visualize object
        visualize_mesh(vis, "object", obj_mesh, color=[169, 169, 169])

        # Visualize grasp points
        from grasp_gen.dataset.suction import colorize_for_meshcat

        grasp_colors = colorize_for_meshcat(success)
        visualize_pointcloud(vis, "grasp_points", points, grasp_colors, size=0.005)

        # Visualize top 10 grasps
        combined_criteria = sealed * success
        top_indices = np.argsort(combined_criteria)[-10:][::-1]

        for i, idx in enumerate(top_indices):
            if combined_criteria[idx] > 0:
                color = [255, 0, 0] if i == 0 else [255, 165, 0]
                visualize_grasp(
                    vis,
                    f"top_grasps/grasp_{i:02d}",
                    contact_transforms[idx],
                    color=color,
                    gripper_name="single_suction_cup_30mm",
                )

        print("Visualization complete. Check meshcat browser window.")
        input("Press Enter to continue...")

    print(f"\nTutorial dataset generation complete!")
    print(f"Object dataset: {tutorial_object_dir}")
    print(f"Grasp dataset: {tutorial_grasp_dir}")
    print(f"Ready for training generator and discriminator models!")


if __name__ == "__main__":
    main()
