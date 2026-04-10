# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Tutorial: Create Model Checkpoints for Inference

This script copies the trained generator and discriminator checkpoints to a models directory
for easy access during inference. It organizes the checkpoints with standardized names and
creates a config file with updated checkpoint paths.

Usage:
    python create_model_checkpoints_for_inference.py \
        --gen_log_dir /results/tutorial/logs/single_suction_cup_30mm_gen_test \
        --dis_log_dir /results/tutorial/logs/single_suction_cup_30mm_dis_test \
        --models_dir /results/tutorial/models
"""

import os
import shutil
import argparse
import yaml
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create model checkpoints for inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gen_log_dir",
        type=str,
        required=True,
        help="Path to generator training log directory",
    )
    parser.add_argument(
        "--dis_log_dir",
        type=str,
        required=True,
        help="Path to discriminator training log directory",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="/results/tutorial/models",
        help="Output directory for model checkpoints and config",
    )

    return parser.parse_args()


def copy_checkpoint(src_path, dst_path, checkpoint_name):
    """Copy a checkpoint file from source to destination."""
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Checkpoint not found: {src_path}")

    # Create destination directory if it doesn't exist
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # Copy the file
    shutil.copy2(src_path, dst_path)
    print(f"Copied {checkpoint_name} checkpoint:")
    print(f"  From: {src_path}")
    print(f"  To: {dst_path}")


def create_config_file(gen_log_dir, dis_log_dir, models_dir):
    """Create config file with updated checkpoint paths and generator diffusion settings."""
    # Source config file paths
    gen_config_path = os.path.join(gen_log_dir, "config.yaml")
    dis_config_path = os.path.join(dis_log_dir, "config.yaml")

    if not os.path.exists(gen_config_path):
        raise FileNotFoundError(f"Generator config file not found: {gen_config_path}")

    if not os.path.exists(dis_config_path):
        raise FileNotFoundError(
            f"Discriminator config file not found: {dis_config_path}"
        )

    # Destination config file path
    dst_config_path = os.path.join(models_dir, "tutorial_model_config.yaml")

    # Read the generator config to get diffusion settings
    with open(gen_config_path, "r") as f:
        gen_config = yaml.safe_load(f)

    # Read the discriminator config (base config)
    with open(dis_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Copy diffusion settings from generator config
    if "diffusion" in gen_config:
        config["diffusion"] = gen_config["diffusion"]
        print(f"Copied diffusion settings from generator config")
    else:
        print(f"Warning: No diffusion settings found in generator config")

    # Override the checkpoint paths
    if "eval" not in config:
        config["eval"] = {}
    config["eval"]["checkpoint"] = "gen.pth"
    config["eval"]["model_name"] = "diffusion-discriminator"

    if "discriminator" not in config:
        config["discriminator"] = {}
    config["discriminator"]["checkpoint"] = "dis.pth"

    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    # Write the updated config
    with open(dst_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Created config file:")
    print(f"  Generator config: {gen_config_path}")
    print(f"  Discriminator config: {dis_config_path}")
    print(f"  Output: {dst_config_path}")
    print(f"  Updated: eval.checkpoint = 'gen.pth'")
    print(f"  Updated: discriminator.checkpoint = 'dis.pth'")
    print(f"  Updated: diffusion settings from generator config")


def main():
    """Main function to create model checkpoints for inference."""
    args = parse_args()

    # Define source and destination paths
    gen_src_path = os.path.join(args.gen_log_dir, "last.pth")
    dis_src_path = os.path.join(args.dis_log_dir, "last.pth")

    gen_dst_path = os.path.join(args.models_dir, "gen.pth")
    dis_dst_path = os.path.join(args.models_dir, "dis.pth")

    print("Creating model checkpoints for inference...")
    print(f"Models directory: {args.models_dir}")
    print()

    # Copy generator checkpoint
    try:
        copy_checkpoint(gen_src_path, gen_dst_path, "Generator")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Make sure the generator training has completed successfully.")
        print()

    # Copy discriminator checkpoint
    try:
        copy_checkpoint(dis_src_path, dis_dst_path, "Discriminator")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Make sure the discriminator training has completed successfully.")
        print()

    # Create config file
    try:
        create_config_file(args.gen_log_dir, args.dis_log_dir, args.models_dir)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Make sure the discriminator training has completed successfully.")
        print()

    # Check if all files exist
    gen_exists = os.path.exists(gen_dst_path)
    dis_exists = os.path.exists(dis_dst_path)
    config_exists = os.path.exists(
        os.path.join(args.models_dir, "tutorial_model_config.yaml")
    )

    print("Summary:")
    print(f"  Generator checkpoint: {'✓' if gen_exists else '✗'}")
    print(f"  Discriminator checkpoint: {'✓' if dis_exists else '✗'}")
    print(f"  Config file: {'✓' if config_exists else '✗'}")

    if gen_exists and dis_exists and config_exists:
        print("\n✓ All files are ready for inference!")
        print(f"Models directory: {args.models_dir}")
        print("Files:")
        print(f"  - gen.pth (Generator)")
        print(f"  - dis.pth (Discriminator)")
        print(f"  - tutorial_model_config.yaml (Config with updated paths)")
    else:
        print("\n⚠ Some files are missing. Please ensure training has completed.")

    print("\nNext steps:")
    print("1. Use gen.pth for grasp generation inference")
    print("2. Use dis.pth for grasp quality evaluation")
    print("3. Use tutorial_model_config.yaml for inference configuration")
    print("4. All models and config are ready for deployment")


if __name__ == "__main__":
    main()
