import pytest
import json
import os
import numpy as np
from pathlib import Path
from grasp_gen.dataset.webdataset_utils import is_webdataset, GraspWebDatasetReader
from grasp_gen.dataset.dataset_utils import GraspJsonDatasetReader


def test_dataset_format_convention():
    """Test that the dataset follows the GraspGen dataset convention."""

    # Check if datasets are mounted
    object_dataset_path = "/object_dataset"
    grasp_dataset_path = "/grasp_dataset"

    if not os.path.exists(object_dataset_path):
        pytest.skip(f"Object dataset not mounted at {object_dataset_path}")

    if not os.path.exists(grasp_dataset_path):
        pytest.skip(f"Grasp dataset not mounted at {grasp_dataset_path}")

    # Test 1: Check splits exist (for franka_panda gripper)
    splits_path = os.path.join(grasp_dataset_path, "splits", "franka_panda")
    train_split_path = os.path.join(splits_path, "train.txt")
    valid_split_path = os.path.join(splits_path, "valid.txt")

    assert os.path.exists(splits_path), f"Splits directory not found at {splits_path}"
    assert os.path.exists(
        train_split_path
    ), f"train.txt not found at {train_split_path}"
    assert os.path.exists(
        valid_split_path
    ), f"valid.txt not found at {valid_split_path}"

    # Test 2: Check splits are not empty
    with open(train_split_path, "r") as f:
        train_objects = [line.strip() for line in f.readlines() if line.strip()]

    with open(valid_split_path, "r") as f:
        valid_objects = [line.strip() for line in f.readlines() if line.strip()]

    assert len(train_objects) > 0, "train.txt is empty"
    assert len(valid_objects) > 0, "valid.txt is empty"

    # Test 4: Check grasp dataset format and initialize appropriate reader
    grasp_data_path = os.path.join(grasp_dataset_path, "grasp_data", "franka_panda")
    assert os.path.exists(
        grasp_data_path
    ), f"Grasp data directory not found at {grasp_data_path}"

    # Determine if this is a WebDataset or JSON dataset
    if is_webdataset(grasp_data_path):
        print(f"Detected WebDataset format in {grasp_data_path}")

        # Check for WebDataset format: tar shards and uuid_index.json
        tar_shards = list(Path(grasp_data_path).glob("shard_*.tar"))
        uuid_index_path = os.path.join(grasp_data_path, "uuid_index.json")

        assert len(tar_shards) > 0, f"No tar shards found in {grasp_data_path}"
        assert os.path.exists(
            uuid_index_path
        ), f"uuid_index.json not found at {uuid_index_path}"

        # Load and validate uuid_index.json
        with open(uuid_index_path, "r") as f:
            uuid_index = json.load(f)

        assert isinstance(uuid_index, dict), "uuid_index.json should be a dictionary"
        assert len(uuid_index) > 0, "uuid_index.json is empty"

        # Initialize WebDataset reader
        grasp_dataset_reader = GraspWebDatasetReader(grasp_data_path)

    else:
        print(f"Detected JSON dataset format in {grasp_data_path}")

        # Check for JSON files
        json_files = list(Path(grasp_data_path).glob("*.json"))
        assert len(json_files) > 0, f"No JSON files found in {grasp_data_path}"

        # Initialize JSON dataset reader
        grasp_dataset_reader = GraspJsonDatasetReader(grasp_data_path)

    # Test 5: Validate grasp data structure using the appropriate reader
    # Sample a few objects from the splits to validate
    sample_objects = (
        train_objects[:3] + valid_objects[:3]
    )  # Test first 3 from each split

    for obj_id in sample_objects:
        try:
            # Use the appropriate reader to get grasp data
            grasps_dict = grasp_dataset_reader.read_grasps_by_uuid(obj_id)

            if grasps_dict is None:
                print(f"Warning: No grasp data found for object {obj_id}")
                continue

            # Check required top-level keys
            assert "object" in grasps_dict, f"Missing 'object' key for object {obj_id}"
            assert "grasps" in grasps_dict, f"Missing 'grasps' key for object {obj_id}"

            # Check object section
            object_section = grasps_dict["object"]
            assert (
                "file" in object_section
            ), f"Missing 'file' key in object section for {obj_id}"
            assert (
                "scale" in object_section
            ), f"Missing 'scale' key in object section for {obj_id}"

            # Check grasps section
            grasps_section = grasps_dict["grasps"]
            assert (
                "transforms" in grasps_section
            ), f"Missing 'transforms' key in grasps section for {obj_id}"
            assert (
                "object_in_gripper" in grasps_section
            ), f"Missing 'object_in_gripper' key in grasps section for {obj_id}"

            # Check data types and shapes
            transforms = np.array(grasps_section["transforms"])
            grasp_mask = np.array(grasps_section["object_in_gripper"])

            # Validate transforms are 4x4 matrices
            assert transforms.ndim == 3, f"Transforms should be 3D array for {obj_id}"
            assert transforms.shape[1:] == (
                4,
                4,
            ), f"Transforms should be 4x4 matrices for {obj_id}"

            # Validate mask length matches number of grasps
            assert len(grasp_mask) == len(
                transforms
            ), f"Mask length doesn't match number of grasps for {obj_id}"

            # Validate mask contains only boolean values
            assert grasp_mask.dtype == bool or np.all(
                np.isin(grasp_mask, [0, 1])
            ), f"Mask should be boolean for {obj_id}"

            # # Check that referenced object file exists (if it's a mesh file path)
            # object_file = object_section["file"]
            # if object_file.lower().endswith(('.obj', '.stl')):
            #     full_object_path = os.path.join(object_dataset_path, object_file)
            #     assert os.path.exists(full_object_path), f"Referenced object file not found: {full_object_path}"

            # Check scale is a positive number
            scale = object_section["scale"]
            assert (
                isinstance(scale, (int, float)) and scale > 0
            ), f"Scale should be positive number for {obj_id}"

        except Exception as e:
            pytest.fail(f"Error processing object {obj_id}: {e}")

    # Test 6: Check that we can extract positive and negative grasps
    for obj_id in sample_objects:
        grasps_dict = grasp_dataset_reader.read_grasps_by_uuid(obj_id)

        if grasps_dict is None:
            continue

        grasps = np.array(grasps_dict["grasps"]["transforms"])
        grasp_mask = np.array(grasps_dict["grasps"]["object_in_gripper"])

        positive_grasps = grasps[grasp_mask]
        negative_grasps = grasps[~grasp_mask]

        # Check that we have some grasps (at least one positive or negative)
        assert (
            len(positive_grasps) > 0 or len(negative_grasps) > 0
        ), f"No grasps found for {obj_id}"

        # Check that positive grasps are valid 4x4 matrices
        for grasp in positive_grasps:
            assert grasp.shape == (
                4,
                4,
            ), f"Positive grasp should be 4x4 matrix for {obj_id}"
            # Check that it's a valid transformation matrix (bottom row should be [0,0,0,1])
            assert np.allclose(
                grasp[3, :], [0, 0, 0, 1]
            ), f"Invalid transformation matrix for {obj_id}"

        # Check that negative grasps are valid 4x4 matrices
        for grasp in negative_grasps:
            assert grasp.shape == (
                4,
                4,
            ), f"Negative grasp should be 4x4 matrix for {obj_id}"
            # Check that it's a valid transformation matrix (bottom row should be [0,0,0,1])
            assert np.allclose(
                grasp[3, :], [0, 0, 0, 1]
            ), f"Invalid transformation matrix for {obj_id}"

    print(f"✅ Dataset format validation passed!")
    print(
        f"   - Object dataset: {len(train_objects)} train objects, {len(valid_objects)} valid objects"
    )
    print(
        f"   - Grasp dataset: {'WebDataset' if is_webdataset(grasp_data_path) else 'JSON'} format"
    )
    print(f"   - All files follow GraspGen convention")
