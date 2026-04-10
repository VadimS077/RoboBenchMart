import pytest
import torch
import numpy as np
import trimesh
from pathlib import Path
from grasp_gen.robot import (
    GripperInfo,
    get_canonical_gripper_control_points,
    generate_circle_points,
    load_visualize_control_points_multi_suction,
    parse_offset_transform_from_yaml,
    load_gripper_yaml_file,
    load_default_gripper_config,
    get_gripper_info,
    get_gripper_depth,
)


def test_canonical_gripper_control_points():
    """Test generation of canonical gripper control points."""
    w = 0.1  # width
    d = 0.05  # depth

    control_points = get_canonical_gripper_control_points(w, d)

    # Check shape
    assert control_points.shape == (4, 3)

    # Check points are in correct positions
    expected_points = np.array(
        [
            [w / 2, 0, d / 2],  # right_front
            [-w / 2, 0, d / 2],  # left_front
            [w / 2, 0, d],  # right_back
            [-w / 2, 0, d],  # left_back
        ]
    )
    assert np.allclose(control_points, expected_points)


def test_load_visualize_control_points_multi_suction():
    """Test generation of visualization points for multiple suction cups."""
    # Create test suction center points
    suction_centers = [[0.0, 0.0, 0.1], [0.1, 0.1, 0.1], [-0.1, -0.1, 0.1]]

    points = load_visualize_control_points_multi_suction(suction_centers)

    # Check shape (N suction cups, M points per cup, 3 coordinates)
    assert points.shape[0] == len(suction_centers)
    assert points.shape[2] == 3

    # Check z-coordinates are all the same
    z_coords = points[:, :, 2]
    assert np.allclose(z_coords, 0.1)

    # Check points form circles around centers
    for i, center in enumerate(suction_centers):
        cup_points = points[i]
        distances = np.sqrt(np.sum((cup_points[:, :2] - center[:2]) ** 2, axis=1))
        assert np.allclose(distances, 0.005, atol=1e-6)  # radius=0.005


def test_parse_offset_transform_from_yaml():
    """Test parsing of offset transform from YAML format."""
    # Create a sample transform in YAML format: [translation (xyz), quaternion (xyzw)]
    yaml_transform = [
        [0.1, 0.2, 0.3],  # translation (x, y, z)
        [0.0, 0.0, 0.0, 1.0],  # quaternion (x, y, z, w) - identity rotation
    ]

    transform = parse_offset_transform_from_yaml(yaml_transform)

    # Check shape
    assert transform.shape == (4, 4)

    # Check values - should be identity rotation with translation
    expected_transform = np.array(
        [
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.0, 0.2],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(transform, expected_transform)


def test_gripper_info_dataclass():
    """Test GripperInfo dataclass initialization and attributes."""
    # Create sample meshes
    collision_mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    visual_mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    offset_transform = np.eye(4)
    control_points = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])

    # Create GripperInfo instance
    gripper_info = GripperInfo(
        gripper_name="test_gripper",
        collision_mesh=collision_mesh,
        visual_mesh=visual_mesh,
        offset_transform=offset_transform,
        control_points=control_points,
        depth=0.1,
        symmetric=True,
    )

    # Check attributes
    assert gripper_info.gripper_name == "test_gripper"
    assert isinstance(gripper_info.collision_mesh, trimesh.base.Trimesh)
    assert isinstance(gripper_info.visual_mesh, trimesh.base.Trimesh)
    assert np.allclose(gripper_info.offset_transform, offset_transform)
    assert np.allclose(gripper_info.control_points, control_points)
    assert gripper_info.depth == 0.1
    assert gripper_info.symmetric == True


@pytest.mark.parametrize("width,depth", [(0.1, 0.05), (0.2, 0.1), (0.05, 0.02)])
def test_canonical_gripper_control_points_parameterized(width, depth):
    """Test canonical gripper control points with different parameters."""
    control_points = get_canonical_gripper_control_points(width, depth)

    # Check shape
    assert control_points.shape == (4, 3)

    # Check points are in correct positions
    expected_points = np.array(
        [
            [width / 2, 0, depth / 2],  # right_front
            [-width / 2, 0, depth / 2],  # left_front
            [width / 2, 0, depth],  # right_back
            [-width / 2, 0, depth],  # left_back
        ]
    )
    assert np.allclose(control_points, expected_points)


def test_load_gripper_yaml_file(tmp_path):
    """Test loading gripper configuration from YAML file."""
    # Create a temporary YAML file
    yaml_content = """
    gripper_name: test_gripper
    width: 0.1
    depth: 0.05
    symmetric: true
    offset_transform:
        - [1.0, 0.0, 0.0, 0.1]
        - [0.0, 1.0, 0.0, 0.2]
        - [0.0, 0.0, 1.0, 0.3]
        - [0.0, 0.0, 0.0, 1.0]
    """
    yaml_file = tmp_path / "test_gripper.yaml"
    yaml_file.write_text(yaml_content)

    # Load the configuration
    config = load_gripper_yaml_file(yaml_file)

    # Check configuration values
    assert config["gripper_name"] == "test_gripper"
    assert config["width"] == 0.1
    assert config["depth"] == 0.05
    assert config["symmetric"] == True
    assert len(config["offset_transform"]) == 4


def test_load_default_gripper_config_franka_panda():
    """Test loading default gripper configuration."""
    # Test with a known gripper name
    config = load_default_gripper_config("franka_panda")

    assert "width" in config
    assert "depth" in config
    assert "transform_offset_from_asset_to_graspgen_convention" in config


def test_get_gripper_info():
    """Test getting gripper information."""
    # Test with a known gripper name
    gripper_info = get_gripper_info("franka_panda")

    # Check that it returns a GripperInfo instance
    assert isinstance(gripper_info, GripperInfo)

    # Check essential attributes
    assert gripper_info.gripper_name == "franka_panda"
    assert isinstance(gripper_info.collision_mesh, trimesh.base.Trimesh)
    assert isinstance(gripper_info.visual_mesh, trimesh.base.Trimesh)
    assert isinstance(gripper_info.control_points, torch.Tensor)


def test_get_gripper_depth():
    """Test getting gripper depth."""
    # Test with a known gripper name
    depth = get_gripper_depth("franka_panda")

    # Check that depth is a positive number
    assert isinstance(depth, float)
    assert depth > 0


@pytest.mark.parametrize("gripper_name", ["franka_panda", "robotiq_2f_140"])
def test_gripper_info_consistency(gripper_name):
    """Test consistency of gripper information across different grippers."""
    gripper_info = get_gripper_info(gripper_name)

    # Check that all required attributes are present
    assert hasattr(gripper_info, "gripper_name")
    assert hasattr(gripper_info, "collision_mesh")
    assert hasattr(gripper_info, "visual_mesh")
    assert hasattr(gripper_info, "control_points")
    assert hasattr(gripper_info, "offset_transform")
    assert hasattr(gripper_info, "transform_from_base_link_to_tool_tcp")

    # Check that the transform from base link to tool TCP is a 4x4 matrix
    assert gripper_info.transform_from_base_link_to_tool_tcp.shape == (4, 4)

    # Check that the offset transform is a 4x4 matrix
    assert gripper_info.offset_transform.shape == (4, 4)

    # Check that meshes are valid
    assert gripper_info.collision_mesh.is_watertight
    assert gripper_info.visual_mesh.is_watertight
