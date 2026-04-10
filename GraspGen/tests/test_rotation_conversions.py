import pytest
import torch
import numpy as np
from grasp_gen.utils.rotation_conversions import (
    matrix_to_euler_angles,
    euler_angles_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
)


def test_euler_matrix_conversion(sample_rotation_matrix):
    """Test conversion between Euler angles and rotation matrix."""
    # Convert matrix to Euler angles
    euler = matrix_to_euler_angles(sample_rotation_matrix, convention="XYZ")

    # Convert back to matrix
    matrix_recovered = euler_angles_to_matrix(euler, convention="XYZ")

    # Check if the recovered matrix matches the original
    assert torch.allclose(sample_rotation_matrix, matrix_recovered, atol=1e-6)


def test_quaternion_matrix_conversion(sample_rotation_matrix):
    """Test conversion between quaternion and rotation matrix."""
    # Convert matrix to quaternion
    quat = matrix_to_quaternion(sample_rotation_matrix)

    # Convert back to matrix
    matrix_recovered = quaternion_to_matrix(quat)

    # Check if the recovered matrix matches the original
    assert torch.allclose(sample_rotation_matrix, matrix_recovered, atol=1e-6)


def test_euler_angles_consistency():
    """Test consistency of Euler angle conversions with different conventions."""
    # Create a rotation matrix
    matrix = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    # Convert to Euler angles with different conventions
    euler_xyz = matrix_to_euler_angles(matrix, convention="XYZ")
    euler_zyx = matrix_to_euler_angles(matrix, convention="ZYX")

    # Convert back to matrix
    matrix_xyz = euler_angles_to_matrix(euler_xyz, convention="XYZ")
    matrix_zyx = euler_angles_to_matrix(euler_zyx, convention="ZYX")

    # All should give the same rotation matrix
    assert torch.allclose(matrix, matrix_xyz, atol=1e-6)
    assert torch.allclose(matrix, matrix_zyx, atol=1e-6)


def test_rotation_identity():
    """Test that identity rotation is preserved through conversions."""
    # Identity rotation matrix
    identity = torch.eye(3)

    # Convert to Euler angles and back
    euler = matrix_to_euler_angles(identity, convention="XYZ")
    matrix_euler = euler_angles_to_matrix(euler, convention="XYZ")

    # Convert to quaternion and back
    quat = matrix_to_quaternion(identity)
    matrix_quat = quaternion_to_matrix(quat)

    # All should give identity matrix
    assert torch.allclose(identity, matrix_euler, atol=1e-6)
    assert torch.allclose(identity, matrix_quat, atol=1e-6)


def test_rotation_composition():
    """Test that rotation compositions work correctly."""
    # Create two rotation matrices
    matrix1 = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )  # 90 degrees around z

    matrix2 = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )  # 90 degrees around x

    # Compose rotations
    composed = torch.matmul(matrix2, matrix1)

    # Convert to Euler angles
    euler = matrix_to_euler_angles(composed, convention="XYZ")

    # Convert back to matrix
    matrix_recovered = euler_angles_to_matrix(euler, convention="XYZ")

    # Check if composition is preserved
    assert torch.allclose(composed, matrix_recovered, atol=1e-6)
