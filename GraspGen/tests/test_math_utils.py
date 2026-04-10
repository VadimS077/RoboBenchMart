import pytest
import torch
import numpy as np
from grasp_gen.utils.math_utils import (
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
    compute_pose_distance_batch,
    compute_pose_emd,
)
from grasp_gen.utils.so3 import so3_log_map, so3_exp_map


def test_rotation_so3_conversion():
    # Test conversion between SO(3) rotation representation and matrix
    # Create a random SO(3) log representation (3D vector) with batch dimension
    so3_log = torch.randn(1, 3)
    so3_log = (
        so3_log / torch.norm(so3_log, dim=1, keepdim=True) * torch.pi * 0.5
    )  # Scale to reasonable rotation angle

    # Convert to matrix and back
    matrix = so3_exp_map(so3_log)
    so3_log_recovered = so3_log_map(matrix)

    # Check if the recovered SO(3) representation matches the original
    assert torch.allclose(so3_log, so3_log_recovered, atol=1e-6)


def test_rotation_so3_batch():
    # Test batch processing of SO(3) rotation conversion
    batch_size = 5
    so3_log_batch = torch.randn(batch_size, 3)
    so3_log_batch = (
        so3_log_batch / torch.norm(so3_log_batch, dim=1, keepdim=True) * torch.pi * 0.5
    )

    matrix_batch = so3_exp_map(so3_log_batch)
    so3_log_batch_recovered = so3_log_map(matrix_batch)

    assert torch.allclose(so3_log_batch, so3_log_batch_recovered, atol=1e-6)


def test_compute_pose_distance_batch():
    # Create two sets of poses
    poses1 = torch.eye(4).unsqueeze(0)  # Identity pose
    poses2 = torch.eye(4).unsqueeze(0)  # Same identity pose

    # Compute distance
    distances = compute_pose_distance_batch(poses1, poses2)

    # Distance should be 0 for identical poses
    assert torch.allclose(distances, torch.zeros(1, 1), atol=1e-6)

    # Test with different poses
    poses2 = torch.eye(4).unsqueeze(0)
    poses2[0, :3, 3] = torch.tensor([1.0, 0.0, 0.0])  # Translate by 1 unit in x

    distances = compute_pose_distance_batch(poses1, poses2)
    assert distances[0, 0] > 0  # Should have non-zero distance


def test_compute_pose_emd():
    # Create two sets of identical poses
    poses1 = torch.eye(4).unsqueeze(0)
    poses2 = torch.eye(4).unsqueeze(0)

    # EMD should be 0 for identical sets
    emd = compute_pose_emd(poses1, poses2)
    assert abs(emd) < 1e-6

    # Test with different poses
    poses2 = torch.eye(4).unsqueeze(0)
    poses2[0, :3, 3] = torch.tensor([1.0, 0.0, 0.0])

    emd = compute_pose_emd(poses1, poses2)
    assert emd > 0  # Should have non-zero EMD


def test_rotation_orthogonality():
    # Test that rotation matrices are orthogonal
    d6 = torch.randn(6)
    d6 = d6 / torch.norm(d6)

    matrix = rotation_6d_to_matrix(d6)

    # Check orthogonality: R * R^T should be identity
    identity = torch.eye(3)
    assert torch.allclose(torch.matmul(matrix, matrix.T), identity, atol=1e-6)
