import pytest
import torch
import numpy as np
from grasp_gen.utils.point_cloud_utils import knn_points, point_cloud_outlier_removal


def test_knn_points_basic():
    # Create a simple point cloud with known distances
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )

    K = 2
    dists, idxs = knn_points(points, K=K, norm=2)

    # Check shapes
    assert dists.shape == (5, K)
    assert idxs.shape == (5, K)

    # Check that distances are sorted
    assert torch.all(dists[:, 0] <= dists[:, 1])

    # Check that self-distances are not included
    for i in range(5):
        assert i not in idxs[i]


def test_knn_points_identical_points():
    # Test with identical points
    points = torch.ones((5, 3))  # All points at (1,1,1)
    K = 2

    dists, idxs = knn_points(points, K=K, norm=2)

    # All distances should be 0 since points are identical
    assert torch.allclose(dists, torch.zeros(5, K))

    # Indices should be different from self
    for i in range(5):
        assert i not in idxs[i]


def test_point_cloud_outlier_removal():
    # Create a point cloud with some obvious outliers
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
            [10.0, 10.0, 10.0],  # Obvious outlier
            [20.0, 20.0, 20.0],  # Another obvious outlier
        ]
    )

    filtered_pc, removed_pc = point_cloud_outlier_removal(points, threshold=0.5, K=3)

    # Check that outliers were removed
    assert filtered_pc.shape[0] < points.shape[0]
    assert removed_pc.shape[0] > 0

    # Check that the filtered points are within reasonable distance
    center = torch.mean(filtered_pc, dim=0)
    max_dist = torch.max(torch.norm(filtered_pc - center, dim=1))
    assert max_dist < 1.0  # All points should be within 1 unit of center


def test_point_cloud_outlier_removal_no_outliers():
    # Create a dense point cloud with no obvious outliers
    points = torch.randn(100, 3) * 0.1  # Random points close to origin

    filtered_pc, removed_pc = point_cloud_outlier_removal(points, threshold=0.5, K=10)

    # Most points should be kept
    assert filtered_pc.shape[0] > points.shape[0] * 0.9
    assert removed_pc.shape[0] < points.shape[0] * 0.1


def test_point_cloud_outlier_removal_identical_points():
    # Create a point cloud where all points are identical
    identical_point = torch.tensor([1.0, 2.0, 3.0])
    points = identical_point.repeat(50, 1)  # 50 identical points

    filtered_pc, removed_pc = point_cloud_outlier_removal(points, threshold=0.5, K=10)

    # All points should be kept since they are identical (no outliers)
    assert filtered_pc.shape[0] == points.shape[0]
    assert removed_pc.shape[0] == 0

    # All filtered points should still be identical to the original
    assert torch.allclose(filtered_pc, points)
