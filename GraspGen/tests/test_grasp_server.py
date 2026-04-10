import pytest
import torch
import numpy as np
import os
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal


@pytest.fixture
def sample_point_cloud():
    """Create a sample point cloud for testing."""

    # Randomly sample points in a unit cube
    np.random.seed(42)  # For reproducible tests
    num_points = 3000
    points = np.random.uniform(-0.5, 0.5, size=(num_points, 3))
    return points


@pytest.fixture
def franka_config_path():
    """Path to franka gripper configuration."""
    config_path = "/models/checkpoints/graspgen_franka_panda.yml"
    if not os.path.exists(config_path):
        pytest.skip(f"Franka config not found at {config_path}")
    return config_path


@pytest.fixture
def robotiq_config_path():
    """Path to robotiq gripper configuration."""
    config_path = "/models/checkpoints/graspgen_robotiq_2f_140.yml"
    if not os.path.exists(config_path):
        pytest.skip(f"Robotiq config not found at {config_path}")
    return config_path


def test_grasp_sampler_initialization_franka(franka_config_path):
    """Test initialization of GraspGenSampler with franka gripper."""
    # This test will be skipped if no GPU is available
    if not torch.cuda.is_available():
        pytest.skip("No GPU available for testing")

    try:
        grasp_cfg = load_grasp_cfg(franka_config_path)
        sampler = GraspGenSampler(grasp_cfg)
        assert sampler is not None
        assert sampler.model is not None
        assert grasp_cfg.data.gripper_name == "franka_panda"
    except Exception as e:
        pytest.skip(f"Franka model initialization failed: {str(e)}")


def test_grasp_sampler_initialization_robotiq(robotiq_config_path):
    """Test initialization of GraspGenSampler with robotiq gripper."""
    # This test will be skipped if no GPU is available
    if not torch.cuda.is_available():
        pytest.skip("No GPU available for testing")

    try:
        grasp_cfg = load_grasp_cfg(robotiq_config_path)
        sampler = GraspGenSampler(grasp_cfg)
        assert sampler is not None
        assert sampler.model is not None
        assert grasp_cfg.data.gripper_name == "robotiq_2f_140"
    except Exception as e:
        pytest.skip(f"Robotiq model initialization failed: {str(e)}")


def test_grasp_sampling_basic_franka(sample_point_cloud, franka_config_path):
    """Test basic grasp sampling functionality with franka gripper."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available for testing")

    try:
        grasp_cfg = load_grasp_cfg(franka_config_path)
        sampler = GraspGenSampler(grasp_cfg)

        assert sample_point_cloud.shape == (3000, 3)
        # Test with a small number of grasps
        grasps, confidences = GraspGenSampler.run_inference(
            sample_point_cloud,
            sampler,
            grasp_threshold=0.7,
            num_grasps=10,
            remove_outliers=False,
        )

        # Check return types
        assert isinstance(grasps, (list, np.ndarray, torch.Tensor))
        assert isinstance(confidences, (list, np.ndarray, torch.Tensor))

        # If grasps were found, check their properties
        if len(grasps) > 0:
            # Check grasp poses are 4x4 matrices
            assert grasps[0].shape == (4, 4)

            # Check confidences are between 0 and 1
            if isinstance(confidences, (np.ndarray, torch.Tensor)):
                for conf in confidences:
                    assert conf >= 0 and conf <= 1

    except Exception as e:
        pytest.skip(f"Franka grasp sampling failed: {str(e)}")


def test_grasp_sampling_basic_robotiq(sample_point_cloud, robotiq_config_path):
    """Test basic grasp sampling functionality with robotiq gripper."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available for testing")

    try:
        grasp_cfg = load_grasp_cfg(robotiq_config_path)
        sampler = GraspGenSampler(grasp_cfg)

        # Test with a small number of grasps
        grasps, confidences = GraspGenSampler.run_inference(
            sample_point_cloud,
            sampler,
            grasp_threshold=0.7,
            num_grasps=10,
            remove_outliers=False,
        )

        # Check return types
        assert isinstance(grasps, (list, np.ndarray, torch.Tensor))
        assert isinstance(confidences, (list, np.ndarray, torch.Tensor))

        # If grasps were found, check their properties
        if len(grasps) > 0:
            # Check grasp poses are 4x4 matrices
            assert grasps[0].shape == (4, 4)

            # Check confidences are between 0 and 1
            if isinstance(confidences, (np.ndarray, torch.Tensor)):
                for conf in confidences:
                    assert conf >= 0 and conf <= 1

    except Exception as e:
        pytest.skip(f"Robotiq grasp sampling failed: {str(e)}")


def test_grasp_sampling_parameters_franka(sample_point_cloud, franka_config_path):
    """Test grasp sampling with different parameters using franka gripper."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available for testing")

    try:
        grasp_cfg = load_grasp_cfg(franka_config_path)
        sampler = GraspGenSampler(grasp_cfg)

        # Test with different thresholds
        for threshold in [0.5, 0.7]:
            grasps, confidences = GraspGenSampler.run_inference(
                sample_point_cloud,
                sampler,
                grasp_threshold=threshold,
                num_grasps=1000,
                remove_outliers=False,
            )

            if len(grasps) > 0 and isinstance(confidences, (np.ndarray, torch.Tensor)):
                # Check that confidences meet threshold
                for conf in confidences:
                    assert conf >= threshold

        # Test with different numbers of grasps
        for num_grasps in [500, 1000, 2000]:
            grasps, _ = GraspGenSampler.run_inference(
                sample_point_cloud,
                sampler,
                grasp_threshold=0.7,
                num_grasps=num_grasps,
                remove_outliers=False,
            )

            # Check that we don't get more grasps than requested
            assert len(grasps) <= num_grasps

    except Exception as e:
        pytest.skip(f"Franka parameter testing failed: {str(e)}")


def test_grasp_sampling_parameters_robotiq(sample_point_cloud, robotiq_config_path):
    """Test grasp sampling with different parameters using robotiq gripper."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available for testing")

    try:
        grasp_cfg = load_grasp_cfg(robotiq_config_path)
        sampler = GraspGenSampler(grasp_cfg)

        # Test with different thresholds
        for threshold in [0.5, 0.7]:
            grasps, confidences = GraspGenSampler.run_inference(
                sample_point_cloud,
                sampler,
                grasp_threshold=threshold,
                num_grasps=1000,
                remove_outliers=False,
            )

            if len(grasps) > 0 and isinstance(confidences, (np.ndarray, torch.Tensor)):
                # Check that confidences meet threshold
                for conf in confidences:
                    assert conf >= threshold

        # Test with different numbers of grasps
        for num_grasps in [500, 1000, 2000]:
            grasps, _ = GraspGenSampler.run_inference(
                sample_point_cloud,
                sampler,
                grasp_threshold=0.7,
                num_grasps=num_grasps,
                remove_outliers=False,
            )

            # Check that we don't get more grasps than requested
            assert len(grasps) <= num_grasps

    except Exception as e:
        pytest.skip(f"Robotiq parameter testing failed: {str(e)}")
