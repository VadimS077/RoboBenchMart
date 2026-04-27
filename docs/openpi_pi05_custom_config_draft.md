# Draft: OpenPI `pi0.5` config for custom LeRobot dataset

This is a practical draft for adapting `openpi` training config to the local dataset:

- Dataset path: `generated_envs/ds_small_scene_0/demos/benchmark/lerobot_graspgen_masked`
- Format: `lerobot-v3`
- Action dim: `13`
- State dim: `15`
- Visual keys:
  - `observation.images.fetch_hand`
  - `observation.images.left_base_camera_link`
  - `observation.images.right_base_camera_link`
  - `observation.images.fetch_hand_target_mask`
  - `observation.images.left_base_camera_link_target_mask`
  - `observation.images.right_base_camera_link_target_mask`
- Labels:
  - `success` (bool per step)
  - `task` (string prompt-like field)

---

## 1) Field mapping (what goes where)

Use this as the source-of-truth mapping in your OpenPI data config.

- **Model action target**
  - OpenPI output: `actions`
  - Dataset field: `action` (shape `[13]`)

- **Robot state input**
  - OpenPI input: `observation/state`
  - Dataset field: `observation.state` (shape `[15]`)

- **Language/prompt input**
  - OpenPI input: `prompt`
  - Dataset field:
    - primary: `task`
    - optional fallback: constant string if empty

- **Image inputs (recommended minimal setup)**
  - OpenPI input: `observation/wrist_image`
  - Dataset field: `observation.images.fetch_hand`
  - OpenPI input: `observation/wrist_target_mask`
  - Dataset field: `observation.images.fetch_hand_target_mask`

- **Optional extra views**
  - `observation.images.left_base_camera_link`
  - `observation.images.right_base_camera_link`
  - and their `*_target_mask` variants.

- **Success**
  - Keep as metadata/analysis field: `success`
  - Useful for filtering train episodes and evaluation slices.

---

## 2) Recommended first training variant

Start simple before multi-view:

- Inputs:
  - `fetch_hand` RGB
  - `fetch_hand_target_mask`
  - `observation.state`
  - `task`
- Output:
  - `action` (13D)

This reduces complexity and makes debugging easier.

---

## 3) Draft pseudo-config structure (OpenPI side)

Below is intentionally a draft skeleton (names may differ slightly between OpenPI versions).
Port this into your OpenPI config file next to other examples (e.g. LIBERO/DROID-style configs).

```python
# draft_openpi_pi05_dsynth.py (pseudo-code)

from dataclasses import dataclass

@dataclass
class DsynthInputs:
    # Required
    state_key: str = "observation.state"
    action_key: str = "action"
    prompt_key: str = "task"

    # Primary camera + mask
    wrist_rgb_key: str = "observation.images.fetch_hand"
    wrist_mask_key: str = "observation.images.fetch_hand_target_mask"

    # Optional extra cameras
    left_rgb_key: str = "observation.images.left_base_camera_link"
    right_rgb_key: str = "observation.images.right_base_camera_link"
    left_mask_key: str = "observation.images.left_base_camera_link_target_mask"
    right_mask_key: str = "observation.images.right_base_camera_link_target_mask"

    use_multi_view: bool = False
    use_success_filter: bool = True
    success_key: str = "success"


@dataclass
class DsynthDataConfig:
    dataset_path: str = "/ABS/PATH/TO/lerobot_graspgen_masked"
    split: str = "train"
    fps: int = 30
    action_dim: int = 13
    state_dim: int = 15
    # image transforms
    image_size: tuple[int, int] = (256, 256)
    normalize_images: bool = True


@dataclass
class TrainConfigPi05Dsynth:
    base_config_name: str = "pi05_droid"   # common starting point
    exp_name: str = "pi05_dsynth_masked"
    # weight source can be pi05_base or pi05_droid depending on your target
    checkpoint_uri: str = "gs://openpi-assets/checkpoints/pi05_base"
    batch_size: int = 16
    num_steps: int = 50000
    learning_rate: float = 1e-4
```

---

## 4) Success-aware filtering (strongly recommended)

Before training, optionally keep only successful episodes.
Since `success` is per-step, terminal success for an episode is:

- `episode_success = success[last_step]`

Use this to create:

- `train_success_only` (main run)
- `train_all` (ablation run)

---

## 5) State/action semantics for this robot

From `scripts/print_action_state_layout.py`:

- `action[0:7]`: arm joints
- `action[7]`: gripper scalar
- `action[8:11]`: body (`head_pan`, `head_tilt`, `torso_lift`)
- `action[11:13]`: base controller outputs

`observation.state` is 15D `qpos` in this order:

1. `root_x_axis_joint`
2. `root_y_axis_joint`
3. `root_z_rotation_joint`
4. `torso_lift_joint`
5. `head_pan_joint`
6. `shoulder_pan_joint`
7. `head_tilt_joint`
8. `shoulder_lift_joint`
9. `upperarm_roll_joint`
10. `elbow_flex_joint`
11. `forearm_roll_joint`
12. `wrist_flex_joint`
13. `wrist_roll_joint`
14. `r_gripper_finger_joint`
15. `l_gripper_finger_joint`

---

## 6) Pre-flight checks before launching OpenPI train

1. `forge inspect <dataset>` shows `success` and all mask keys.
2. `forge quality <dataset>` runs without schema errors.
3. A quick loader script (LeRobot) can read:
   - `action`
   - `observation.state`
   - `observation.images.fetch_hand`
   - `observation.images.fetch_hand_target_mask`
   - `success`
4. Norm stats script in OpenPI runs successfully for this config.

---

## 7) First launch template (adapt in OpenPI repo)

```bash
# 1) compute normalization stats
uv run scripts/compute_norm_stats.py --config-name pi05_dsynth_masked

# 2) train
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_dsynth_masked --exp-name=pi05_dsynth_masked_v1 --overwrite
```

If using PyTorch path in OpenPI, replace with `scripts/train_pytorch.py`.

