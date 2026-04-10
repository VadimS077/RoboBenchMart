# Tutorial: Training a GraspGen model on a single object

This is a minimal tutorial demonstrating how to generate a dataset and train a GraspGen (generator and discriminator) models for a single object using a suction cup gripper. This tutorial skips the advanced concepts such as On-Generator training. The model trained in this tutorial will generalize to unknown poses of the same object, but will be overfit to just the one object used for training.

## 📋 Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Generate Dataset](#step-1-generate-dataset)
3. [Step 2: Train Generator Model](#step-2-train-generator-model)
4. [Step 3: Train Discriminator Model](#step-3-train-discriminator-model)
5. [Step 4: Create Model Checkpoints for Inference](#step-4-create-model-checkpoints-for-inference)
6. [Step 5: Visualize Predictions](#step-5-visualize-predictions)
7. [Complete Workflow Example](#complete-workflow-example)
8. [File Structure](#file-structure)

## Prerequisites

1. **Docker Setup**: Follow the prerequisites in the [main README](../README.md) to start the Docker container suitable for training. Make sure to have [downloaded the model checkpoints](#download-checkpoints) as this repo (located at `<path_to_models_repo>`) contains the sample mesh data used in this tutorial:
   ```bash
   # Create results directory
   mkdir -p <path_to_results>
   
   # Start Docker container
   ./docker/run.sh <path_to_graspgen_code> --grasp_dataset <path_to_grasp_dataset> --object_dataset <path_to_object_dataset> --results <path_to_results> --models <path_to_models_repo> 
   ```

2. **Object Mesh**: In this tutorial we use the object `/models/sample_data/meshes/box.obj`. But you can use any other 3D mesh file (a `.obj`, `.stl` file).

## Step 1: Generate Dataset

Use the `generate_dataset_suction_single_object.py` script to create a dataset for your single object.

### Basic Usage:
```bash
cd /code && python tutorials/generate_dataset_suction_single_object.py \
    --object_path /models/sample_data/meshes/box.obj \
    --object_scale 1.0 \
    --output_dir /results/tutorial \
    --no_visualization
```

### Parameters:
- `--object_path`: Path to your object mesh file (required)
- `--output_dir`: Directory to save datasets (default: `/results/tutorial`)
- `--num_grasps`: Total number of grasps to generate (default: 2000)
- `--object_scale`: Scale factor for the object (default: 1.0)
- `--gripper_config`: Gripper configuration file (default: `single_suction_cup_30mm.yaml`)
- `--num_disturbances`: Number of disturbance samples for evaluation (default: 10)
- `--no_visualization`: Disable visualization

### Output:
The script creates the following dataset with the directory structure. It follows the convention in the [GraspGen Dataset Format](../docs/GRASP_DATASET_FORMAT.md).

```
/results/tutorial/
├── tutorial_object_dataset/
│   ├── <object_name>.obj
│   ├── train.txt
│   └── valid.txt
└── tutorial_grasp_dataset/
    └── <object_name>_grasps.json
```

## Step 2: Train Generator Model

Train a diffusion model to generate grasp poses for your object.

### Command:
```bash
cd /code && bash tutorials/tutorial_train_gen.sh
```

### Expected Output:
- Training logs in `/results/tutorial/logs/single_suction_cup_30mm_gen_test/`
- Model checkpoints saved as `last.pth`
- Cache files in `/results/tutorial/cache/`

### Monitoring Training:
- For best perf, the validation the grasp reconstruction error `reconstruction/error_trans_l2` should be a few cm
- This run may take at least 1K epochs to converge. However, for a large object dataset (e.g. set of 8K objects), it takes about 3-5K epochs to converge.

## Step 3: Train Discriminator Model

Train a discriminator model to evaluate grasp quality.

### Command:
```bash
cd /code && bash tutorials/tutorial_train_dis.sh
```

### Expected Output:
- Training logs in `/results/tutorial/logs/single_suction_cup_30mm_dis_test/`
- Model checkpoints saved as `last.pth`
- For best perf, Validation AP score should be > 0.8; `bce_topk` loss should go down

### Monitoring Training:
- Watch for validation AP score (should be > 0.8)
- This run may take at least 1K epochs to converge. However, for a larger object dataset (e.g. set of 8K objects), it takes about 3-5K epochs to converge.
- Monitor losses for different grasp types (pos, neg, freespace, onpolicy)
- Check that all loss curves are decreasing

## Step 4: Create Final Model config file for Inference

After training both models, create standardized checkpoints for inference:

### Command:
```bash
cd /code && python tutorials/generate_model_inference_config.py \
    --gen_log_dir /results/tutorial/logs/single_suction_cup_30mm_gen_test \
    --dis_log_dir /results/tutorial/logs/single_suction_cup_30mm_dis_test
```

### Expected Output:
- Models directory: `/results/tutorial/models/`
- Generator checkpoint: `gen.pth`
- Discriminator checkpoint: `dis.pth`


## Step 5: Visualize Predictions

Visualize grasp predictions from your trained model on the sample object mesh.

### Command:
```bash
cd /code && python scripts/demo_object_mesh.py \
    --mesh_file /models/sample_data/meshes/box.obj \
    --mesh_scale 1.0 \
    --gripper_config /results/tutorial/models/tutorial_model_config.yaml
```

### Parameters:
- `--mesh_file`: Path to the object mesh file (using the same box.obj from the example)
- `--mesh_scale`: Scale factor for the object mesh (default: 1.0)
- `--gripper_config`: Path to the config file created in Step 4 (default: `/results/tutorial/models/tutorial_model_config.yaml`)

### Expected Output:
- Interactive 3D visualization showing the object mesh
- Generated grasp poses overlaid on the object
- Color-coded grasp quality scores
- Best grasps highlighted

## Complete Workflow Example

Here's a complete example for training on a mug object:

```bash
# 1. Generate dataset
cd /code && python tutorials/generate_dataset_suction_single_object.py \
    --object_path /path/to/mug.obj \
    --output_dir /results/tutorial \
    --num_grasps 2000 \
    --object_scale 1.0

# 2. Train generator
cd /code && bash tutorials/tutorial_gen.sh

# 3. Train discriminator
cd /code && bash tutorials/tutorial_dis.sh

# 4. Create model checkpoints for inference
cd /code && python tutorials/create_model_checkpoints_for_inference.py \
    --gen_log_dir /results/tutorial/logs/single_suction_cup_30mm_gen_test \
    --dis_log_dir /results/tutorial/logs/single_suction_cup_30mm_dis_test

# 5. Visualize predictions
cd /code && python scripts/demo_object_mesh.py \
    --mesh_file /models/sample_data/meshes/box.obj \
    --mesh_scale 1.0 \
    --gripper_config /results/tutorial/models/tutorial_model_config.yaml
```

## File Structure

After running all scripts, you'll have:
```
/results/tutorial/
├── tutorial_object_dataset/
│   ├── mug.obj
│   ├── train.txt
│   └── valid.txt
├── tutorial_grasp_dataset/
│   └── mug_grasps.json
├── logs/
│   ├── single_suction_cup_30mm_gen_test/
│   │   ├── last.pth
│   │   └── training_logs/
│   └── single_suction_cup_30mm_dis_test/
│       ├── last.pth
│       └── training_logs/
├── models/
│   ├── gen.pth
│   ├── dis.pth
│   └── tutorial_model_config.yaml
└── cache/
    └── tutorial_object_dataset/
       ├── cache_train_meshandpc_gen.h5
       └── cache_valid_meshandpc_gen.h5
```
