#!/usr/bin/env bash

# Fixed parameters
export NGPU=1
export NWORKER=4
export NEPOCH=5000
export BATCH=8
export PRINT_FREQ=10
export PLOT_FREQ=10
export SAVE_FREQ=500
export DATASET_NAME="objaverse"
export DATASET_VERSION="v2"
export TIMESTEPS=10
export NUM_GRASPS_PER_OBJ=500
export BACKBONE="ptv3"
export NUM_POINTS=3500
export NAME="meshandpc"
export P_PC=0.50
export RED=7
export GRIPPER_NAME="single_suction_cup_30mm"
export OBJECT_DATASET_DIR="/results/tutorial/tutorial_object_dataset"
export GRASP_DIR="/results/tutorial/tutorial_grasp_dataset"
export CODE_DIR="/code"
export RESULTS_DIR="/results/tutorial"
export GRASP_DATASET_DIR="$GRASP_DIR"
export SPLIT_DATASET_DIR="$OBJECT_DATASET_DIR"
export METHOD="grasp_gen"
export PYRENDER_INSTALL_PREFIX="apt-get update -y && apt-get install -y tmux libosmesa6-dev && pip install pyrender && pip install PyOpenGL==3.1.5 && export PYOPENGL_PLATFORM=osmesa && "
export ROTATION_REPR="r3_so3"
export PYOPENGL_PLATFORM="osmesa"
export NOISE_SCALE=1.0
export LOG_DIR="$RESULTS_DIR/logs/${GRIPPER_NAME}_gen_test"
export CHECKPOINT="$LOG_DIR/last.pth"
export CACHE_DIR="$RESULTS_DIR/cache"

echo "Running Training for $GRIPPER_NAME"

rm -rf $LOG_DIR
mkdir -p $LOG_DIR
mkdir -p $CACHE_DIR

cd $CODE_DIR && pip install -e . && cd $CODE_DIR/scripts && \
    python train_graspgen.py \
    data.num_points=$NUM_POINTS \
    data.load_contact=False \
    data.dataset_cls="ObjectPickDataset" \
    data.rotation_augmentation=True \
    data.cache_dir=$CACHE_DIR \
    data.root_dir=$SPLIT_DATASET_DIR \
    data.object_root_dir=$OBJECT_DATASET_DIR \
    data.grasp_root_dir=$GRASP_DATASET_DIR \
    data.dataset_name=$DATASET_NAME \
    data.dataset_version=$DATASET_VERSION \
    data.prob_point_cloud=$P_PC \
    data.redundancy=$RED \
    data.gripper_name=$GRIPPER_NAME \
    train.log_dir=$LOG_DIR \
    train.batch_size=$BATCH \
    train.num_gpus=$NGPU \
    train.num_epochs=$NEPOCH \
    train.num_workers=$NWORKER \
    train.print_freq=$PRINT_FREQ \
    train.plot_freq=$PLOT_FREQ \
    train.save_freq=$SAVE_FREQ \
    train.checkpoint=$CHECKPOINT \
    train.model_name='diffusion' \
    train.debug=True \
    optimizer.type="ADAMW" \
    optimizer.lr=0.00001 \
    optimizer.grad_clip=-1 \
    diffusion.gripper_name=$GRIPPER_NAME \
    diffusion.num_diffusion_iters=$TIMESTEPS \
    diffusion.num_diffusion_iters_eval=$TIMESTEPS \
    diffusion.obs_backbone=$BACKBONE \
    diffusion.grasp_repr=$ROTATION_REPR \
    diffusion.attention='cat_attn' \
    diffusion.compositional_schedular=True \
    diffusion.loss_pointmatching=False \
    diffusion.loss_l1_pos=True \
    diffusion.loss_l1_rot=True \
    diffusion.ptv3.grid_size=0.01 \
    diffusion.pose_repr='mlp' \
    data.num_grasps_per_object=$NUM_GRASPS_PER_OBJ \
    data.load_discriminator_dataset=False \
    data.visualize_batch=False \
    diffusion.kappa=$NOISE_SCALE 