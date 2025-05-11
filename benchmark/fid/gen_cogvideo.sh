#!/bin/bash
set -x

export PYTHONPATH=$PWD:$PYTHONPATH
# CogVideoX configuration
SCRIPT="cogvideox_generate.py"
MODEL_ID="/root/autodl-fs/CogVideoX-2b"
INFERENCE_STEP=50
export CAPTION_FILE="vprompts.json"
export SAMPLE_VIDEOS_FOLDER="sample_videos_ring_6gpu"

# CogVideoX specific task args
TASK_ARGS="--height 480 --width 720 --num_frames 49"

# CogVideoX parallel configuration
export TQDM_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
N_GPUS=6
PARALLEL_ARGS="--ulysses_degree 1 --ring_degree $N_GPUS"
# CFG_ARGS="--use_cfg_parallel"

# Uncomment and modify these as needed
# OUTPUT_ARGS="--output_type latent"
# PARALLLEL_VAE="--use_parallel_vae"
ENABLE_TILING="--enable_tiling --enable_slicing"
PORT=29501
torchrun --nproc_per_node=$N_GPUS --master_port=$PORT ./benchmark/fid/$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 0 \
$CFG_ARGS \
$PARALLLEL_VAE \
$ENABLE_TILING \
$COMPILE_FLAG \
--caption_file $CAPTION_FILE \
--sample_videos_folder $SAMPLE_VIDEOS_FOLDER \