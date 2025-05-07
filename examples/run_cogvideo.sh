#!/bin/bash
set -x

export PYTHONPATH=$PWD:$PYTHONPATH

# CogVideoX configuration
SCRIPT="cogvideox_example.py"
MODEL_ID="/mnt/CogVideoX1.5-5B"
INFERENCE_STEP=50

mkdir -p ./results

# CogVideoX specific task args
TASK_ARGS="--height 768 --width 1360 --num_frames 33"

# CogVideoX parallel configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
N_GPUS=4
PARALLEL_ARGS="--ulysses_degree 1 --ring_degree 4"
CFG_ARGS=""
# CFG_ARGS="--use_cfg_parallel"

# Uncomment and modify these as needed
# PIPEFUSION_ARGS="--num_pipeline_patch 8"
# OUTPUT_ARGS="--output_type latent"
# PARALLLEL_VAE="--use_parallel_vae"
ENABLE_TILING="--enable_tiling"
COMPILE_FLAG="--use_torch_compile"

prompt="A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. "
prompt+="The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other "
prompt+="pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, "
prompt+="casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. "
prompt+="The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical "
prompt+="atmosphere of this unique musical performance."

torchrun --nproc_per_node=$N_GPUS ./examples/$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 0 \
--prompt "$prompt" \
$CFG_ARGS \
$PARALLLEL_VAE \
$ENABLE_TILING \
$COMPILE_FLAG