set -x

export PYTHONPATH=$PWD:$PYTHONPATH
# export COMPACT_TEST_ENABLE=True
export COMPACT_TEST_MODEL="Flux"
export COMPACT_TEST_METHOD="binary"
export COMPACT_TEST_LOOP=1

# Select the model type
export MODEL_TYPE="Flux"
# Configuration for different model types
# script, model_id, inference_step
declare -A MODEL_CONFIGS=(
    ["Pixart-alpha"]="pixartalpha_example.py /root/autodl-fs/PixArt-XL-2-1024-MS 20"
    ["Pixart-sigma"]="pixartsigma_example.py /cfs/dit/PixArt-Sigma-XL-2-2K-MS 30"
    ["Sd3"]="sd3_example.py stabilityai/stable-diffusion-3-medium-diffusers 28"
    ["Flux"]="flux_example.py black-forest-labs/FLUX.1-dev 28"
    ["HunyuanDiT"]="hunyuandit_example.py /cfs/dit/HunyuanDiT-v1.2-Diffusers 50"
)

if [[ -v MODEL_CONFIGS[$MODEL_TYPE] ]]; then
    IFS=' ' read -r SCRIPT MODEL_ID INFERENCE_STEP <<< "${MODEL_CONFIGS[$MODEL_TYPE]}"
    export SCRIPT MODEL_ID INFERENCE_STEP
else
    echo "Invalid MODEL_TYPE: $MODEL_TYPE"
    exit 1
fi

mkdir -p ./results
export LOG_LEVEL=info # level: A string ('debug', 'info', 'warning', 'error', 'critical') 
IMG_SIZE=1024
# task args
TASK_ARGS="--height $IMG_SIZE --width $IMG_SIZE --no_use_resolution_binning"

# cache args
# CACHE_ARGS="--use_teacache"
# CACHE_ARGS="--use_fbcache"

# On 8 gpus, pp=2, ulysses=2, ring=1, cfg_parallel=2 (split batch)
export CUDA_VISIBLE_DEVICES=0,1,2,3
N_GPUS=4
PARALLEL_ARGS="--ulysses_degree 1 --ring_degree 4 --pipefusion_parallel_degree 1" #--pipefusion_parallel_degree 1

# CFG_ARGS="--use_cfg_parallel"

# By default, num_pipeline_patch = pipefusion_degree, and you can tune this parameter to achieve optimal performance.
# PIPEFUSION_ARGS="--num_pipeline_patch 8 "

# For high-resolution images, we use the latent output type to avoid runing the vae module. Used for measuring speed.
# OUTPUT_ARGS="--output_type latent"

# PARALLLEL_VAE="--use_parallel_vae"


# Another compile option is `--use_onediff` which will use onediff's compiler.
# COMPILE_FLAG="--use_torch_compile"


# Use this flag to quantize the T5 text encoder, which could reduce the memory usage and have no effect on the result quality.
# QUANTIZE_FLAG="--use_fp8_t5_encoder"

# export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nproc_per_node=$N_GPUS ./examples/$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 1 \
--prompt "Romantic painting of a ship sailing in a stormy sea, with dramatic lighting and powerful waves." \
$CFG_ARGS \
$PARALLLEL_VAE \
$COMPILE_FLAG \
$QUANTIZE_FLAG \
$CACHE_ARGS \
# 3 dogs wearing coats
# Astronaut in a jungle, cold color palette, muted colors, detailed, 8k
# brown dog laying on the ground with a metal bowl in front of him.
# A child holding a flowered umbrella and petting a yak.
# A street scene at an intersection with tall skyscrapers in the background. 
# A man in glasses eating a donut out of a cup.
# Romantic painting of a ship sailing in a stormy sea, with dramatic lighting and powerful waves.
# Ethereal fantasy concept art of an elf, magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy.