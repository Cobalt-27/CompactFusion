set -x

export PYTHONPATH=$PWD:$PYTHONPATH
export CAPTION_FILE="/root/autodl-tmp/ref_images/prompts.json"
export SAMPLE_IMAGES_FOLODER="/root/autodl-tmp/generated_images_int2_warmpup2"

# Select the model type
export MODEL_TYPE="Flux"
# Configuration for different model types
# script, model_id, inference_step
declare -A MODEL_CONFIGS=(
    ["Pixart-alpha"]="pixartalpha_generate.py /root/autodl-fs/PixArt-XL-2-1024-MS 20"
    ["Flux"]="flux_generate.py /root/autodl-fs/FLUX.1-dev 28"
)

if [[ -v MODEL_CONFIGS[$MODEL_TYPE] ]]; then
    IFS=' ' read -r SCRIPT MODEL_ID INFERENCE_STEP <<< "${MODEL_CONFIGS[$MODEL_TYPE]}"
    export SCRIPT MODEL_ID INFERENCE_STEP
else
    echo "Invalid MODEL_TYPE: $MODEL_TYPE"
    exit 1
fi

# task args
TASK_ARGS="--height 1024 --width 1024 --no_use_resolution_binning"

export TQDM_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
N_GPUS=4
PARALLEL_ARGS="--pipefusion_parallel_degree 1 --ulysses_degree 1 --ring_degree 4"

torchrun --nproc_per_node=$N_GPUS ./benchmark/fid/$SCRIPT \
--model $MODEL_ID \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 2 \
$CFG_ARGS \
$PARALLLEL_VAE \
$COMPILE_FLAG \
--caption_file $CAPTION_FILE \
--sample_images_folder $SAMPLE_IMAGES_FOLODER \
