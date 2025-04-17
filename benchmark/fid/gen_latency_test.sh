set -x

export PYTHONPATH=$PWD:$PYTHONPATH
export CAPTION_FILE="prompts.json"
export SAMPLE_IMAGES_FOLDER="results/gen_latency_test"

# Select the model type
export MODEL_TYPE="Pixart-alpha"

export COMPACT_TEST_ENABLE=1
export COMPACT_TEST_MODEL="$MODEL_TYPE"
export COMPACT_TEST_METHOD="ring"
export COMPACT_TEST_LOOP=20



# level: A string ('debug', 'info', 'warning', 'error', 'critical') 
export LOG_LEVEL=info 
# no progress bar
export TQDM_DISABLE=1

IMG_SIZE=1024

# --- Validate Test Method ---
VALID_METHODS="binary lowrank12 lowrank8 df pipe ring patch ulysses"
if [[ ! " $VALID_METHODS " =~ " $COMPACT_TEST_METHOD " ]]; then
    echo "Invalid COMPACT_TEST_METHOD: $COMPACT_TEST_METHOD"
    echo "Valid methods are: $VALID_METHODS"
    exit 1
fi

# Configuration for different model types
# script, model_id, inference_step
declare -A MODEL_CONFIGS=(
    ["Pixart-alpha"]="pixartalpha_generate.py PixArt-alpha/PixArt-XL-2-1024-MS 20"
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
TASK_ARGS="--height $IMG_SIZE --width $IMG_SIZE --no_use_resolution_binning"

export CUDA_VISIBLE_DEVICES=0
N_GPUS=1

# --- Set Parallelism based on Method and N_GPUS ---
if [ "$N_GPUS" -eq 1 ]; then
  # Force sequential if only 1 GPU
  PARALLEL_ARGS="--pipefusion_parallel_degree 1 --ulysses_degree 1 --ring_degree 1"
else
  case $COMPACT_TEST_METHOD in
    pipe)
      PARALLEL_ARGS="--pipefusion_parallel_degree $N_GPUS --ulysses_degree 1 --ring_degree 1"
      ;;
    ulysses)
      PARALLEL_ARGS="--pipefusion_parallel_degree 1 --ulysses_degree $N_GPUS --ring_degree 1"
      ;;
    ring|patch|df|binary|lowrank*)
      # Default to ring parallelism for compact methods, patch, df, and ring itself
      PARALLEL_ARGS="--pipefusion_parallel_degree 1 --ulysses_degree 1 --ring_degree $N_GPUS"
      ;;
    *)
      # Should not happen due to validation, but exit if it does
      echo "Error: Unexpected method $COMPACT_TEST_METHOD encountered after validation."
      exit 1
      ;;
  esac
fi

echo "Using Parallel Args: $PARALLEL_ARGS"

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
--sample_images_folder $SAMPLE_IMAGES_FOLDER \
