
set -x
# !/bin/bash

TEST_METHOD=${1:-ring}
TARGET_RATE=${2:-"unlimited"}
TEST_LOOP=${3:-10}

# TEST_ENABLE = bool(os.environ.get('COMPACT_TEST_ENABLE', False))
export COMPACT_TEST_ENABLE=true
export COMPACT_TEST_METHOD=$TEST_METHOD
export COMPACT_TEST_LOOP=$TEST_LOOP
# Select the model type
export COMPACT_TEST_MODEL=Flux

export MODEL_TYPE="Flux"

N_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
IMG_SIZE=1024



# --- 配置 NCCL 环境变量 ---
# 1. 禁用 P2P (阻止使用 PHB/SYS)
export NCCL_P2P_DISABLE=1

# 2. 指定使用 lo 网络接口
export NCCL_SOCKET_IFNAME=lo

# 3. 禁用共享内存 (防止节点内优化)
export NCCL_SHM_DISABLE=1
# 4. 禁用 InfiniBand/RoCE (以防万一)
export NCCL_IB_DISABLE=1

export NCCL_TIMEOUT=600000

# --- 配置 tc 带宽限制 ---
# 定义要限制的目标带宽
# TARGET_RATE="10gbit"
INTERFACE_TO_LIMIT="lo"

echo "--- Applying bandwidth limit (${TARGET_RATE}) to ${INTERFACE_TO_LIMIT} using tc ---"
# 清理旧的 tc 规则 (忽略错误，如果不存在的话)
tc qdisc del dev ${INTERFACE_TO_LIMIT} root 2> /dev/null || true

# 添加 HTB (Hierarchical Token Bucket) qdisc 到根
tc qdisc add dev ${INTERFACE_TO_LIMIT} root handle 1: htb default 10

# 添加一个 class，设置带宽上限
if [ "$TARGET_RATE" = "unlimited" ]; then
  echo "Running without bandwidth limit"
  # 删除现有限制但不添加新的
  tc qdisc del dev lo root 2>/dev/null || true
else
  echo "Applying bandwidth limit ($TARGET_RATE) to lo using tc"
  # 执行tc命令
  tc class add dev ${INTERFACE_TO_LIMIT} parent 1: classid 1:1 htb rate ${TARGET_RATE}
fi

# 添加一个过滤器，将所有从此接口出去的 IP 流量导向限速 class
tc filter add dev ${INTERFACE_TO_LIMIT} protocol ip parent 1: prio 1 u32 match ip src 0.0.0.0/0 flowid 1:1

# # =========================
# # ====== 可以运行代码 ======
# # =========================





# save_dir_name
SAVE_DIR_NAME="testLog/BWTest0429/MODEL_TYPE_${MODEL_TYPE}_IMG_SIZE_${IMG_SIZE}_COMPACT_TEST_METHOD_${COMPACT_TEST_METHOD}_COMPACT_TEST_LOOP_${COMPACT_TEST_LOOP}_TARGET_RATE_${TARGET_RATE}"

export PYTHONPATH=$PWD:$PYTHONPATH

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/hf
# Configuration for different model types
# script, model_id, inference_step
declare -A MODEL_CONFIGS=(
    ["Pixart-alpha"]="pixartalpha_example.py PixArt-alpha/PixArt-XL-2-1024-MS 20"
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
export LOG_LEVEL=error # level: A string ('debug', 'info', 'warning', 'error', 'critical') 

# task args
TASK_ARGS="--height $IMG_SIZE --width $IMG_SIZE --no_use_resolution_binning"

# cache args
# CACHE_ARGS="--use_teacache"
# CACHE_ARGS="--use_fbcache"

# On 8 gpus, pp=2, ulysses=2, ring=1, cfg_parallel=2 (split batch)


PARALLEL_ARGS="--ulysses_degree 1 --ring_degree 1 --pipefusion_parallel_degree $N_GPUS --enable_slicing --enable_tiling" #--pipefusion_parallel_degree 1

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
--save_dir $SAVE_DIR_NAME \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--num_inference_steps $INFERENCE_STEP \
--warmup_steps 1 \
--prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
$CFG_ARGS \
$PARALLLEL_VAE \
$COMPILE_FLAG \
$QUANTIZE_FLAG \
$CACHE_ARGS \
--output_type latent \
# 3 dogs wearing coats
# Astronaut in a jungle, cold color palette, muted colors, detailed, 8k
# brown dog laying on the ground with a metal bowl in front of him.

# --- 清理 tc 带宽限制规则 ---
echo "--- Removing bandwidth limit from ${INTERFACE_TO_LIMIT} ---"
tc qdisc del dev ${INTERFACE_TO_LIMIT} root
echo "--- tc rules removed. ---"


# (可选) 取消环境变量设置
unset NCCL_P2P_DISABLE
unset NCCL_SHM_DISABLE
unset NCCL_SOCKET_IFNAME
unset NCCL_IB_DISABLE