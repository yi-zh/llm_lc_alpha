#!/bin/bash
# 校验TENSOR_PARALLEL_SIZE合法性
if [[ -z "$TENSOR_PARALLEL_SIZE" ]]; then
    echo "Error: TENSOR_PARALLEL_SIZE environment variable is not set." >&2
    exit 1
fi

if ! [[ "$TENSOR_PARALLEL_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: TENSOR_PARALLEL_SIZE must be a positive integer, but got '$TENSOR_PARALLEL_SIZE'." >&2
    exit 1
fi

# 检查nvidia-smi是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi command not found. Is NVIDIA drivers installed?" >&2
    exit 1
fi

# 获取GPU信息并检查有效性
gpu_info=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)
if [[ -z "$gpu_info" ]]; then
    echo "Error: Failed to retrieve GPU information using nvidia-smi." >&2
    exit 1
fi

# 处理GPU数据：删除空格并按显存降序排序
sorted_gpus=$(echo "$gpu_info" | tr -d ' ' | sort -t ',' -k2,2nr)

# 提取前N个GPU的索引
selected_indices=()
count=0
while IFS=',' read -r index free_memory; do
    selected_indices+=("$index")
    ((count++))
    if (( count >= TENSOR_PARALLEL_SIZE )); then
        break
    fi
done <<< "$sorted_gpus"

# 校验GPU数量是否足够
if (( ${#selected_indices[@]} < TENSOR_PARALLEL_SIZE )); then
    echo "Error: Only found ${#selected_indices[@]} GPU(s), but TENSOR_PARALLEL_SIZE requires $TENSOR_PARALLEL_SIZE." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${selected_indices[*]}")
echo "Selected GPUs: $CUDA_VISIBLE_DEVICES" >&2

HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-"8000"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-"0.95"}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-"1"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-"32768"}
DTYPE=${DTYPE:-"auto"}
UVICORN_LOG_LEVEL=${UVICORN_LOG_LEVEL:-"info"}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-"auto"}
MAX_PARALLEL_LOADING_WORKERS=${MAX_PARALLEL_LOADING_WORKERS:-"4"}

ENABLE_AUTO_TOOL_CHOICE=${ENABLE_AUTO_TOOL_CHOICE:-"false"}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-"mistral"}

echo "starting..."
echo "LLM_MODEL_PATH: ${LLM_MODEL_PATH}"
echo "MODEL_NAME: ${MODEL_NAME}"
echo "HOST: ${HOST}"
echo "PORT: ${PORT}"
echo "GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION}"
echo "TENSOR_PARALLEL_SIZE: ${TENSOR_PARALLEL_SIZE}"
echo "MAX_MODEL_LEN: ${MAX_MODEL_LEN}"
echo "DTYPE: ${DTYPE}"
echo "UVICORN_LOG_LEVEL: ${UVICORN_LOG_LEVEL}"
echo "KV_CACHE_DTYPE: ${KV_CACHE_DTYPE}"
echo "MAX_PARALLEL_LOADING_WORKERS: ${MAX_PARALLEL_LOADING_WORKERS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "ENABLE_AUTO_TOOL_CHOICE: ${ENABLE_AUTO_TOOL_CHOICE}"
echo "TOOL_CALL_PARSER: ${TOOL_CALL_PARSER}"

args=()

if [ "${ENABLE_AUTO_TOOL_CHOICE}" = "true" ]; then
    args+=(--enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER}")
fi

if [ "${TENSOR_PARALLEL_SIZE}" -eq 1 ]; then
    args+=(--max-parallel-loading-workers "${MAX_PARALLEL_LOADING_WORKERS}")
fi

vllm serve "${LLM_MODEL_PATH}" --host "${HOST}" --port "${PORT}" \
    --served-model-name "${MODEL_NAME}" --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" --dtype "${DTYPE}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --uvicorn-log-level "${UVICORN_LOG_LEVEL}" \
    "${args[@]}"