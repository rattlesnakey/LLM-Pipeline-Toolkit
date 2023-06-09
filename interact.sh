set -v
set -e

DEVICE=0
TEMPERATURE=0.35
TOP_P=0.85
TOP_K=40
NUM_BEAMS=1
REPETITION_PENALTY=1.2
MAX_SEQ_LEN=256

BASE_DIR=~
LORA_BASE_DIR=${BASE_DIR}/checkpoints/lora

PRETRAINED_MODEL_PATH=${BASE_DIR}/pretrained_models/LLAMA/7b
LORA_WEIGHTS_PATH=${LORA_BASE_DIR}/xx
PRETRAINED_TOKENIZER_PATH=${PRETRAINED_MODEL_PATH}
USE_LORA=True
DECODE_MODE=greedy

LOG_PATH=${BASE_DIR}/logs/inference
mkdir -p ${LOG_PATH}

MODEL_NAME=`echo ${PRETRAINED_MODEL_PATH} | awk -F '/' '{print $NF}'`

RUN_NAME=${MODEL_NAME}-temperature_${TEMPERATURE}-topp_${TOP_P}-repetition_${REPETITION_PENALTY}.log

export CUDA_VISIBILE_DEVICES=${DEVICE}
deepspeed --num_gpus=1 src/interact.py \
    --pretrained_model_path ${PRETRAINED_MODEL_PATH} \
    --pretrained_tokenizer_path ${PRETRAINED_TOKENIZER_PATH} \
    --use_lora ${USE_LORA} \
    --lora_weights_path ${LORA_WEIGHTS_PATH} \
    --model_type ${MODEL_TYPE} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --top_k ${TOP_K} \
    --num_beams ${NUM_BEAMS} \
    --decoding_mode ${DECODE_MODE} \
    --repetition_penalty ${REPETITION_PENALTY} \
    --max_new_tokens ${MAX_SEQ_LEN} \
    --deepspeed_inference 2>&1 | tee ${LOG_PATH}/${RUN_NAME}
