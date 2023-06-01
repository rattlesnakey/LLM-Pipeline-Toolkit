set -v
set -e

DEVICE=0
TEMPERATURE=0.35
TOP_P=0.85
TOP_K=40
NUM_BEAMS=1
REPETITION_PENALTY=1.2
MAX_SEQ_LEN=256

MODEL_TYPE=LLAMA #! FLAN-T5, LLAMA
PRETRAINED_MODEL_PATH=/apdcephfs/share_916081/effidit_shared_data/chillzhang/projects/LLM_inference/checkpoints/official-llama-7b-train-batch_2-lr_2e-5-epochs_3
PRETRAINED_TOKENIZER_PATH=${PRETRAINED_MODEL_PATH}

# LOG_PATH=/workspace/share_916081/effidit_shared_data/chillzhang/projects/LLM_inference/logs/interact
LOG_PATH=/apdcephfs/share_916081/effidit_shared_data/chillzhang/projects/LLM_inference/logs/inference
mkdir -p ${LOG_PATH}

MODEL_NAME=`echo ${PRETRAINED_MODEL_PATH} | awk -F '/' '{print $NF}'`

RUN_NAME=${MODEL_NAME}-temperature_${TEMPERATURE}-topp_${TOP_P}-repetition_${REPETITION_PENALTY}.log


 #! 记得 docker 里面的路径和盘上面的路径的区别


export CUDA_VISIBILE_DEVICES=${DEVICE}

#! if use deepspeed, use deepspeed to launchd
# deepspeed --num_gpus=1
# python -u
deepspeed --num_gpus=1 src/decode.py \
    --pretrained_model_path ${PRETRAINED_MODEL_PATH} \
    --pretrained_tokenizer_path ${PRETRAINED_TOKENIZER_PATH} \
    --model_type ${MODEL_TYPE} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --top_k ${TOP_K} \
    --num_beams ${NUM_BEAMS} \
    --repetition_penalty ${REPETITION_PENALTY} \
    --max_new_tokens ${MAX_SEQ_LEN} \
    --deepspeed_inference 2>&1 | tee ${LOG_PATH}/${RUN_NAME}
# deepspeed --num_gpus 2


    # --no_cuda
    # --deepspeed_inference
