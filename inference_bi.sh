set -v
set -e

DEVICE=0,1,2,3,4,5,6,7
TEMPERATURE=0.35
TOP_P=0.85
TOP_K=40
NUM_BEAMS=1
REPETITION_PENALTY=1.2
MAX_SEQ_LEN=1024

# for size in 0.2 0.6 1 2
# do 
MODEL_TYPE=BLOOMZ+para-data-500w_ckpt-14000
# MODEL_TYPE=BELLE-${size}-revised #! FLAN-T5, LLAMA, BELLE, GPT2, ALPACA, Chinese_ALPACA, ChatGLM
# MODEL_TYPE=ALPACA+en2zh+zh2en
# MODEL_TYPE=BLOOMZ+alpaca+en2zh
# MODEL_TYPE=BLOOMZ+alpaca+en2zh+zh2en
# MODEL_TYPE=ChatGLM
# MODEL_TYPE=GPT2

BASE_DIR=/apdcephfs/share_916081/victoriabi
# PRETRAINED_MODEL_PATH=${BASE_DIR}/para_data/train/checkpoints/official-bloom-7b-mt-50k_v2_out.json-prompt-type_stanford-train-batch_2-lr_2e-5-epochs_3
PRETRAINED_MODEL_PATH=${BASE_DIR}/para_data/train/checkpoints/official-bloom-7b-mt-500w_out.json-prompt-type_stanford-train-batch_2-lr_2e-5-epochs_2/checkpoint-14000
PRETRAINED_TOKENIZER_PATH=${PRETRAINED_MODEL_PATH}


# TEST_FILE_PATH=${BASE_DIR}/datas/self-instruct-davinci-001/predictions/text-davinci-003_predictions.jsonl
#! fill your test data path, need to be jsonl format
#! 500_test
TEST_FILE_PATH=/apdcephfs/share_916081/victoriabi/para_data/train_data/5k_test.json
TEST_FILE_NAME=`echo ${TEST_FILE_PATH} | awk -F '/' '{print $NF}'`
# TEST_FILE_PATH=${BASE_DIR}/datas/belle/zh_seed_tasks_jsonlines.jsonl
# TEST_FILE_PATH=${BASE_DIR}/datas/en_zh_seed_task/en_zh_seed_tasks.jsonl
DATA_TYPE=seed_task #! or zh_seed_task, self-instruct, en_zh_seed_tasks
PROMPT_TYPE=stanford #! stanford, origin, BELLE
OUTPUT_FILE_DIR=${PRETRAINED_MODEL_PATH}/predictions/${DATA_TYPE}/${PROMPT_TYPE}
mkdir -p ${OUTPUT_FILE_DIR}

OUTPUT_FILE_PATH=${OUTPUT_FILE_DIR}/${MODEL_TYPE}_${TEST_FILE_NAME}_prediction.jsonl


# LOG_PATH=/workspace/share_916081/effidit_shared_data/chillzhang/projects/LLM_inference/logs/inference/${DATA_TYPE}

LOG_PATH=${BASE_DIR}/projects/LLM_inference/logs/inference/${DATA_TYPE}/${PROMPT_TYPE}


mkdir -p ${LOG_PATH}

MODEL_NAME=`echo ${PRETRAINED_MODEL_PATH} | awk -F '/' '{print $NF}'`

RUN_NAME=${MODEL_NAME}-${DATA_TYPE}-${PROMPT_TYPE}-temperature_${TEMPERATURE}-topk_${TOP_K}-topp_${TOP_P}-repetition_${REPETITION_PENALTY}-beams_${NUM_BEAMS}.log

#! use --add_lang when specify target language
#! instruction 的语言
SRC_LANG=zh
#! response 的语言
TGT_LANG=zh


export CUDA_VISIBLE_DEVICES=${DEVICE}

#! if use deepspeed, use deepspeed to launchd
# deepspeed --num_gpus=1
# python -u
#! deepspeed --include=localhost:2
deepspeed --include=localhost:5 src/inference_v2.py \
        --test_file_path ${TEST_FILE_PATH} \
        --output_file_path ${OUTPUT_FILE_PATH} \
        --pretrained_model_path ${PRETRAINED_MODEL_PATH} \
        --pretrained_tokenizer_path ${PRETRAINED_TOKENIZER_PATH} \
        --model_type ${MODEL_TYPE} \
        --temperature ${TEMPERATURE} \
        --top_p ${TOP_P} \
        --top_k ${TOP_K} \
        --num_beams ${NUM_BEAMS} \
        --repetition_penalty ${REPETITION_PENALTY} \
        --max_new_tokens ${MAX_SEQ_LEN} \
        --dataset_type ${DATA_TYPE} \
        --src_lang ${SRC_LANG} \
        --tgt_lang ${TGT_LANG} \
        --prompt_type ${PROMPT_TYPE} \
        --deepspeed_inference \
        2>&1 | tee ${LOG_PATH}/${RUN_NAME}
        #    --add_lang \ 
# done
# --deepspeed_inference
# deepspeed --num_gpus 2



    # --no_cuda
    # --deepspeed_inference
