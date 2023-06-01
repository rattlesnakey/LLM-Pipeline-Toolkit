set -v
set -e

#! 容器路径记得修改

#! MODEL_ARGS
# PRETRAINED_MODEL=/workspace/share_916081/chillzhang/pretrained_models/flan-t5/flan-t5-small
PRETRAINED_MODEL=/apdcephfs/share_916081/effidit_shared_data/chillzhang/pretrained_models/flan-t5/flan-t5-xl
# MODEL_TYPE=M2M # mBART
MODEL_NAME=`echo ${PRETRAINED_MODEL} | awk -F '/' '{print $NF}'`
#! DATA ARGS
export HF_DATASETS_CACHE="/apdcephfs/share_916081/chillzhang/projects/LLM_inference/cache_dir/datasets"
# export HF_DATASETS_CACHE="/workspace/share_916081/chillzhang/projects/LLM_inference/cache_dir/datasets"

# DATA_DIR=/workspace/share_916081/chillzhang/datas/alpaca_instruction_data/encoder_decoder
DATA_DIR=/apdcephfs/share_916081/effidit_shared_data/chillzhang/datas/alpaca_instruction_data/encoder_decoder

TRAIN_DATASET_PATH=${DATA_DIR}/train.json
VALID_DATASET_PATH=${DATA_DIR}/valid.json
# TEST_DATASET_PATH=/home/yangshiping/yangsp/projects/prompt-cross-lin/data/zh-and-en-small/test.json #! 目前所有的测试集都用这个

MAX_SOURCE_LENGTH=256
MAX_TARGET_LENGTH=256
GENERATION_MAX_LENGTH=256

TRAIN_BATCH_SIZE=4
EVAL_BATCH_SIZE=4


#! TRAINING ARGS
EPOCHS=3 #! 测试的时候记得修改
SEED=42 # 不传的话，默认是 42
LR=1e-4 #,3e-5, 默认是用的 5e-5
NUM_BEAMS=5 # 不传的话默认用 generation 那边自带的
WARMUP_RATIO=0.1 #,0.2 # 0.1,0.2
METRIC_FOR_BEST_MODEL=loss # rouge1, rouge2,rougeL,rougeLsum or loss, 默认前面会添加 eval

GRADIENT_ACCUMULATE=4


# LOG_PATH=../logs/${WANDB_RUN_NAME}.log

#! WANDB
#! taiji 上面用不了
export WANDB_DISABLED=true
WANDB_PROJECT_NAME=instruction-tuning-llm
ENTITY=hengyuan
WANDB_RUN_NAME=${MODEL_NAME}-train-batch_${TRAIN_BATCH_SIZE}-gradient-acc_${GRADIENT_ACCUMULATE}-lr-${LR}-epochs_${EPOCHS}-beams_${NUM_BEAMS}
# export WANDB_PROJECT=${WANDB_PROJECT_NAME}
# export WANDB_ENTITY=${ENTITY}


#!OUTPUT
OUTPUT_MODEL_DIR=/apdcephfs/share_916081/chillzhang/projects/LLM_inference/checkpoints/${WANDB_RUN_NAME}
# OUTPUT_MODEL_DIR=/workspace/share_916081/chillzhang/projects/LLM_inference/checkpoints/${WANDB_RUN_NAME}

mkdir -p ${OUTPUT_MODEL_DIR}
# DEVICE=0

DEVICE=0,1,2,3,4,5,6,7 #! 两个卡的话，汇总到第一个卡上会爆掉，测试的时候，只用一张卡
# DEVICE=0,1,2,3

export CUDA_VISIBLE_DEVICES=${DEVICE}

LOG_PATH=/apdcephfs/share_916081/effidit_shared_data/chillzhang/projects/LLM_inference/logs
mkdir -p ${LOG_PATH}

# accelerate launch --config_file accelerate_config.yaml  ../src/train.py \
# accelerate launch --config_file accelerate_config.yaml  ../src/train.py \ #! 测试的时候记得改
# python -u
# deepspeed --num_gpus=4 --master_addr=${CHIEF_IP} --master_port=50002
deepspeed --num_gpus=8 --master_port=50002 src/train_encoder_decoder.py \
    --deepspeed deepspeed_zero3_offload_config.json \
    --model_name_or_path ${PRETRAINED_MODEL} \
    --do_train \
    --do_eval \
    --output_dir ${OUTPUT_MODEL_DIR} \
    --train_file ${TRAIN_DATASET_PATH} \
    --validation_file ${VALID_DATASET_PATH} \
    --max_source_length ${MAX_SOURCE_LENGTH} \
    --max_target_length ${MAX_TARGET_LENGTH} \
    --generation_max_length ${GENERATION_MAX_LENGTH} \
    --per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATE} \
    --learning_rate ${LR} \
    --num_train_epochs ${EPOCHS} \
    --warmup_ratio ${WARMUP_RATIO} \
    --seed ${SEED} \
    --metric_for_best_model ${METRIC_FOR_BEST_MODEL} \
    --num_beams ${NUM_BEAMS} \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --overwrite_output_dir \
    --pad_to_max_length \
    --save_total_limit 1 \
    --run_name ${WANDB_RUN_NAME} \
    --max_train_samples 100 \
    --max_eval_samples 20 2>&1 | tee ${LOG_PATH}/${WANDB_RUN_NAME}.log

    # --max_train_samples 100 \
    # --max_eval_samples 10 \
    # --predict_with_generate
    #     --fp16 True \
    #    --report_to wandb \

#! convert 
CHECKPOINT_PATH=${OUTPUT_MODEL_DIR}

OUTPUT_FILE=${CHECKPOINT_PATH}/pytorch_model.bin
python src/zero_to_fp32.py --checkpoint_dir ${CHECKPOINT_PATH} --output_file ${OUTPUT_FILE}

