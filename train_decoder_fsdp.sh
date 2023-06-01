set -v
set -e

#! 容器路径记得修改

#! MODEL_ARGS
# PRETRAINED_MODEL=/workspace/share_916081/chillzhang/pretrained_models/GPT2/gpt2-base
# PRETRAINED_MODEL=/apdcephfs/share_916081/chillzhang/pretrained_models/GPT2/gpt2-base
PRETRAINED_MODEL=/apdcephfs/share_916081/effidit_shared_data/chillzhang/pretrained_models/LLAMA/hf/llama-7b
# MODEL_TYPE=M2M # mBART
MODEL_NAME=`echo ${PRETRAINED_MODEL} | awk -F '/' '{print $NF}'`
#! DATA ARGS
export HF_DATASETS_CACHE="/apdcephfs/share_916081/effidit_shared_data/chillzhang/projects/LLM_inference/cache_dir/datasets"
# export HF_DATASETS_CACHE="/workspace/share_916081/effidit_shared_data/chillzhang/projects/LLM_inference/cache_dir/datasets"

# DATA_DIR=/workspace/share_916081/chillzhang/datas/alpaca_instruction_data/causal
DATA_DIR=/apdcephfs/share_916081/effidit_shared_data/chillzhang/datas/alpaca_instruction_data/causal

TRAIN_DATASET_PATH=${DATA_DIR}/train.json
VALID_DATASET_PATH=${DATA_DIR}/valid.json
# TEST_DATASET_PATH=/home/yangshiping/yangsp/projects/prompt-cross-lin/data/zh-and-en-small/test.json #! 目前所有的测试集都用这个

MAX_SEQ_LENGTH=512


TRAIN_BATCH_SIZE=4
EVAL_BATCH_SIZE=4


#! TRAINING ARGS
EPOCHS=3 #! 测试的时候记得修改
SEED=42 # 不传的话，默认是 42
LR=2e-5 #,3e-5, 默认是用的 5e-5
WARMUP_RATIO=0.03 #,0.2 # 0.1,0.2
METRIC_FOR_BEST_MODEL=loss # rouge1, rouge2,rougeL,rougeLsum or loss, 默认前面会添加 eval

GRADIENT_ACCUMULATE=2


# LOG_PATH=../logs/${WANDB_RUN_NAME}.log

#! WANDB
export WANDB_DISABLED=true
WANDB_PROJECT_NAME=instruction-tuning-llm
ENTITY=hengyuan
WANDB_RUN_NAME=${MODEL_NAME}-train-batch_${TRAIN_BATCH_SIZE}-lr_${LR}-epochs_${EPOCHS}
# export WANDB_PROJECT=${WANDB_PROJECT_NAME}
# export WANDB_ENTITY=${ENTITY}


#!OUTPUT
OUTPUT_MODEL_DIR=/apdcephfs/share_916081/effidit_shared_data/chillzhang/projects/LLM_inference/checkpoints/${WANDB_RUN_NAME}
# OUTPUT_MODEL_DIR=/workspace/share_916081/effidit_shared_data/chillzhang/projects/LLM_inference/checkpoints/${WANDB_RUN_NAME}

mkdir -p ${OUTPUT_MODEL_DIR}
# DEVICE=0
DEVICE=0,1,2,3,4,5,6,7 #! 两个卡的话，汇总到第一个卡上会爆掉，测试的时候，只用一张卡
export CUDA_VISIBLE_DEVICES=${DEVICE}

LOG_PATH=/apdcephfs/share_916081/effidit_shared_data/chillzhang/projects/LLM_inference/logs
mkdir -p ${LOG_PATH}

# accelerate launch --config_file accelerate_config.yaml  ../src/train.py \
# accelerate launch --config_file accelerate_config.yaml  ../src/train.py \ #! 测试的时候记得改
# python -u src/train_decoder.py \
# accelerate launch --config_file accelerate_training_fsdp.yaml src/train_decoder.py \
# python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
# deepspeed --num_gpus=1 src/train_decoder.py
torchrun --nnodes=2 --nproc_per_node=8  --master_addr=${CHIEF_IP} --master_port=50005 src/train_decoder.py \
    --model_name_or_path ${PRETRAINED_MODEL} \
    --do_train \
    --do_eval \
    --output_dir ${OUTPUT_MODEL_DIR} \
    --train_file ${TRAIN_DATASET_PATH} \
    --validation_file ${VALID_DATASET_PATH} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATE} \
    --learning_rate ${LR} \
    --num_train_epochs ${EPOCHS} \
    --warmup_ratio ${WARMUP_RATIO} \
    --seed ${SEED} \
    --metric_for_best_model ${METRIC_FOR_BEST_MODEL} \
    --evaluation_strategy epoch \
    --save_strategy steps \
    --save_steps 2000 \
    --overwrite_output_dir \
    --pad_to_max_length \
    --save_total_limit 2 \
    --run_name ${WANDB_RUN_NAME} \
    --max_train_samples 100 \
    --max_eval_samples 10 \
    --fp16 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LLaMADecoderLayer' 2>&1 | tee ${LOG_PATH}/${WANDB_RUN_NAME}-${INDEX}.log
    # --report_to wandb \


    # --predict_with_generate
