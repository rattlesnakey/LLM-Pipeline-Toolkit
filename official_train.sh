set -v
set -e



#! MODEL_ARGS
BASE_DIR=/apdcephfs/share_916081/effidit_shared_data/chillzhang
# PRETRAINED_MODEL=/workspace/share_916081/chillzhang/pretrained_models/GPT2/gpt2-base
# PRETRAINED_MODEL=/apdcephfs/share_916081/chillzhang/pretrained_models/GPT2/gpt2-base
# PRETRAINED_MODEL=${BASE_DIR}/pretrained_models/LLAMA/hf/llama-7b
PRETRAINED_MODEL=${BASE_DIR}/pretrained_models/bloom/bloom-7b-mt
# PRETRAINED_MODEL=/apdcephfs/share_916081/effidit_shared_data/chillzhang/pretrained_models/gpt2/gpt2-large
# PRETRAINED_MODEL=/apdcephfs/share_916081/chillzhang/pretrained_models/GPT2/gpt2-base
# MODEL_TYPE=M2M # mBART
MODEL_NAME=`echo ${PRETRAINED_MODEL} | awk -F '/' '{print $NF}'`
#! DATA ARGS

declare -a DATASETS
DATASETS=('alpaca_data.json' 'alpaca_data+en2zh.json' 'alpaca_data+en2zh+zh2en.json')

#! alpaca_data.json
for dataset_type in ${DATASETS[@]}
do
    DATA_PATH=/apdcephfs/share_916081/effidit_shared_data/chillzhang/datas/alpaca_instruction_data/${dataset_type}
    MAX_SEQ_LENGTH=512
    TRAIN_BATCH_SIZE=2
    EVAL_BATCH_SIZE=2

    #! TRAINING ARGS
    EPOCHS=3 #! 测试的时候记得修改
    SEED=42 #! 不传的话，默认是 42
    LR=2e-5 #! 2e-5  
    WARMUP_RATIO=0.03 # 0.1,0.2


    GRADIENT_ACCUMULATE=8

    #! WANDB
    export WANDB_DISABLED=true
    WANDB_PROJECT_NAME=instruction-tuning-llm
    ENTITY=hengyuan
    WANDB_RUN_NAME=official-${MODEL_NAME}-${dataset_type}-train-batch_${TRAIN_BATCH_SIZE}-lr_${LR}-epochs_${EPOCHS}
    # export WANDB_PROJECT=${WANDB_PROJECT_NAME}
    # export WANDB_ENTITY=${ENTITY}


    #!OUTPUT
    OUTPUT_MODEL_DIR=/apdcephfs/share_916081/effidit_shared_data/chillzhang/projects/LLM_inference/checkpoints/${WANDB_RUN_NAME}


    mkdir -p ${OUTPUT_MODEL_DIR}
    # DEVICE=0
    DEVICE=0,1,2,3,4,5,6,7 
    export CUDA_VISIBLE_DEVICES=${DEVICE}

    LOG_PATH=/apdcephfs/share_916081/effidit_shared_data/chillzhang/projects/LLM_inference/logs/training
    mkdir -p ${LOG_PATH}

    #! multi-node & multi-gpu
    # deepspeed --num_gpus=8 --master_addr=${CHIEF_IP} --master_port=50002 src/official_train.py 

    #! single-node & multi-gpu
    # deepspeed --num_gpus=8 
    # deepspeed_zero3_offload_config.json
    deepspeed --num_gpus=8 src/official_train.py \
        --deepspeed deepspeed_zero3_offload_config.json \
        --model_name_or_path ${PRETRAINED_MODEL} \
        --data_path ${DATA_PATH} \
        --fp16 False \
        --output_dir ${OUTPUT_MODEL_DIR} \
        --num_train_epochs ${EPOCHS} \
        --per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
        --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRADIENT_ACCUMULATE} \
        --evaluation_strategy "no" \
        --model_max_length ${MAX_SEQ_LENGTH} \
        --save_strategy "steps" \
        --save_steps 2000 \
        --save_total_limit 1 \
        --learning_rate ${LR} \
        --weight_decay 0. \
        --warmup_ratio ${WARMUP_RATIO} \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        2>&1 | tee ${LOG_PATH}/${WANDB_RUN_NAME}.log
        # --fsdp "full_shard auto_wrap" \
        # --fsdp_transformer_layer_cls_to_wrap 'LLaMADecoderLayer' 2>&1 | tee ${LOG_PATH}/${WANDB_RUN_NAME}-${INDEX}.log
        # --tf32 True
        #  --ddp_timeout 50000 \

    #! convert 
    CHECKPOINT_PATH=${OUTPUT_MODEL_DIR}

    OUTPUT_FILE=${CHECKPOINT_PATH}/pytorch_model.bin
    python src/zero_to_fp32.py --checkpoint_dir ${CHECKPOINT_PATH} --output_file ${OUTPUT_FILE}
done
