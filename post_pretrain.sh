set -v
set -e




post_pretrain(){
    #! MODEL_ARGS
    BASE_DIR=/apdcephfs/share_916081/effidit_shared_data/chillzhang
    # PRETRAINED_MODEL=${BASE_DIR}/pretrained_models/gpt2/gpt2-base
    # PRETRAINED_MODEL=${BASE_DIR}/pretrained_models/LLAMA/hf/llama-7b
    #! 要改 pretraining model
    PRETRAINED_MODEL=${BASE_DIR}/${1}
    # PRETRAINED_MODEL=${BASE_DIR}/pretrained_models/bloom/bloom-7b-mt
    # PRETRAINED_MODEL=${BASE_DIR}/pretrained_models/gpt2/gpt2-large

    MODEL_NAME=`echo ${PRETRAINED_MODEL} | awk -F '/' '{print $NF}'`

    #! DATA ARGS
    #! Ms. Bi change dataset info here!!
    DATA_DIR=${BASE_DIR}/datas/wmt23
    OUTPUT_BASE_DIR=${BASE_DIR}/${2}
    LOG_DIR=${BASE_DIR}/projects/LLM_inference/logs/training

    

    mkdir -p ${LOG_DIR}


    # DATASETS=('alpaca_data.json' 'alpaca_data+en2zh.json' 'alpaca_data+en2zh+zh2en.json')
    #! alpaca_data.json
    DATA_PATH=${DATA_DIR}/${3}
    MAX_SEQ_LENGTH=256
    TRAIN_BATCH_SIZE=2
    EVAL_BATCH_SIZE=2

    #! TRAINING ARGS
    EPOCHS=5
    SEED=42 #! default 42
    LR=1e-4 #! 2e-5  
    WARMUP_RATIO=0.03 # 0.1,0.2


    GRADIENT_ACCUMULATE=8
    PROMPT_TYPE=origin #! pretraining need to be origin

    #! WANDB
    export WANDB_DISABLED=true
    WANDB_PROJECT_NAME=instruction-tuning-llm
    ENTITY=hengyuan
    WANDB_RUN_NAME=official-${MODEL_NAME}-${DATASET}-prompt-type_${PROMPT_TYPE}-train-batch_${TRAIN_BATCH_SIZE}-lr_${LR}-epochs_${EPOCHS}
    # export WANDB_PROJECT=${WANDB_PROJECT_NAME}
    # export WANDB_ENTITY=${ENTITY}


    #!OUTPUT
    OUTPUT_MODEL_DIR=${OUTPUT_BASE_DIR}/${WANDB_RUN_NAME}


    mkdir -p ${OUTPUT_MODEL_DIR}
    # DEVICE=0
    DEVICE=0,1,2,3,4,5,6,7 
    export CUDA_VISIBLE_DEVICES=${DEVICE}


    #! multi-node & multi-gpu
    # deepspeed --num_gpus=8 --master_addr=${CHIEF_IP} --master_port=50002 src/official_train.py 

    #! single-node & multi-gpu
    # deepspeed --num_gpus=8 
    deepspeed --num_gpus=8 src/post_pretrain.py \
        --deepspeed deepspeed_zero3_config.json \
        --prompt_type ${PROMPT_TYPE} \
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
        --only_tune_embedding \
        --logging_steps 1 \
        2>&1 | tee ${LOG_DIR}/${WANDB_RUN_NAME}.log
        # --fsdp "full_shard auto_wrap" \
        # --fsdp_transformer_layer_cls_to_wrap 'LLaMADecoderLayer' 2>&1 | tee ${LOG_PATH}/${WANDB_RUN_NAME}-${INDEX}.log
        # --tf32 True
        #  --ddp_timeout 50000 \
        # --only_tune_embedding \

    #! convert 
    CHECKPOINT_PATH=${OUTPUT_MODEL_DIR}

    OUTPUT_FILE=${CHECKPOINT_PATH}/pytorch_model.bin
    python src/zero_to_fp32.py --checkpoint_dir ${CHECKPOINT_PATH} --output_file ${OUTPUT_FILE}

    #! combine embedding weigth to model
    #! 这个到时候跑完得重新跑一下
    python -u src/aggregate_embedding.py --embedding_ckpt_dir ${CHECKPOINT_PATH} --backbone_dir ${PRETRAINED_MODEL}

}

declare -a BACKBONES
# BACKBONES=('pretrained_models/extended_zh_vocab_llama' 'pretrained_models/LLAMA/hf/llama-7b')
BACKBONES=('pretrained_models/extended_zh_vocab_llama')

declare -a OUTPUT_DIRS
# OUTPUT_DIRS=('pretrained_models/extended_zh_vocab_llama+translation' 'pretrained_models/LLAMA+translation')
OUTPUT_DIRS=('pretrained_models/extended_zh_vocab_llama+translation')
declare -a DATASETS
# DATASETS=('post_training_translation_instruction_100000.json')
DATASETS=('post_training_translation_concat_100000.json')
for ((i=0;i<${#BACKBONES[@]};i++))
do 
    for DATASET in ${DATASETS[@]}
    do
        post_pretrain ${BACKBONES[${i}]} ${OUTPUT_DIRS[${i}]} ${DATASET}
    done
done

# post_pretrain ${BACKBONES[1]} ${OUTPUT_DIRS[1]} ${DATASETS[0]}
# pretrained_models/extended_zh_vocab_llama
# extended_zh_vocab_llama+translation