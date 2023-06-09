set -ve

#! global var
BASE_DIR=~/LLM_Pipeline_Toolkit


#! eval args
LLM_EVAL_PROJECT=${BASE_DIR}/lm-evaluation-harness
export HF_DATASETS_CACHE=${LLM_EVAL_PROJECT}/huggingface_datasets_cache_dir

#! ddp
DS_BASE_DIR=${BASE_DIR}/ds_config


#! assign in function
OUTPUT_MODEL_DIR=NULL
OUTPUT_FILE_BASE_DIR=NULL
PRETRAINED_MODEL=NULL


DEVICE=0,1,2,3,4,5,6,7


train_fn(){
    MODEL_PATH_SUFFIX=${1}
    MODEL_NAME=${2}
    DATA_PATH_SUFFIX=${3}
    DATASET_TYPE=${4}
    TRAIN_BATCH_SIZE=${5}
    EVAL_BATCH_SIZE=${5}



    PRETRAINED_MODEL=${BASE_DIR}/${MODEL_PATH_SUFFIX}

    #! DATA ARGS
    DATA_DIR=${BASE_DIR}/datas
    OUTPUT_BASE_DIR=${BASE_DIR}/checkpoints
    LOG_DIR=${BASE_DIR}/logs/training
    DATA_PATH=${DATA_DIR}/${DATA_PATH_SUFFIX}
    

    mkdir -p ${LOG_DIR}

    
    MAX_SEQ_LENGTH=512
    
    #! TRAINING ARGS
    EPOCHS=${11} #! lora 5 epochs, full tuning 3 epochs
    SEED=42 #! default 42
    LR=${6} #! 2e-5 for full training, 1e-4 for lora or prompt tuning 
    WARMUP_RATIO=0.03 
    GRADIENT_ACCUMULATE=${7}
    PROMPT_TYPE=stanford #! stanford or BELLE or origin

    #! WANDB
    export WANDB_DISABLED=true #! setting false if you use wandb to log
    WANDB_PROJECT_NAME=xxxxx
    ENTITY=xxxxx
    WANDB_RUN_NAME=${MODEL_NAME}-prompt-type_${PROMPT_TYPE}-train-batch_${TRAIN_BATCH_SIZE}-grad-acc_${GRADIENT_ACCUMULATE}-lr_${LR}-epochs_${EPOCHS}

    MODE=${8}
    IS_EXTRA_LARGE=${9}
    LORA_TRAIN=${10}
    PROMPT_TRAIN=${11}
    PROMPT_TOKENS=${12}
    ONLY_TUNE_EMBED=${13}
    
    
    #!OUTPUT
    if [ "${LORA_TRAIN}" = 'True' ]; then
        TRAINING_TYPE=lora
        DEEPSPEED_CONFIG=${DS_BASE_DIR}/deepspeed_zero1_config.json
        CONVERT_CKPT=False
    elif [ "${PROMPT_TRAIN}" = 'True' ]; then
        TRAINING_TYPE=prompt
        DEEPSPEED_CONFIG=${DS_BASE_DIR}/deepspeed_zero1_config.json
        CONVERT_CKPT=False
    else
        TRAINING_TYPE=full
        DEEPSPEED_CONFIG=${DS_BASE_DIR}/deepspeed_zero3_offload_config.json
        CONVERT_CKPT=True
    fi

    OUTPUT_MODEL_DIR=${OUTPUT_BASE_DIR}/${TRAINING_TYPE}/${MODEL_NAME}/${DATASET_TYPE}/${WANDB_RUN_NAME}


    mkdir -p "${OUTPUT_MODEL_DIR}"

    #! multi-node & multi-gpu
    # deepspeed --num_gpus=8 --master_addr=${CHIEF_IP} --master_port=50002 src/official_train.py 

    #! single-node & multi-gpu
    if [ "${MODE}" = 'train' ]; then
        deepspeed --include=localhost:${DEVICE} src/train.py \
            --deepspeed ${DEEPSPEED_CONFIG} \
            --prompt_type ${PROMPT_TYPE} \
            --model_name_or_path "${PRETRAINED_MODEL}" \
            --data_path "${DATA_PATH}" \
            --fp16 True \
            --seed ${SEED} \
            --output_dir "${OUTPUT_MODEL_DIR}" \
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
            --lora_train "${LORA_TRAIN}" \
            --prompt_train ${PROMPT_TRAIN} \
            --is_extra_large_model "${IS_EXTRA_LARGE}" \
            --only_tune_embedding "${ONLY_TUNE_EMBED}" \
            --prompt_tokens "${PROMPT_TOKENS}" \
            2>&1 | tee ${LOG_DIR}/"${WANDB_RUN_NAME}".log

        if [ "${CONVERT_CKPT}" = 'True' ]; then
            #! convert 
            CHECKPOINT_PATH=${OUTPUT_MODEL_DIR}
            OUTPUT_FILE=${CHECKPOINT_PATH}/pytorch_model.bin
            #! Make sure have enough cpu memory
            python src/zero_to_fp32.py --checkpoint_dir ${CHECKPOINT_PATH} --output_file ${OUTPUT_FILE}
        fi
    fi

}


ddp_inference_fn(){
    TEMPERATURE=0.35
    TOP_P=0.85
    TOP_K=40
    NUM_BEAMS=1
    REPETITION_PENALTY=1.2
    MAX_SEQ_LEN=256

    DECODING_MODE=${9} #! greedy or sample
    USE_LORA=${10} #! loading lora weight or not - only use pre-trained model weight
    PROMPT_TYPE=${11} #! stanford, origin, BELLE
    MODEL_TYPE=${1} #! FLAN-T5, LLAMA, BELLE, GPT2, ALPACA, Chinese_ALPACA, ChatGLM, BLOOM
    EVAL_PRETRAIN=${12}
    USE_PROMPT=${13}

    #! eval pre-trained model or tuning model
    if [ "${EVAL_PRETRAIN}" = 'True' ]; then
        OUTPUT_FILE_BASE_DIR=${PRETRAINED_MODEL}
        TOKENIZER_PATH=${PRETRAINED_MODEL}
    else
        if [ "${USE_LORA}" = 'True' ]; then
            EVAL_TYPE=lora
            LORA_WEIGHTS_PATH=${OUTPUT_MODEL_DIR}
            OUTPUT_FILE_BASE_DIR=${LORA_WEIGHTS_PATH}
            TOKENIZER_PATH=${PRETRAINED_MODEL}
        elif [ "${USE_PROMPT}" = 'True' ]; then
            EVAL_TYPE=prompt
            PROMPT_WEIGHTS_PATH=${OUTPUT_MODEL_DIR}
            OUTPUT_FILE_BASE_DIR=${PROMPT_WEIGHTS_PATH}
            TOKENIZER_PATH=${PRETRAINED_MODEL}
        else
            #! fully-finetune
            EVAL_TYPE=full
            OUTPUT_FILE_BASE_DIR=${OUTPUT_MODEL_DIR}
            PRETRAINED_MODEL=${OUTPUT_MODEL_DIR}
            TOKENIZER_PATH=${OUTPUT_MODEL_DIR}
            LORA_WEIGHTS_PATH=None
        fi
    fi

    #! change test file
    TEST_FILE_PATH=${BASE_DIR}/${2}
    TEST_FILE_NAME=`echo ${TEST_FILE_PATH} | awk -F '/' '{print $NF}'`
    DATA_TYPE=${3} #! zh_seed_task, self-instruct, en_zh_seed_tasks, X_CSQA, translation

    #! set example type
    if [ "${6}" = "True" ]; then 
        EXAMPLE_TYPE=use_example
    else
        EXAMPLE_TYPE=no_example
    fi


    #! set output_file_dir
    OUTPUT_FILE_DIR=${OUTPUT_FILE_BASE_DIR}/predictions/${DATA_TYPE}/${PROMPT_TYPE}/${EXAMPLE_TYPE}

    mkdir -p ${OUTPUT_FILE_DIR}

    OUTPUT_FILE_PATH=${OUTPUT_FILE_DIR}/${MODEL_TYPE}-for-${TEST_FILE_NAME}_prediction.jsonl
    LOG_PATH=${BASE_DIR}/logs/inference/${EVAL_TYPE}/${DATA_TYPE}/${MODEL_NAME}/${PROMPT_TYPE}/${EXAMPLE_TYPE}


    mkdir -p ${LOG_PATH}

    MODEL_NAME=`echo ${PRETRAINED_MODEL} | awk -F '/' '{print $NF}'`

    RUN_NAME=${MODEL_NAME}-${DATA_TYPE}-${PROMPT_TYPE}-temperature_${TEMPERATURE}-topk_${TOP_K}-topp_${TOP_P}-repetition_${REPETITION_PENALTY}-beams_${NUM_BEAMS}.log

    export CUDA_VISIBLE_DEVICES=${DEVICE}
    #! ddp batch size
    BATCH_SIZE=8

    #! deepspeed --include=localhost:GPU_RANK
    deepspeed --include=localhost:${DEVICE} src/ddp_batch_inference.py \
        --batch_size ${BATCH_SIZE} \
        --test_file_path ${TEST_FILE_PATH} \
        --output_file_path ${OUTPUT_FILE_PATH} \
        --pretrained_model_path ${PRETRAINED_MODEL} \
        --lora_weights_path ${LORA_WEIGHTS_PATH} \
        --prompt_weights_path ${PROMPT_WEIGHTS_PATH} \
        --pretrained_tokenizer_path ${TOKENIZER_PATH} \
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
        --deepspeed_inference True \
        --use_lora ${USE_LORA} \
        --use_prompt ${USE_PROMPT} \
        --add_lang "${4}" \
        --is_extra_large_model "${5}" \
        --use_demonstration "${6}" \
        --decoding_mode ${DECODING_MODE} \
        --use_probing_strategy "${7}" \
        --probing_text "${8}" \
        2>&1 | tee ${LOG_PATH}/${RUN_NAME} 
}


llm_harness_eval(){

    USE_LORA=${1} #! loading lora weight or not - only use pre-trained model weight
    USE_PROMPT=${2}
    USE_EXP=${3}
    EVAL_DATA_TYPES=${4}    #! piqa,arc_easy,arc_challenge,rte,cb,wsc273,winogrande 
    #! batch size
    BATCH_SIZE=${5}
    EVAL_PRETRAIN=${6}
    
    #! eval pre-trained model or tuning model
    if [ "${EVAL_PRETRAIN}" = 'True' ]; then
        OUTPUT_FILE_BASE_DIR=${PRETRAINED_MODEL}
        TASK_DESCRIPTION_FILE=None
        MODEL_ARGS="pretrained=${PRETRAINED_MODEL}"
    else
        TASK_DESCRIPTION_FILE=${LLM_EVAL_PROJECT}/task_description.json
        if [ "${USE_LORA}" = 'True' ] || [ "${USE_PROMPT}" = 'True' ]; then
            ADAPTER_WEIGHTS_PATH=${OUTPUT_MODEL_DIR}
            OUTPUT_FILE_BASE_DIR=${ADAPTER_WEIGHTS_PATH}
            MODEL_ARGS="pretrained=${PRETRAINED_MODEL},peft=${ADAPTER_WEIGHTS_PATH}"
        else
            OUTPUT_FILE_BASE_DIR=${OUTPUT_MODEL_DIR}
            MODEL_ARGS="pretrained=${OUTPUT_MODEL_DIR}"
        fi
    fi

    #! set example type
    if [ "${USE_EXP}" = "True" ]; then 
        EXAMPLE_TYPE=use_example
        NUM_EXP=5
    else
        EXAMPLE_TYPE=no_example
        NUM_EXP=0
    fi

    #! set output_file_dir
    OUTPUT_FILE_DIR=${OUTPUT_FILE_BASE_DIR}/metrics/llm_harness_eval/${EXAMPLE_TYPE}
    mkdir -p ${OUTPUT_FILE_DIR}
    OUTPUT_FILE_PATH=${OUTPUT_FILE_DIR}/${EVAL_DATA_TYPES}-metric.json
    

    python ${LLM_EVAL_PROJECT}/main.py \
        --model hf-causal-experimental \
        --model_args ${MODEL_ARGS} \
        --tasks ${EVAL_DATA_TYPES} \
        --no_cache \
        --batch_size ${BATCH_SIZE} \
        --output_path ${OUTPUT_FILE_PATH} \
        --num_fewshot ${NUM_EXP} \
        --description_dict_path ${TASK_DESCRIPTION_FILE} \
        --device cuda:0
# 
}

#! MODEL_ARGS
declare -a MODEL_NAMES
MODEL_NAMES=('pre-trained model name, e.g. LLAMA')
declare -a PRETRAINED_MODELS
PRETRAINED_MODELS=('pre-trained model path, e.g. pretrained_models/llama/7b')
declare -a TRAIN_DATASETS

#! change dataset file path here!!
TRAIN_DATASETS=('train dataset path, e.g. alpaca_data.json')

TRAIN_DATASET_TYPES=('train dataset type, e.g. alpaca_data')

LRS=('1e-4')
USE_LORAS=('True')
USE_PROMPT=('False')
PROMPT_TOKENS=('100')
ONLY_TUNE_EMBED=('False')
EPOCHS=('4')


train_eval_llm_harness(){
       
    EXTRA_LARGE=False
    USE_EXPS=('False' 'True')
    #! eval pre-trained model, then eval instruction tuned model
    EVAL_PRETRAINS=('True' 'False')
    MODE=eval
    EVAL_DATA_TYPES=rte,cb #! rte,cb,arc_easy,piqa,arc_challenge
    for ((i=0;i<${#MODEL_NAMES[@]};i++))
    do 
        for j in ${!TRAIN_DATASETS[@]}
        do
            train_fn "${PRETRAINED_MODELS[${i}]}" "${MODEL_NAMES[${i}]}" "${TRAIN_DATASETS[${j}]}" "${TRAIN_DATASET_TYPES[${j}]}" "${BATCH_SIZES[${i}]}" "${LRS[${i}]}" "${GRAD_ACC[${i}]}" "${MODE}" "${EXTRA_LARGE}" "${USE_LORAS[${i}]}" "${EPOCHS[${i}]}" "${USE_LORAS[${i}]}" "${PROMPT_TOKENS[${i}]}" "${ONLY_TUNE_EMBED[${i}]}"
            
            for EVAL_PRETRAIN in ${EVAL_PRETRAINS[@]}
            do
                for USE_EXP in ${USE_EXPS[@]}
                do
                    llm_harness_eval "${USE_LORAS[${i}]}" "${USE_EXP}" "${EVAL_DATA_TYPES}" "${BATCH_SIZES[${i}]}" "${EVAL_PRETRAIN}"
                done
            done
        done
    done
}




train_eval_generation(){
    declare -a EVAL_DATA_TYPES=('self-instruct') 
    DECODING_MODE=sample #! greedy or sample
    EXTRA_LARGE=False
    USE_EXPS=('False') #! corresponding eval dataset, now only for CSQA dataset

    MODE=${1} #! train or eval
    PROB_DECODE=False
    PROB_TEXT=False
    PROMPT_TYPE=stanford #! stanford or origin
    EVAL_PRETRAIN=False


    for ((i=0;i<${#MODEL_NAMES[@]};i++))
    do 
        for j in ${!TRAIN_DATASETS[@]}
        do
            train_lora "${PRETRAINED_MODELS[${i}]}" "${MODEL_NAMES[${i}]}" "${TRAIN_DATASETS[${j}]}" "${TRAIN_DATASET_TYPES[${j}]}" "${BATCH_SIZES[${i}]}" "${LRS[${i}]}" "${GRAD_ACC[${i}]}" "${MODE}" "${EXTRA_LARGE}" "${USE_LORAS[${i}]}" "${EPOCHS[${i}]}" #! modify mode
            for EVAL_DATA_TYPE in ${EVAL_DATA_TYPES[@]}
            do
                for ADD_LANG in False
                do 
                    for USE_EXP in ${USE_EXPS[@]}
                    do
                        TEST_FILE=datas/alpaca_instruction_data/text-davinci-003_predictions.jsonl
                        
                            
                        ddp_inference_fn "${MODEL_NAMES[${i}]}+${TRAIN_DATASET_TYPES[${j}]}" "${TEST_FILE}" "${EVAL_DATA_TYPE}" "${ADD_LANG}" "${EXTRA_LARGE}" "${USE_EXP}" "${PROB_DECODE}" "${PROB_TEXT}" "${DECODING_MODE}" "${USE_LORAS[${i}]}" "${PROMPT_TYPE}" "${EVAL_PRETRAIN}"
                
                    done
                done
            done
        done
    done
}



train_eval_generation train
train_eval_llm_harness eval

