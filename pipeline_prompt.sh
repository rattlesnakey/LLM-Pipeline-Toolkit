set -v
#! set- e will cut down pid if any command fail
#! global var
BASE_DIR=/apdcephfs/share_916081/effidit_shared_data/chillzhang

#! assign in train function
OUTPUT_MODEL_DIR=None
PRETRAINED_MODEL=None
LORA_WEIGHTS_PATH=None
# PROMPT_WEIGHTS_PATH=None
LORA_TRAIN=None
PROMPT_TRAIN=None

DEVICE=0,1,2,3,4,5,6,7
# DEVICE=0,1

train_adapter(){
    PRETRAINED_MODEL=${BASE_DIR}/${1}
    MODEL_NAME=${2}
    # MODEL_NAME=`echo ${PRETRAINED_MODEL} | awk -F '/' '{print $NF}'`

    #! DATA ARGS
    DATA_DIR=${BASE_DIR}/datas/alpaca_instruction_data
    OUTPUT_BASE_DIR=${BASE_DIR}/projects/LLM_inference/checkpoints
    LOG_DIR=${BASE_DIR}/projects/LLM_inference/logs/training
    DATA_PATH=${DATA_DIR}/${3}
    DATASET_TYPE=${4}

    mkdir -p ${LOG_DIR}

    
    MAX_SEQ_LENGTH=512

    LORA_TRAIN_BATCH_SIZE=2
    PROMPT_TRAIN_BATCH_SIZE=2

    LORA_EVAL_BATCH_SIZE=2
    PROMPT_EVAL_BATCH_SIZE=2

    #! TRAINING ARGS
    LORA_EPOCHS=4 #! lora 4 epochs
    PROMPT_EPOCHS=5 #! 8 epochs

    SEED=42 #! default 42

    LR_LORA=1e-4
    LR_PROMPT=1e-2 #! 2e-5 for full training, 3e-4 or 2e-4 or 1e-4 for lora, prompt tuning 我记得要 1e-2 才能训的起来
    WARMUP_RATIO=0.03 # 0.1,0.2


    LORA_GRADIENT_ACCUMULATE=8
    PROMPT_GRADIENT_ACCUMULATE=8

    PROMPT_TYPE=stanford #! stanford or BELLE or origin

    #! WANDB
    export WANDB_DISABLED=true
    WANDB_PROJECT_NAME=instruction-tuning-llm
    ENTITY=hengyuan
    WANDB_RUN_NAME_LORA=official-${MODEL_NAME}+${3}-prompt-type_${PROMPT_TYPE}-train-batch_${LORA_TRAIN_BATCH_SIZE}-lr_${LR_LORA}-epochs_${LORA_EPOCHS}
    WANDB_RUN_NAME_PROMPT=official-${MODEL_NAME}+${3}-prompt-type_${PROMPT_TYPE}-train-batch_${PROMPT_TRAIN_BATCH_SIZE}-lr_${LR_PROMPT}-epochs_${PROMPT_EPOCHS}
    
    # export WANDB_PROJECT=${WANDB_PROJECT_NAME}
    # export WANDB_ENTITY=${ENTITY}
    IS_EXTRA_LARGE_MODEL=${6}
    LORA_TRAIN=${7}
    PROMPT_TRAIN=${8}
    PROMPT_TOKENS=${9}
    SPEC_ATT=${10} #! special attention mechanism

    #!OUTPUT
    #! 不考虑单独 prompt tuning, 只考虑 prompt tuning with lora weight merging
    if [ ${LORA_TRAIN} = 'True' ] && [ ${PROMPT_TRAIN} = 'True' ]; then
        echo 'Cannot simultaneously set LORA_TRAIN=True and PROMPT_TRAIN=True'
        exit
        # OUTPUT_MODEL_DIR=${OUTPUT_BASE_DIR}/prompt_with_lora/one_stage/${MODEL_NAME}/${DATASET_TYPE}/${WANDB_RUN_NAME}
        # MERGE_LORA=False #! ONLY when PROMPT_TRAIn = True, need to set MERGE_LORA
        # LORA_WEIGHTS_PATH=None #! only output prompt weights, prompt weight should include lora weights
    elif [ ${LORA_TRAIN} = 'True' ] && [ ${PROMPT_TRAIN} = 'False' ]; then
        #! Only Lora training
        OUTPUT_MODEL_DIR=${OUTPUT_BASE_DIR}/lora/${MODEL_NAME}/${DATASET_TYPE}/${WANDB_RUN_NAME_LORA}
        LR=${LR_LORA}
        EPOCHS=${LORA_EPOCHS}
        GRADIENT_ACCUMULATE=${LORA_GRADIENT_ACCUMULATE}
        TRAIN_BATCH_SIZE=${LORA_TRAIN_BATCH_SIZE}
        EVAL_BATCH_SIZE=${LORA_EVAL_BATCH_SIZE}
    
    elif [ ${LORA_TRAIN} = 'False' ] && [ ${PROMPT_TRAIN} = 'True' ]; then
        OUTPUT_MODEL_DIR=${OUTPUT_BASE_DIR}/prompt_with_lora_v3/two_stage/${MODEL_NAME}/spec_att_${SPEC_ATT}/${DATASET_TYPE}/${WANDB_RUN_NAME_PROMPT}-virtual-tokens_${PROMPT_TOKENS}
        #! 自动获取对应 Model name 和 data type 的 LORA WEIGHTS
        #! Lora 的 learning rate 和 prompt 不一样，所以 ..
        LORA_WEIGHTS_PATH=${OUTPUT_BASE_DIR}/lora/${MODEL_NAME}/${DATASET_TYPE}/${WANDB_RUN_NAME_LORA}
        LR=${LR_PROMPT}
        EPOCHS=${PROMPT_EPOCHS}
        GRADIENT_ACCUMULATE=${PROMPT_GRADIENT_ACCUMULATE}
        TRAIN_BATCH_SIZE=${PROMPT_TRAIN_BATCH_SIZE}
        EVAL_BATCH_SIZE=${PROMPT_EVAL_BATCH_SIZE}
        
    fi

    mkdir -p ${OUTPUT_MODEL_DIR}
    # DEVICE=0
    export CUDA_VISIBLE_DEVICES=${DEVICE}


    #! multi-node & multi-gpu
    # deepspeed --num_gpus=8 --master_addr=${CHIEF_IP} --master_port=50002 src/official_train.py 

    #! single-node & multi-gpu
    # deepspeed --num_gpus=8 

    #! zero1 + fp16 for 7b lora, prompt tuning
    #! 13b need to add "is_extra_large_model"
    #! replace zero config file
    

    if [ "${5}" = 'train' ]; then
        # deepspeed --include=localhost:${DEVICE} src/official_train_v5.py \
        # --deepspeed deepspeed_nonzero_config.json \
        # python -u src/official_train_v5.py \
        deepspeed --include=localhost:${DEVICE} src/official_train_v7.py \
            --deepspeed deepspeed_zero1_config.json \
            --prompt_type ${PROMPT_TYPE} \
            --model_name_or_path ${PRETRAINED_MODEL} \
            --data_path ${DATA_PATH} \
            --fp16 True \
            --output_dir ${OUTPUT_MODEL_DIR} \
            --num_train_epochs ${EPOCHS} \
            --per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
            --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
            --gradient_accumulation_steps ${GRADIENT_ACCUMULATE} \
            --evaluation_strategy "no" \
            --model_max_length ${MAX_SEQ_LENGTH} \
            --save_strategy "steps" \
            --save_steps 5000 \
            --save_total_limit 1 \
            --learning_rate ${LR} \
            --weight_decay 0. \
            --warmup_ratio ${WARMUP_RATIO} \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --lora_train ${LORA_TRAIN} \
            --prompt_train ${PROMPT_TRAIN} \
            --is_extra_large_model ${IS_EXTRA_LARGE_MODEL} \
            --lora_weights_path ${LORA_WEIGHTS_PATH} \
            --prompt_tokens ${PROMPT_TOKENS} \
            --use_special_attention_mechanism ${SPEC_ATT} \
            2>&1 | tee ${LOG_DIR}/${WANDB_RUN_NAME}.log
    fi

}

ddp_inference_adapter(){
    TEMPERATURE=0.35
    TOP_P=0.85
    TOP_K=40
    NUM_BEAMS=1
    REPETITION_PENALTY=1.2
    MAX_SEQ_LEN=256

    MODEL_TYPE=${1} #! FLAN-T5, LLAMA, BELLE, GPT2, ALPACA, Chinese_ALPACA, ChatGLM, BLOOM

    BASE_DIR=/apdcephfs/share_916081/effidit_shared_data/chillzhang
    PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL}
    
    PRETRAINED_TOKENIZER_PATH=${PRETRAINED_MODEL_PATH}
    # #! change test file

    TEST_FILE_PATH=${BASE_DIR}/${2}

    TEST_FILE_NAME=`echo ${TEST_FILE_PATH} | awk -F '/' '{print $NF}'`

    DATA_TYPE=${3} #! zh_seed_task, self-instruct, en_zh_seed_tasks, X_CSQA
    ADD_LANG=${4}
    EXTRA_LARGE=${5}
    USE_EXP=${6}
    SPEC_ATT=${7}
    #! global
    USE_LORA=${LORA_TRAIN}
    USE_PROMPT=${PROMPT_TRAIN}
    ADAPTER_WEIGHTS_PATH=${OUTPUT_MODEL_DIR}
    if [ ${USE_LORA} = 'True' ] && [ ${USE_PROMPT} = 'True' ]; then
        echo 'Cannot simultaneously set LORA_TRAIN=True and PROMPT_TRAIN=True'
        exit
    fi
    
    #! set example type
    # EXAMPLE_TYPE=use_example
    if [ "${USE_EXP}" = "True" ]; then 
        EXAMPLE_TYPE=use_example
    else
        EXAMPLE_TYPE=no_example
    fi
        
    # DATA_TYPE=en_instruct_zh_input_seed_task 
    PROMPT_TYPE=stanford #! stanford, origin, BELLE

    #! instruction 的语言
    SRC_LANG=en
    #! response 的语言
    TGT_LANG=zh
    
    #! set output_file_dir
    # "$str1" == "$prefix"* 
    # "${DATA_TYPE}" = "X_CSQA"
    if [[ "${DATA_TYPE}" == *"X_"* ]]; then
        LANG=$(basename $(dirname ${TEST_FILE_PATH}))
        if [ "${ADD_LANG}" = "True" ]; then
            OUTPUT_FILE_DIR=${ADAPTER_WEIGHTS_PATH}/predictions/${DATA_TYPE}/${LANG}/${SRC_LANG}-${TGT_LANG}/${PROMPT_TYPE}/${EXAMPLE_TYPE}
        else
            OUTPUT_FILE_DIR=${ADAPTER_WEIGHTS_PATH}/predictions/${DATA_TYPE}/${LANG}/no-add-lang/${PROMPT_TYPE}/${EXAMPLE_TYPE}
        fi
        
    else 
        if [ "${ADD_LANG}" = "True" ]; then
            OUTPUT_FILE_DIR=${ADAPTER_WEIGHTS_PATH}/predictions/${DATA_TYPE}/${SRC_LANG}-${TGT_LANG}/${PROMPT_TYPE}/${EXAMPLE_TYPE}
        else
            OUTPUT_FILE_DIR=${ADAPTER_WEIGHTS_PATH}/predictions/${DATA_TYPE}/no-add-lang/${PROMPT_TYPE}/${EXAMPLE_TYPE}
        fi
    fi
    
    

    mkdir -p ${OUTPUT_FILE_DIR}

    OUTPUT_FILE_PATH=${OUTPUT_FILE_DIR}/${MODEL_TYPE}-for-${TEST_FILE_NAME}_prediction.jsonl
    LOG_PATH=${BASE_DIR}/projects/LLM_inference/logs/inference/prompt/${DATA_TYPE}/${MODEL_NAME}/${PROMPT_TYPE}/${EXAMPLE_TYPE}


    mkdir -p ${LOG_PATH}

    MODEL_NAME=`echo ${PRETRAINED_MODEL_PATH} | awk -F '/' '{print $NF}'`

    RUN_NAME=${MODEL_NAME}-${DATA_TYPE}-${PROMPT_TYPE}-temperature_${TEMPERATURE}-topk_${TOP_K}-topp_${TOP_P}-repetition_${REPETITION_PENALTY}-beams_${NUM_BEAMS}.log



    export CUDA_VISIBLE_DEVICES=${DEVICE}
    #! ddp batch size
    BATCH_SIZE=4
    #! if use deepspeed, use deepspeed to launchd
    # deepspeed --num_gpus=1
    # python -u
    #! deepspeed --include=localhost:GPU_RANK


    deepspeed --include=localhost:${DEVICE} src/ddp_batch_inference_v3.py \
        --batch_size ${BATCH_SIZE} \
        --test_file_path ${TEST_FILE_PATH} \
        --output_file_path ${OUTPUT_FILE_PATH} \
        --pretrained_model_path ${PRETRAINED_MODEL_PATH} \
        --lora_weights_path ${LORA_WEIGHTS_PATH} \
        --adapter_weights_path ${ADAPTER_WEIGHTS_PATH} \
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
        --deepspeed_inference True \
        --use_lora ${USE_LORA} \
        --use_prompt ${USE_PROMPT} \
        --add_lang ${ADD_LANG} \
        --add_self_understand False \
        --is_extra_large_model ${EXTRA_LARGE} \
        --use_demonstration ${USE_EXP} \
        --use_special_attention_mechanism ${SPEC_ATT} \
        2>&1 | tee ${LOG_PATH}/${RUN_NAME}
        #    --add_lang \ 
}



#! MODEL_ARGS
# PRETRAINED_MODEL=${BASE_DIR}/pretrained_models/LLAMA/hf/llama-7b
declare -a MODEL_NAMES
#! 原本的 LLAMA 就默认代码 7b 的
# MODEL_NAMES=('Chinese-LLAMA')
# MODEL_NAMES=('GPT2-base')
MODEL_NAMES=('LLAMA')
# MODEL_NAMES=('LLAMA-13B')
# MODEL_NAMES=('LLAMA' 'Chinese-LLAMA')
# MODEL_NAMES=('Extended-LLAMA-translation-concat-10w')
# MODEL_NAMES=('Extended-LLAMA-translation-instruction-10w')
# MODEL_NAMES=('Extended-zh-vocab-LLAMA-translation-instruction-10w')
# MODEL_NAMES=('Extended-zh-vocab-LLAMA-translation-concat-10w')
# MODEL_NAMES=('Extended-zh-vocab-LLAMA-no-post-pretraining')
declare -a PRETRAINED_MODELS
# PRETRAINED_MODELS=('pretrained_models/gpt2/gpt2-base')
PRETRAINED_MODELS=('pretrained_models/LLAMA/hf/llama-7b')
# PRETRAINED_MODELS=('pretrained_models/LLAMA/hf/llama-13b')
# PRETRAINED_MODELS=('pretrained_models/extended_zh_vocab_llama')
# PRETRAINED_MODELS=('pretrained_models/LLAMA+translation/official-llama-7b-post_training_translation_instruction_100000.json-prompt-type_origin-train-batch_2-lr_1e-4-epochs_5')
# PRETRAINED_MODELS=('pretrained_models/extended_zh_vocab_llama+translation/official-extended_zh_vocab_llama-post_training_translation_concat_100000.json-prompt-type_origin-train-batch_2-lr_1e-4-epochs_5')
# PRETRAINED_MODELS=('pretrained_models/chinese_llama/hf/chinese-llama-7b')
# PRETRAINED_MODELS=('pretrained_models/LLAMA/hf/llama-7b' 'pretrained_models/chinese_llama/hf/chinese-llama-7b')
# PRETRAINED_MODELS=('pretrained_models/gpt2/gpt2-base' 'pretrained_models/gpt2/gpt2-base')
# PRETRAINED_MODELS=('pretrained_models/bloom/bloom-7b-mt' 'pretrained_models/LLAMA/hf/llama-7b')

declare -a TRAIN_DATASETS
#! change dataset file path here!!
# DATASETS=('test_pipeline_data.json' 'test_pipeline_500.json')
# DATASETS=('test_pipeline_500.json')

# TRAIN_DATASETS=('alpaca_data+zh_alpaca_data_revised.json')
# DATASETS=('alpaca_data.json' 'alpaca_data+en2zh+zh2en-not-same-1000.json' 'alpaca_data+en2zh+zh2en-not-same-10000.json' 'alpaca_data+en2zh+zh2en-not-same-5000.json' 'alpaca_data+wmt-en2zh-1000.json' 'alpaca_data+wmt-en2zh-10000.json' 'alpaca_data+wmt-en2zh-5000.json')
# DATASETS=('alpaca_data.json' 'alpaca_data+en2zh+zh2en-not-same-1000.json' 'alpaca_data+en2zh+zh2en-not-same-10000.json' 'alpaca_data+en2zh+zh2en-not-same-5000.json' 'alpaca_data+wmt-en2zh-1000.json' 'alpaca_data+wmt-en2zh-10000.json' 'alpaca_data+wmt-en2zh-5000.json')
# DATASETS=('alpaca_data+en2zh+zh2en-not-same-50000.json' 'alpaca_data+en2zh+zh2en-not-same-100000.json' 'alpaca_data+en2zh+zh2en-not-same-25000.json' 'alpaca_data+en2zh-25000.json' 'alpaca_data+en2zh-50000.json' 'alpaca_data+en2zh-100000.json')
# DATASETS=('alpaca_data+en2zh-100000.json')
# DATASETS=('alpaca_data.json' 'alpaca_data+en2zh-50.json' 'alpaca_data+en2zh-100.json' 'alpaca_data+en2zh-200.json' 'alpaca_data+en2zh-300.json' 'alpaca_data+en2zh-400.json' 'alpaca_data+en2zh-500.json' 'alpaca_data+en2zh+zh2en-not-same-50.json' 'alpaca_data+en2zh+zh2en-not-same-100.json' 'alpaca_data+en2zh+zh2en-not-same-200.json' 'alpaca_data+en2zh+zh2en-not-same-300.json' 'alpaca_data+en2zh+zh2en-not-same-400.json' 'alpaca_data+en2zh+zh2en-not-same-500.json' 'alpaca_data+en2zh-25000.json' 'alpaca_data+en2zh-50000.json' 'alpaca_data+en2zh-100000.json' 'alpaca_data+en2zh+zh2en-not-same-25000.json' 'alpaca_data+en2zh+zh2en-not-same-50000.json' 'alpaca_data+en2zh+zh2en-not-same-10000.json')
TRAIN_DATASETS=('alpaca_data.json')
# TRAIN_DATASETS=('test_pipeline_data.json')
# TRAIN_DATASET_TYPES=('test')
TRAIN_DATASET_TYPES=('alpaca_data')
# TRAIN_DATASET_TYPES=('alpaca_data+zh_alpaca_data')


eval_x_csqa(){
    # declare -a EVAL_DATA_TYPES=('X_CSQA' 'X_CSQA_ARC_Easy' 'X_CSQA_ARC_Challenge' 'X_CSQA_PIQA') #'X_CSQA''X_CSQA_ARC_Easy' 'X_CSQA_ARC_Challenge'
    declare -a EVAL_DATA_TYPES=('X_CSQA') #'X_CSQA''X_CSQA_ARC_Easy' 'X_CSQA_ARC_Challenge' 
    # declare -a TEST_LANGS=('en' 'zh' 'ar' 'de' 'es' 'fr' 'hi' 'it' 'jap' 'nl' 'pl' 'pt' 'ru' 'sw' 'ur' 'vi')
    declare -a TEST_LANGS=('en')
    
    EXTRA_LARGE=False
    USE_EXP=False

    LORA_TRAIN=False #! if LORA_TRAIN=TRUE, PROMPT_TRAIN need to be False, vice versa
    PROMPT_TRAIN=True
    PROMPT_TOKENS=100
    MODE=eval #! eval or train
    # SPEC_ATT=True #! only for prompt train
    SPEC_ATTS=('True')
    for ((i=0;i<${#MODEL_NAMES[@]};i++))
    do 
        for j in ${!TRAIN_DATASETS[@]}
        do
            for SPEC_ATT in ${SPEC_ATTS[@]}
            do
                train_adapter "${PRETRAINED_MODELS[${i}]}" "${MODEL_NAMES[${i}]}" "${TRAIN_DATASETS[${j}]}" "${TRAIN_DATASET_TYPES[${j}]}" "${MODE}" "${EXTRA_LARGE}" "${LORA_TRAIN}" "${PROMPT_TRAIN}" "${PROMPT_TOKENS}" "${SPEC_ATT}" #! modify mode
                for EVAL_DATA_TYPE in ${EVAL_DATA_TYPES[@]}
                do
                    for ADD_LANG in False
                    do 
                        for LANG in ${TEST_LANGS[@]}
                        do
                            TEST_FILE=datas/X-CSR_datasets/${EVAL_DATA_TYPE}/${LANG}/dev_v2.jsonl #! 有的是 dev.jsonl, 有的是 dev_v2.jsonl
                            
                            # ddp_inference_adapter "${MODEL_NAMES[${i}]}+${TRAIN_DATASETS[${j}]}" "${TEST_FILE}" "${EVAL_DATA_TYPE}" "${ADD_LANG}" "${EXTRA_LARGE}" "${USE_EXP}" "${SPEC_ATT}"
                        done

                        # ! get_metric after all lang inferencing
                        echo 'getting metrics ...'
                        RESULT_DIR=${OUTPUT_MODEL_DIR}/predictions/${EVAL_DATA_TYPE}
                        METRIC_DIR=${OUTPUT_MODEL_DIR}/metrics/${EVAL_DATA_TYPE}
                        mkdir -p ${METRIC_DIR}
                        python -u src/eval_XQA.py \
                            --result_dir ${RESULT_DIR} \
                            --metric_dir ${METRIC_DIR} 
                    done
                done
            done
        done
    done
    
}

eval_x_nli(){
    declare -a EVAL_DATA_TYPES=('X_NLI')
    declare -a TEST_LANGS=('en' 'zh' 'ar' 'bg' 'de' 'el' 'es' 'fr' 'hi' 'ru' 'sw' 'th' 'tr' 'ur' 'vi')
    # declare -a TEST_LANGS=('en')

   
    EXTRA_LARGE=False
    USE_EXP=False

    LORA_TRAIN=False #! if LORA_TRAIN=TRUE, PROMPT_TRAIN need to be False, vice versa
    PROMPT_TRAIN=True
    PROMPT_TOKENS=100
    MODE=eval #! eval or train
    # SPEC_ATT=True #! only for prompt train
    SPEC_ATTS=('True')
    for ((i=0;i<${#MODEL_NAMES[@]};i++))
    do 
        for j in ${!TRAIN_DATASETS[@]}
        do
            for SPEC_ATT in ${SPEC_ATTS[@]}
            do
                train_adapter "${PRETRAINED_MODELS[${i}]}" "${MODEL_NAMES[${i}]}" "${TRAIN_DATASETS[${j}]}" "${TRAIN_DATASET_TYPES[${j}]}" "${MODE}" "${EXTRA_LARGE}" "${LORA_TRAIN}" "${PROMPT_TRAIN}" "${PROMPT_TOKENS}" "${SPEC_ATT}" #! modify mode
                for EVAL_DATA_TYPE in ${EVAL_DATA_TYPES[@]}
                do
                    for ADD_LANG in False
                    do 
                        for LANG in ${TEST_LANGS[@]}
                        do
                            TEST_FILE=datas/X-CSR_datasets/${EVAL_DATA_TYPE}/${LANG}/dev.jsonl #! 有的是 dev.jsonl, 有的是 dev_v2.jsonl
                            
                            ddp_inference_adapter "${MODEL_NAMES[${i}]}+${TRAIN_DATASETS[${j}]}" "${TEST_FILE}" "${EVAL_DATA_TYPE}" "${ADD_LANG}" "${EXTRA_LARGE}" "${USE_EXP}" "${SPEC_ATT}"
                        done

                        # ! get_metric after all lang inferencing
                        echo 'getting metrics ...'
                        RESULT_DIR=${OUTPUT_MODEL_DIR}/predictions/${EVAL_DATA_TYPE}
                        METRIC_DIR=${OUTPUT_MODEL_DIR}/metrics/${EVAL_DATA_TYPE}
                        mkdir -p ${METRIC_DIR}
                        python -u src/eval_XQA.py \
                            --result_dir ${RESULT_DIR} \
                            --metric_dir ${METRIC_DIR} 
                    done
                done
            done
        done
    done
    
}
eval_x_csqa
# eval_x_nli
# eval_x_csqa
# train_and_inference_different_size
# train_and_inference_different_dataset

