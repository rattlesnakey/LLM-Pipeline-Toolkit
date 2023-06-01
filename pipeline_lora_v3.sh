set -v
#! set- e will cut down pid if any command fail
#! global var
BASE_DIR=/apdcephfs/share_916081/effidit_shared_data/chillzhang
LLM_EVAL_PROJECT=${BASE_DIR}/projects/llm_eval_harness/lm-evaluation-harness
export HF_DATASETS_CACHE=${LLM_EVAL_PROJECT}/huggingface_datasets_cache_dir


#! assign in function
OUTPUT_MODEL_DIR=NULL
OUTPUT_FILE_BASE_DIR=NULL
PRETRAINED_MODEL=NULL


DEVICE=0,1,2,3,4,5,6,7
# DEVICE=0,1

train_lora(){
    MODEL_PATH_SUFFIX=${1}
    MODEL_NAME=${2}
    DATA_PATH_SUFFIX=${3}
    DATASET_TYPE=${4}
    TRAIN_BATCH_SIZE=${5}
    EVAL_BATCH_SIZE=${5}



    PRETRAINED_MODEL=${BASE_DIR}/${MODEL_PATH_SUFFIX}
    # MODEL_NAME=`echo ${PRETRAINED_MODEL} | awk -F '/' '{print $NF}'`

    #! DATA ARGS
    # DATA_DIR=${BASE_DIR}/datas/alpaca_instruction_data
    DATA_DIR=${BASE_DIR}/datas
    OUTPUT_BASE_DIR=${BASE_DIR}/projects/LLM_inference/checkpoints
    LOG_DIR=${BASE_DIR}/projects/LLM_inference/logs/training
    DATA_PATH=${DATA_DIR}/${DATA_PATH_SUFFIX}
    

    mkdir -p ${LOG_DIR}

    
    MAX_SEQ_LENGTH=512
    

    #! TRAINING ARGS
    EPOCHS=${11} #! lora 4 epochs
    SEED=42 #! default 42
    LR=${6} #! 2e-5 for full training, 3e-4 or 2e-4 or 1e-4 for lora
    WARMUP_RATIO=0.03 #! 0.03,0.1,0.2
    GRADIENT_ACCUMULATE=${7}
    PROMPT_TYPE=stanford #! stanford or BELLE or origin

    #! WANDB
    export WANDB_DISABLED=true
    WANDB_PROJECT_NAME=instruction-tuning-llm
    ENTITY=hengyuan
    WANDB_RUN_NAME=official-${MODEL_NAME}-prompt-type_${PROMPT_TYPE}-train-batch_${TRAIN_BATCH_SIZE}-grad-acc_${GRADIENT_ACCUMULATE}-lr_${LR}-epochs_${EPOCHS}
    # export WANDB_PROJECT=${WANDB_PROJECT_NAME}
    # export WANDB_ENTITY=${ENTITY}





    MODE=${8}
    IS_EXTRA_LARGE=${9}
    LORA_TRAIN=${10}
    
    #!OUTPUT
    if [ "${LORA_TRAIN}" = 'True' ]; then
        TRAINING_TYPE=lora
    else
        TRAINING_TYPE=full
    fi
    OUTPUT_MODEL_DIR=${OUTPUT_BASE_DIR}/${TRAINING_TYPE}/${MODEL_NAME}/${DATASET_TYPE}/${WANDB_RUN_NAME}


    mkdir -p "${OUTPUT_MODEL_DIR}"

    #! multi-node & multi-gpu
    # deepspeed --num_gpus=8 --master_addr=${CHIEF_IP} --master_port=50002 src/official_train.py 

    #! single-node & multi-gpu
    # deepspeed --num_gpus=8 
    #! zero1 + fp16 for 7b, 
    #! 13b need to add "is_extra_large_model"
    #! replace zero config file
    #! lr_scheduler 用的是 deepspeed config 里面的
    if [ "${MODE}" = 'train' ]; then
        deepspeed --include=localhost:${DEVICE} src/official_train_v4.py \
            --deepspeed deepspeed_zero1_config.json \
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
            --is_extra_large_model "${IS_EXTRA_LARGE}" \
            2>&1 | tee ${LOG_DIR}/"${WANDB_RUN_NAME}".log
    fi

}


ddp_inference_lora(){
    TEMPERATURE=0.35
    TOP_P=0.85
    TOP_K=40
    NUM_BEAMS=1
    REPETITION_PENALTY=1.2
    MAX_SEQ_LEN=256

    DECODING_MODE=${9} #! greedy or sample
    USE_LORA=${10} #! loading lora weight or not - only use pre-trained model weight
    PROMPT_TYPE=${11} #! stanford, origin, BELLE
    # PROB_DECODE=${7}
    MODEL_TYPE=${1} #! FLAN-T5, LLAMA, BELLE, GPT2, ALPACA, Chinese_ALPACA, ChatGLM, BLOOM
    EVAL_PRETRAIN=${12}

    #! eval pre-trained model or tuning model
    if [ "${EVAL_PRETRAIN}" = 'True' ]; then
        OUTPUT_FILE_BASE_DIR=${PRETRAINED_MODEL}
        TOKENIZER_PATH=${PRETRAINED_MODEL}
    else
        if [ "${USE_LORA}" = 'True' ]; then
            LORA_WEIGHTS_PATH=${OUTPUT_MODEL_DIR}
            OUTPUT_FILE_BASE_DIR=${LORA_WEIGHTS_PATH}
            TOKENIZER_PATH=${PRETRAINED_MODEL}
        else
            #! fine-tune
            OUTPUT_FILE_BASE_DIR=${OUTPUT_MODEL_DIR}
            PRETRAINED_MODEL=${OUTPUT_MODEL_DIR}
            TOKENIZER_PATH=${OUTPUT_MODEL_DIR}
            LORA_WEIGHTS_PATH=None
        fi
    fi

    # #! change test file
    # TEST_FILE_PATH=${BASE_DIR}/datas/en_zh_seed_task/en_instruction_zh_input_seed_tasks.jsonl
    # TEST_FILE_PATH=${BASE_DIR}/datas/alpaca_instruction_data/en_seed_tasks.jsonl
    TEST_FILE_PATH=${BASE_DIR}/${2}
    # TEST_FILE_PATH=${BASE_DIR}/datas/X-CSR_datasets/X-CSQA/
    TEST_FILE_NAME=`echo ${TEST_FILE_PATH} | awk -F '/' '{print $NF}'`

    DATA_TYPE=${3} #! zh_seed_task, self-instruct, en_zh_seed_tasks, X_CSQA, translation

    #! set example type
    # EXAMPLE_TYPE=use_example
    if [ "${6}" = "True" ]; then 
        EXAMPLE_TYPE=use_example
    else
        EXAMPLE_TYPE=no_example
    fi


        
    # DATA_TYPE=en_instruct_zh_input_seed_task 


    #! instruction 的语言
    SRC_LANG=en
    #! response 的语言
    TGT_LANG=zh



    #! set output_file_dir
    # "$str1" == "$prefix"* 
    # "${DATA_TYPE}" = "X_CSQA"
    if [[ "${DATA_TYPE}" == *"X_"* ]]; then
        LANG=$(basename $(dirname ${TEST_FILE_PATH}))
        if [ "${4}" = "True" ]; then
            OUTPUT_FILE_DIR=${OUTPUT_FILE_BASE_DIR}/predictions/${DATA_TYPE}/${LANG}/${SRC_LANG}-${TGT_LANG}/${PROMPT_TYPE}/${EXAMPLE_TYPE}
        else
            OUTPUT_FILE_DIR=${OUTPUT_FILE_BASE_DIR}/predictions/${DATA_TYPE}/${LANG}/no-add-lang/${PROMPT_TYPE}/${EXAMPLE_TYPE}
        fi
        
    else 
        if [ "${4}" = "True" ]; then
            OUTPUT_FILE_DIR=${OUTPUT_FILE_BASE_DIR}/predictions/${DATA_TYPE}/${SRC_LANG}-${TGT_LANG}/${PROMPT_TYPE}/${EXAMPLE_TYPE}
        else
            OUTPUT_FILE_DIR=${OUTPUT_FILE_BASE_DIR}/predictions/${DATA_TYPE}/no-add-lang/${PROMPT_TYPE}/${EXAMPLE_TYPE}
        fi
    fi
    
    

    mkdir -p ${OUTPUT_FILE_DIR}

    OUTPUT_FILE_PATH=${OUTPUT_FILE_DIR}/${MODEL_TYPE}-for-${TEST_FILE_NAME}_prediction.jsonl
    LOG_PATH=${BASE_DIR}/projects/LLM_inference/logs/inference/lora/${DATA_TYPE}/${MODEL_NAME}/${PROMPT_TYPE}/${EXAMPLE_TYPE}


    mkdir -p ${LOG_PATH}

    MODEL_NAME=`echo ${PRETRAINED_MODEL} | awk -F '/' '{print $NF}'`

    RUN_NAME=${MODEL_NAME}-${DATA_TYPE}-${PROMPT_TYPE}-temperature_${TEMPERATURE}-topk_${TOP_K}-topp_${TOP_P}-repetition_${REPETITION_PENALTY}-beams_${NUM_BEAMS}.log



    export CUDA_VISIBLE_DEVICES=${DEVICE}
    #! ddp batch size
    BATCH_SIZE=8
    #! if use deepspeed, use deepspeed to launchd
    # deepspeed --num_gpus=1
    # python -u
    #! deepspeed --include=localhost:GPU_RANK
    deepspeed --include=localhost:${DEVICE} src/ddp_batch_inference_v2.py \
        --batch_size ${BATCH_SIZE} \
        --test_file_path ${TEST_FILE_PATH} \
        --output_file_path ${OUTPUT_FILE_PATH} \
        --pretrained_model_path ${PRETRAINED_MODEL} \
        --lora_weights_path ${LORA_WEIGHTS_PATH} \
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
        --add_lang "${4}" \
        --add_self_understand False \
        --is_extra_large_model "${5}" \
        --use_demonstration "${6}" \
        --decoding_mode ${DECODING_MODE} \
        --use_probing_strategy "${7}" \
        --probing_text "${8}" \
        2>&1 | tee ${LOG_PATH}/${RUN_NAME} 
}


llm_harness_eval(){

    USE_LORA=${1} #! loading lora weight or not - only use pre-trained model weight
    USE_EXP=${2}
    EVAL_DATA_TYPES=${3}    #! piqa,arc_easy,arc_challenge,rte,cb,wsc273,winogrande 
    #! batch size
    BATCH_SIZE=${4}
    EVAL_PRETRAIN=${5}
    
    #! eval pre-trained model or tuning model
    if [ "${EVAL_PRETRAIN}" = 'True' ]; then
        OUTPUT_FILE_BASE_DIR=${PRETRAINED_MODEL}
        TASK_DESCRIPTION_FILE=None
        MODEL_ARGS="pretrained=${PRETRAINED_MODEL}"
    else
        TASK_DESCRIPTION_FILE=${LLM_EVAL_PROJECT}/task_description.json
        if [ "${USE_LORA}" = 'True' ]; then
            LORA_WEIGHTS_PATH=${OUTPUT_MODEL_DIR}
            OUTPUT_FILE_BASE_DIR=${LORA_WEIGHTS_PATH}
            MODEL_ARGS="pretrained=${PRETRAINED_MODEL},peft=${LORA_WEIGHTS_PATH}"
        else
            OUTPUT_FILE_BASE_DIR=${OUTPUT_MODEL_DIR}
            MODEL_ARGS="pretrained=${OUTPUT_MODEL_DIR}"
        fi
    fi

    #! model args

    #! setting OUTPUT_FILE_BASE_DIR
    #! eval pretrain model or lora model
    


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
# PRETRAINED_MODEL=${BASE_DIR}/pretrained_models/LLAMA/hf/llama-7b
declare -a MODEL_NAMES
#! 原本的 LLAMA 就默认代码 7b 的
# MODEL_NAMES=('Chinese-LLAMA')
# MODEL_NAMES=('GPT2-base' 'GPT2-large')
# MODEL_NAMES=('GPT-Neo-125M')
# MODEL_NAMES=('OPT-1.3B' 'GPT2-large')
# MODEL_NAMES=('OPT-125M' 'OPT-350M' 'GPT2-large')
# MODEL_NAMES=('OPT-1.3B')
# MODEL_NAMES=('OPT-2.7B')
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
#  PRETRAINED_MODELS=('pretrained_models/opt/opt-1.3B' 'pretrained_models/gpt2/gpt2-large')
PRETRAINED_MODELS=('pretrained_models/LLAMA/hf/llama-7b')
# PRETRAINED_MODELS=('pretrained_models/gpt-neo/gpt2-neo-125M')
# PRETRAINED_MODELS=('pretrained_models/opt/opt-2.7B')
# PRETRAINED_MODELS=('pretrained_models/opt/opt-1.3B')
# PRETRAINED_MODELS=('pretrained_models/opt/opt-125M' 'pretrained_models/opt/opt-350M' 'pretrained_models/gpt2/gpt2-large')
# PRETRAINED_MODELS=('pretrained_models/gpt2/gpt2-base' 'pretrained_models/gpt2/gpt2-large')
# PRETRAINED_MODELS=('pretrained_models/chinese_llama/hf/chinese-llama-7b')
# PRETRAINED_MODELS=('pretrained_models/LLAMA/hf/llama-13b')
# PRETRAINED_MODELS=('pretrained_models/extended_zh_vocab_llama')
# PRETRAINED_MODELS=('pretrained_models/LLAMA+translation/official-llama-7b-post_training_translation_instruction_100000.json-prompt-type_origin-train-batch_2-lr_1e-4-epochs_5')
# PRETRAINED_MODELS=('pretrained_models/extended_zh_vocab_llama+translation/official-extended_zh_vocab_llama-post_training_translation_concat_100000.json-prompt-type_origin-train-batch_2-lr_1e-4-epochs_5')
# PRETRAINED_MODELS=('pretrained_models/LLAMA/hf/llama-7b' 'pretrained_models/chinese_llama/hf/chinese-llama-7b')
# PRETRAINED_MODELS=('pretrained_models/gpt2/gpt2-base' 'pretrained_models/gpt2/gpt2-base')
# PRETRAINED_MODELS=('pretrained_models/bloom/bloom-7b-mt' 'pretrained_models/LLAMA/hf/llama-7b')

declare -a TRAIN_DATASETS
#! change dataset file path here!!
# DATASETS=('test_pipeline_data.json' 'test_pipeline_500.json')
# DATASETS=('test_pipeline_500.json')

# TRAIN_DATASETS=('alpaca_data+zh_alpaca_data_revised.json')
# DATASETS=('alpaca_data+en2zh+zh2en-not-same-1000.json' 'alpaca_data+en2zh+zh2en-not-same-10000.json' 'alpaca_data+en2zh+zh2en-not-same-5000.json' 'alpaca_data+wmt-en2zh-1000.json' 'alpaca_data+wmt-en2zh-10000.json' 'alpaca_data+wmt-en2zh-5000.json')
# DATASETS=('alpaca_data.json' 'alpaca_data+en2zh+zh2en-not-same-1000.json' 'alpaca_data+en2zh+zh2en-not-same-10000.json' 'alpaca_data+en2zh+zh2en-not-same-5000.json' 'alpaca_data+wmt-en2zh-1000.json' 'alpaca_data+wmt-en2zh-10000.json' 'alpaca_data+wmt-en2zh-5000.json')
# DATASETS=('alpaca_data+en2zh+zh2en-not-same-50000.json' 'alpaca_data+en2zh+zh2en-not-same-100000.json' 'alpaca_data+en2zh+zh2en-not-same-25000.json' 'alpaca_data+en2zh-25000.json' 'alpaca_data+en2zh-50000.json' 'alpaca_data+en2zh-100000.json')
# DATASETS=('alpaca_data+en2zh-100000.json')
# DATASETS=('alpaca_data.json' 'alpaca_data+en2zh-50.json' 'alpaca_data+en2zh-100.json' 'alpaca_data+en2zh-200.json' 'alpaca_data+en2zh-300.json' 'alpaca_data+en2zh-400.json' 'alpaca_data+en2zh-500.json' 'alpaca_data+en2zh+zh2en-not-same-50.json' 'alpaca_data+en2zh+zh2en-not-same-100.json' 'alpaca_data+en2zh+zh2en-not-same-200.json' 'alpaca_data+en2zh+zh2en-not-same-300.json' 'alpaca_data+en2zh+zh2en-not-same-400.json' 'alpaca_data+en2zh+zh2en-not-same-500.json' 'alpaca_data+en2zh-25000.json' 'alpaca_data+en2zh-50000.json' 'alpaca_data+en2zh-100000.json' 'alpaca_data+en2zh+zh2en-not-same-25000.json' 'alpaca_data+en2zh+zh2en-not-same-50000.json' 'alpaca_data+en2zh+zh2en-not-same-10000.json')
# TRAIN_DATASETS=('alpaca_instruction_data/alpaca_data.json')
# TRAIN_DATASETS=('dolly-human/dolly-15k.json')
# TRAIN_DATASETS=('test_pipeline_data.json')
# TRAIN_DATASET_TYPES=('test')
# TRAIN_DATASET_TYPES=('alpaca_data+translation')
# TRAIN_DATASET_TYPES=('alpaca_data')
# TRAIN_DATASET_TYPES=('dolly_human-subset-1000')
# TRAIN_DATASET_TYPES=('alpaca_data+zh_alpaca_data')
# LRS=('6e-4' '3e-4' '2e-4')
# LRS=('6e-4' '6e-4')
# LRS=('6e-4' '1e-3')
# BATCH_SIZES=('16' '16' '8')
# BATCH_SIZES=('4' '8')
# BATCH_SIZES=('2' '2')
# GRAD_ACC=('1' '1' '2')
# GRAD_ACC=('4' '2')
# GRAD_ACC=('4' '4') #! update 的数量增加一倍
#TODO: 不同的模型用 lora 和不用lora 
# USE_LORAS=('False' 'False' 'False') #! True or False, True for lora training, false for full tuning
# USE_LORAS=('True' 'True') #! True or False, True for lora training, false for full tuning
# USE_LORAS=('True') #! True or False, True for lora training, false for full tuning
# EPOCHS=('7' '7')
# EPOCHS=('1' '1')

#! subset training
SUBSET_NUM=1000
SUBSET_SIZE=1000

for ((i=701; i<=1000; i++)); do
    TRAIN_DATASETS+=("dolly-human-subset/${SUBSET_NUM}-${SUBSET_SIZE}/subset_data/dolly-human-size-1000-subset_data-${i}.json")
    TRAIN_DATASET_TYPES+=("dolly_human-subset-${SUBSET_NUM}-${SUBSET_SIZE}/dolly_human-subset-${i}")
    LRS+=("1e-4")
    BATCH_SIZES+=("2")
    GRAD_ACC+=("2")
    USE_LORAS+=("True")
    EPOCHS+=("4")
done

train_eval_llm(){
    #! llm_eval 的话，很大的模型估计就跑不了了，所以就一直为 False 
    EXTRA_LARGE=False
    USE_EXPS=('False' 'True')
    EVAL_PRETRAINS=('True')
    MODE=eval
    EVAL_DATA_TYPES=rte,cb #! rte,cb,arc_easy,piqa,arc_challenge
    for ((i=0;i<${#MODEL_NAMES[@]};i++))
    do 
        for j in ${!TRAIN_DATASETS[@]}
        do
            train_lora "${PRETRAINED_MODELS[${i}]}" "${MODEL_NAMES[${i}]}" "${TRAIN_DATASETS[${j}]}" "${TRAIN_DATASET_TYPES[${j}]}" "${BATCH_SIZES[${i}]}" "${LRS[${i}]}" "${GRAD_ACC[${i}]}" "${MODE}" "${EXTRA_LARGE}" "${USE_LORAS[${i}]}" "${EPOCHS[${i}]}" #! modify mode
            
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




eval_generation(){
    declare -a EVAL_DATA_TYPES=('self-instruct')
    DECODING_MODE=sample
    EXTRA_LARGE=False
    #! 必须是 False
    USE_EXPS=('False')
    #! change mode
    MODE=${1}
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
                        # TEST_FILE=datas/X-CSR_datasets/${EVAL_DATA_TYPE}/${LANG}/dev_v2.jsonl #! 有的是 dev.jsonl, 有的是 dev_v2.jsonl
                            
                        # ddp_inference_lora "${MODEL_NAMES[${i}]}+${TRAIN_DATASET_TYPES[${j}]}" "${TEST_FILE}" "${EVAL_DATA_TYPE}" "${ADD_LANG}" "${EXTRA_LARGE}" "${USE_EXP}" "${PROB_DECODE}" "${PROB_TEXT}" "${DECODING_MODE}" "${USE_LORAS[${i}]}" "${PROMPT_TYPE}" "${EVAL_PRETRAIN}"
                
                    done
                done
            done
        done
    done
}


test_pipeline(){
    declare -a MODEL_NAMES
    #! 原本的 LLAMA 就默认代码 7b 的

    MODEL_NAMES=('GPT2-base')

    declare -a PRETRAINED_MODELS
    PRETRAINED_MODELS=('pretrained_models/gpt2/gpt2-base')
    
    declare -a TRAIN_DATASETS
    #! change dataset file path here!!
    TRAIN_DATASETS=('test_pipeline_data.json')
    TRAIN_DATASET_TYPES=('test')

    # declare -a EVAL_DATA_TYPES=('X_CSQA' 'X_CSQA_ARC_Easy' 'X_CSQA_ARC_Challenge' 'X_CSQA_PIQA') #'X_CSQA''X_CSQA_ARC_Easy' 'X_CSQA_ARC_Challenge'
    declare -a EVAL_DATA_TYPES=('X_CSQA_PIQA') #'X_CSQA''X_CSQA_ARC_Easy' 'X_CSQA_ARC_Challenge' 
    # declare -a TEST_LANGS=('en' 'zh' 'ar' 'de' 'es' 'fr' 'hi' 'it' 'jap' 'nl' 'pl' 'pt' 'ru' 'sw' 'ur' 'vi')
    # declare -a TEST_LANGS=('en' 'zh')
    declare -a TEST_LANGS=('en')
    # declare -a TEST_LANGS=('en' 'zh' 'ar' 'de' 'es' 'fr' 'hi' 'it' 'jap' 'nl' 'pl' 'pt' 'ru' 'sw' 'ur' 'vi')

    EXTRA_LARGE=False
    USE_EXPS=('True')
    MODE=eval
    PROB_DECODE=True
    PROB_TEXT=True
    USE_LORA=False #! True or False
    PROMPT_TYPE=origin #! stanford or origin
    for ((i=0;i<${#MODEL_NAMES[@]};i++))
    do 
        for j in ${!TRAIN_DATASETS[@]}
        do
            train_lora "${PRETRAINED_MODELS[${i}]}" "${MODEL_NAMES[${i}]}" "${TRAIN_DATASETS[${j}]}" "${TRAIN_DATASET_TYPES[${j}]}" "${MODE}" "${EXTRA_LARGE}" #! modify mode
            for EVAL_DATA_TYPE in ${EVAL_DATA_TYPES[@]}
            do
                for ADD_LANG in False
                do 
                    for USE_EXP in ${USE_EXPS[@]}
                    do
                        for LANG in ${TEST_LANGS[@]}
                        do
                            TEST_FILE=datas/X-CSR_datasets/${EVAL_DATA_TYPE}/${LANG}/dev_v2.jsonl #! 有的是 dev.jsonl, 有的是 dev_v2.jsonl
                            
                            ddp_inference_lora "${MODEL_NAMES[${i}]}+${TRAIN_DATASETS[${j}]}" "${TEST_FILE}" "${EVAL_DATA_TYPE}" "${ADD_LANG}" "${EXTRA_LARGE}" "${USE_EXP}" "${PROB_DECODE}" "${PROB_TEXT}" "${DECODING_MODE}"
                        done

                        #! get_metric after all lang inferencing
                        echo 'getting metrics ...'
                        RESULT_DIR=${OUTPUT_FILE_BASE_DIR}/predictions/${EVAL_DATA_TYPE}
                        METRIC_DIR=${OUTPUT_FILE_BASE_DIR}/metrics/${EVAL_DATA_TYPE}
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

# eval_x_csqa
eval_generation train
# train_eval_llm
# test_pipeline

# eval_x_csqa
# eval_x_nli

# eval_x_csqa_different_size
# eval_x_nli_different_size
# eval_translation
# eval_translation_different_size
# eval_x_csqa
# train_and_inference_different_size
# train_and_inference_different_dataset

