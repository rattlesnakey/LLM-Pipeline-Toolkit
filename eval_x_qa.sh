BASE_DIR=/apdcephfs/share_916081/effidit_shared_data/chillzhang

OUTPUT_MODEL_DIR=${BASE_DIR}/projects/LLM_inference/checkpoints/lora/LLAMA/alpaca_data/official-LLAMA+alpaca_data.json-prompt-type_stanford-train-batch_2-lr_1e-4-epochs_4

get_XQA_metric(){
    DATA_TYPES=('X_CSQA_PIQA' 'X_CSQA_ARC_Easy' 'X_CSQA_ARC_Challenge' 'X_CSQA' 'X_NLI')
    for DATA_TYPE in ${DATA_TYPES[@]}
    do
        RESULT_DIR=${OUTPUT_MODEL_DIR}/predictions/${DATA_TYPE}
        METRIC_DIR=${OUTPUT_MODEL_DIR}/metrics/${DATA_TYPE}

        mkdir -p ${METRIC_DIR}
        python -u src/eval_XQA.py \
            --result_dir ${RESULT_DIR} \
            --metric_dir ${METRIC_DIR} 
    done
    

}

get_XQA_metric
