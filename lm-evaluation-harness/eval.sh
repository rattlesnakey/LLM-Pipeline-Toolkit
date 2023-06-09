
export HF_DATASETS_CACHE=./huggingface_datasets_cache_dir

BASE_DIR=/apdcephfs/share_916081/effidit_shared_data/chillzhang
# PRETRAINED_MODEL_PATH=${BASE_DIR}/pretrained_models/LLAMA/hf/llama-7b
# PRETRAINED_MODEL_PATH=${BASE_DIR}/pretrained_models/gpt-neo/gpt2-neo-125M
PRETRAINED_MODEL_PATH=${BASE_DIR}/pretrained_models/gpt2/gpt2-base
# PEFT_MODEL_PATH=${BASE_DIR}/projects/LLM_inference/checkpoints/lora/LLAMA/alpaca_data/official-LLAMA+alpaca_data.json-prompt-type_stanford-train-batch_2-lr_1e-4-epochs_4
PEFT_MODEL_PATH=${BASE_DIR}/projects/LLM_inference/checkpoints/lora/GPT2-base/alpaca_data/official-GPT2-base+alpaca_data.json-prompt-type_stanford-train-batch_2-lr_1e-4-epochs_4
BATCH_SIZE=4

# piqa,arc_easy,arc_challenge,rte,cb,wsc273,winogrande
python main.py \
    --model hf-causal-experimental \
    --model_args pretrained=${PRETRAINED_MODEL_PATH} \
    --tasks rte,cb \
    --no_cache \
    --batch_size ${BATCH_SIZE} \
    --output_path ./metric.json \
    --num_fewshot 5 \
    --device cuda:0

# python main.py \
#     --model hf-causal-experimental \
#     --model_args pretrained=${PRETRAINED_MODEL_PATH},peft=${PEFT_MODEL_PATH} \
#     --tasks piqa \
#     --no_cache \
#     --batch_size ${BATCH_SIZE} \
#     --device cuda:0