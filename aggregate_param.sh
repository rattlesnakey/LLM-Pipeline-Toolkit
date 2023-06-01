BASE_DIR=/apdcephfs/share_916081/effidit_shared_data/chillzhang

embedding_ckpt_dir=${BASE_DIR}/pretrained_models/extended_zh_vocab_llama+translation/official-extended_zh_vocab_llama-post_training_translation_instruction_100000.json-prompt-type_origin-train-batch_2-lr_1e-4-epochs_5
# backbone_dir=${BASE_DIR}/pretrained_models/LLAMA/hf/llama-7b
backbone_dir=${BASE_DIR}/pretrained_models/extended_zh_vocab_llama


python -u src/aggregate_embedding.py --embedding_ckpt_dir ${embedding_ckpt_dir} --backbone_dir ${backbone_dir}


