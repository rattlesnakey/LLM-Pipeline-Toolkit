# CHECKPOINT_PATH=/apdcephfs/share_916081/effidit_shared_data/chillzhang/projects/LLM_inference/checkpoints/flan-t5-xl-train-batch_4-gradient-acc_4-lr-1e-4-epochs_5-beams_5
# CHECKPOINT_PATH=/workspace/share_916081/effidit_shared_data/chillzhang/projects/LLM_inference/checkpoints/official-llama-7b-train-batch_2-lr_2e-5-epochs_3
# CHECKPOINT_PATH=/apdcephfs/share_916081/effidit_shared_data/chillzhang/projects/LLM_inference/checkpoints/llama-7b-train-batch_4-lr_2e-5-epochs_3
CHECKPOINT_PATH=/apdcephfs/share_916081/victoriabi/para_data/train/checkpoints/official-bloom-7b-mt-500w_out.json-prompt-type_stanford-train-batch_2-lr_2e-5-epochs_2/checkpoint-14000
OUTPUT_FILE=${CHECKPOINT_PATH}/pytorch_model.bin
python src/zero_to_fp32.py --checkpoint_dir ${CHECKPOINT_PATH} --output_file ${OUTPUT_FILE}
