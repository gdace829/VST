export HF_HOME="../eval_data"
export HF_DATASETS_CACHE="../eval_data"

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
    --model qwen2_5_vl_stream_think \
    --model_args=pretrained="${MODEL_PATH}",attn_implementation=flash_attention_2,stream_think_times=2-3-5-5,max_stream_vid_tokens=8192,max_num_frames=384 \
    --tasks ovobench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix vst \
    --output_path ./logs/