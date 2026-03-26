TASK_JSON="eval_data/anno/eval/StreamingBench/json/real_time_visual_understanding.json"
VIDEO_DIR="/path/to/StreamingBench/Real-Time Visual Understanding"

RUN_NAME=video_streaming_thinking

cd ../

python lmms-eval/lmms_eval/models/simple/qwen2_5_vl_sf.py \
    --run_name $RUN_NAME \
    --ckpt_path "$MODEL_PATH" \
    --task_json "$TASK_JSON" \
    --video_dir "$VIDEO_DIR" \
    --world_size 1
