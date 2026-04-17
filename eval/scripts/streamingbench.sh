TASK_JSON="eval_data/anno/eval/StreamingBench/json/real_time_visual_understanding.json"
VIDEO_DIR="${VIDEO_DIR:-/data/StreamingBench/data}"
ROLLOUT_LIKE="${ROLLOUT_LIKE:-1}"
STREAM_THINK_TIMES="${STREAM_THINK_TIMES:-4}"
MAX_STREAM_VID_TOKENS="${MAX_STREAM_VID_TOKENS:-8192}"

RUN_NAME=video_streaming_thinking

cd ../

CMD=(python lmms-eval/lmms_eval/models/simple/qwen2_5_vl_sf.py \
    --run_name $RUN_NAME \
    --ckpt_path "$MODEL_PATH" \
    --task_json "$TASK_JSON" \
    --video_dir "$VIDEO_DIR" \
    --world_size 1 \
    --fps 1 \
    --max_num_frames 256 \
    --max_pixels 401408 \
    --time_window_size 256)

if [ "$ROLLOUT_LIKE" = "1" ]; then
    CMD+=(
        --rollout_like
        --stream_think_times "$STREAM_THINK_TIMES"
        --max_stream_vid_tokens "$MAX_STREAM_VID_TOKENS"
    )
fi

"${CMD[@]}"
