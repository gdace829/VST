export DECORD_EOF_RETRY_MAX=40960

export VIDEO_MIN_PIXELS=78400
export FPS_MAX_FRAMES=384

export VIDEO_MAX_PIXELS=19267584

text_sink=512
TEXT_SLIDING_WINDOW=32768

export DATASET_PATH=""
export EVAL_DATASET_PATH=""

# Key parameters
epoch_num=1
gradient_accumulation_steps=8
learning_rate=5e-6
model_name="Qwen2.5-VL-7B-Instruct"

timestamp=$(date +%Y%m%d_%H%M%S)

export RUN_NAME="${WANDB_PROJECT_NAME}_e${epoch_num}_lr${learning_rate}_ps${text_sink}_pw${TEXT_SLIDING_WINDOW}"
export OUTPUT_DIR="./checkpoints/${RUN_NAME}_${timestamp}" # wo a / at the end

export DATASET_PATH=/workspace/images-ks3-hd/dataset/
export EVAL_DATASET_PATH=/workspace/images-ks3-hd/workspace/yinliang/datasets/Inf-Stream-Eval


TRAIN_DATASET_NAMES=(
    "train_with_seeks.jsonl"
)

LABEL_PATH=""

TRAIN_FILES=("${TRAIN_DATASET_NAMES[@]/#/$LABEL_PATH/}")
VALID_FILES=("${VALID_DATASET_NAMES[@]/#/$LABEL_PATH/}")

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 


NNODES=2                      
NODE_RANK=0                    
GPUS_PER_NODE=8                

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# === 启动命令 ===
torchrun --nproc_per_node=$RESOURCE_GPU \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --deepspeed ./scripts/zero3.json \
    --overwrite_output_dir True \
    --output_dir "${OUTPUT_DIR}/${RUN_NAME}_${timestamp}" \
    --run_name $RUN_NAME \
    --save_on_each_node True \
    --do_train True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate $learning_rate \
    --warmup_ratio 0.03 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --num_train_epochs $epoch_num \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --pretrained_model_name_or_path $model_name \
    --train_annotation_paths "${TRAIN_FILES[@]}" \
    --dataloader_num_workers 8 \
    --use_liger_kernel True \
    --report_to tensorboard \
    --ignore_data_skip False \
    --save_strategy steps \
    --save_steps 25 \
    --save_total_limit 100 \
    --load_best_model_at_end False \
    --greater_is_better False \
    --prediction_loss_only true \
    --eval_steps 50 \
    --metric_for_best_model eval_loss \
    --eval_strategy steps \
    --per_device_eval_batch_size 1 \
    --eval_annotation_paths "${VALID_FILES[@]}" \
    --text_sink $text_sink \
    --text_sliding_window $TEXT_SLIDING_WINDOW