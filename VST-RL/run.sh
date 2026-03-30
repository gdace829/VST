#!/bin/bash
set -x

NNODES=4
NGPUS_PER_NODE=8
PROJ_ROOT=/VST/VST-RL
DATASET_ROOT=/VST/VST-RL/data

MODEL_PATH="base model path"
TRAIN_PATH="${DATASET_ROOT}/training_data.parquet"

EXP=video_streaming_thinking

PROJ_DIR=${PROJ_ROOT}/${EXP}
BASE_LOG_DIR="${PROJ_ROOT}/tensorboard_logs"
VIDEO_MEM_PATH="${PROJ_ROOT}/recurrent/impls/video_memory.py"

VIDEO_LENGTH=5000
MAX_VIDEO_FRAME=280
MAXLEN=$((6000 + VIDEO_LENGTH)) 
MAX_NEW_TOKEN=1000
export RAY_SCHEDULER_EVENTS=0

python3 -m verl.trainer.main_ppo \
    recurrent.enable=video_memory \
    recurrent.video_memory.path=$VIDEO_MEM_PATH \
    recurrent.video_memory.config.video_clip_token_size=$VIDEO_LENGTH \
    recurrent.video_memory.config.max_video_frame=$MAX_VIDEO_FRAME \
    recurrent.video_memory.config.prompt_type="type1" \
    data.prog_video_length=False \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.freeze_vision_tower=True \
    actor_rollout_ref.model.enable_embed_cache=False \
    algorithm.grpo_use_adv=False \
    trainer.save_freq=10 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    trainer.logger=['console','tensorboard'] \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.clip_ratio_high=0.20 \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    data.train_files=$TRAIN_PATH \
    data.shuffle=False \
    data.filter_overlong_prompts=True \
    data.train_batch_size=256 \
    data.truncation='center' \
    +data.context_key='context' \
    data.max_prompt_length=$MAXLEN \
    data.max_response_length=$MAX_NEW_TOKEN \
    reward_model.reward_manager='thread' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=0.98 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.project_name='video_streaming_thinking' \
    trainer.experiment_name=${EXP} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$NGPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$PROJ_DIR \
    trainer.total_epochs=1