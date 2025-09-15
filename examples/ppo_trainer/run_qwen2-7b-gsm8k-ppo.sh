#!/bin/bash

set -x

ulimit -Sn 20480

source /env/bin/start-ctx-user
conda activate tejasverl

unset ROCR_VISIBLE_DEVICES

export WANDB_ENTITY=contextual 

# Auto-detect number of GPUs per node from SLURM --gres
function get_gpus_per_node() {
    # Check if we're running under SLURM
    if [[ -n "$SLURM_JOB_ID" ]]; then
        # Get GRES information from SLURM (look for gres/gpu=X in ReqTRES or AllocTRES)
        # Use head -1 to get only the first match, and tr -d '\n' to remove any newlines
        GRES_INFO=$(scontrol show job $SLURM_JOB_ID | grep -o 'gres/gpu=[0-9]*' | head -1 | cut -d'=' -f2 | tr -d '\n')
        
        if [[ -n "$GRES_INFO" ]]; then
            GPU_COUNT=$GRES_INFO
        else
            echo "Warning: No gres/gpu information found in SLURM job"
            GPU_COUNT=1
        fi
    else
        # Fallback: detect GPUs from nvidia-smi if available
        if command -v nvidia-smi &> /dev/null; then
            GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        else
            echo "Warning: Not running under SLURM and nvidia-smi not available, defaulting to 1 GPU"
            GPU_COUNT=1
        fi
    fi
    
    echo "$GPU_COUNT"
}

# Auto-detect GPUs per node
N_GPUS_PER_NODE=$(get_gpus_per_node)
echo "Auto-detected GPUs per node: $N_GPUS_PER_NODE"

# Calculate total GPUs if running multi-node
if [[ -n "$SLURM_NNODES" ]]; then
    TOTAL_GPUS=$((N_GPUS_PER_NODE * SLURM_NNODES))
    echo "Total GPUs across $SLURM_NNODES nodes: $TOTAL_GPUS"
fi

GSM8K_DATA_PATH=/data/env/lib/repos/tejas_srinivasan/gsm8k_trial
gsm8k_train_path=$GSM8K_DATA_PATH/train.parquet
gsm8k_test_path=$GSM8K_DATA_PATH/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

export MODEL="Qwen/Qwen2-7B-Instruct"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=$gsm8k_train_path \
 data.val_files=$gsm8k_test_path \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 actor_rollout_ref.model.path=$MODEL \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=$MODEL \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger='["console","wandb"]' \
 trainer.val_before_train=False \
 trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
 trainer.nnodes=${SLURM_NNODES:-1} \
 trainer.project_name='verl_example' \
 trainer.experiment_name="qwen2-7b-gsm8k_ppo-${N_GPUS_PER_NODE}gpu" \
 trainer.save_freq=10 \
 trainer.max_actor_ckpt_to_keep=3 \
 trainer.max_critic_ckpt_to_keep=3 \
 trainer.test_freq=10 \
 trainer.total_epochs=15 2>&1 | tee verl_demo.log