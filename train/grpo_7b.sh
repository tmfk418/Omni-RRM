#!/usr/bin/env bash

# ----------------------- Environment Variables -----------------------
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # Use 8 GPUs
export NPROC_PER_NODE=8                       # Number of processes per node
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # Reduce CUDA OOM risk

# Multimodal input parameters to reduce GPU memory usage
export MAX_PIXELS=131072          # Max image resolution (width * height)
export VIDEO_MAX_PIXELS=12288     # Max resolution per video frame
export FPS=1                      # Frames per second for video sampling
export FPS_MAX_FRAMES=8           # Max frames per video (reduces VRAM load)
export AUDIO_MEMMAP=true          # Use memory-mapped audio loading
export AUDIO_CHUNK_SIZE=4000      # Audio chunk size (samples)
export AUDIO_NUM_WORKERS=4        # Parallel workers for audio preprocessing

# ----------------------- GRPO Training -----------------------
swift rlhf \
  --rlhf_type grpo \
  --model /path/to/models/Qwen/Qwen2.5-Omni-7B \   # Base Omni-7B model
  --resume_from_checkpoint /path/to/checkpoint-200 \  # Resume training from checkpoint
  --reward_funcs mm_format mm_content mm_rubric \   # Reward functions
  --reward_weights 0.4 0.4 0.2 \                    # Weights for each reward
  --train_type lora \                               # Use LoRA tuning
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \                     # Apply LoRA to all linear layers
  --torch_dtype bfloat16 \                          # Use BF16 precision
  --gradient_checkpointing \                        # Enable gradient checkpointing
  --dataset \                                       # Training datasets
      /path/to/dataset/audio/final_rl_data.jsonl \
      /path/to/dataset/video/final_rl_data.jsonl \
      /path/to/dataset/image/final_rl_data.jsonl \
  --external_plugins /path/to/plugin/new_plugin.py \ # External plugin for custom logic
  --max_completion_length 1024 \                    # Max generated output length
  --max_length 6144 \                               # Max input sequence length
  --num_generations 2 \                             # Number of candidate generations
  --num_train_epochs 1 \                            # Training epochs
  --per_device_train_batch_size 1 \                 # Train batch size per GPU
  --per_device_eval_batch_size 1 \                  # Eval batch size per GPU
  --gradient_accumulation_steps 2 \                 # Gradient accumulation steps
  --learning_rate 1e-5 \                            # Learning rate
  --warmup_ratio 0.05 \                             # Warmup ratio for LR scheduler
  --eval_steps 100 \                                # Evaluate every 100 steps
  --save_steps 100 \                                # Save checkpoint every 100 steps
  --save_total_limit 4 \                            # Keep only last 4 checkpoints
  --logging_steps 5 \                               # Log every 5 steps
  --dataloader_num_workers 4 \                      # Number of DataLoader workers
  --dataset_num_proc 4 \                            # Number of dataset preprocessing workers
  --temperature 1.0 \                               # Sampling temperature
  --top_p 0.95 \                                    # Nucleus sampling (top-p)
  --top_k 50 \                                      # Top-k sampling
  --lazy_tokenize false \                           # Disable lazy tokenization
  --system /path/to/prompt.txt \                    # System prompt file
  --deepspeed zero2 \                               # Use DeepSpeed ZeRO stage 2
  --log_completions true \                          # Log model completions
  2>&1 | tee /path/to/logs/log_grpo_7b.txt          # Save logs
