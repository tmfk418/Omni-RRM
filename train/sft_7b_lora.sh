# ================================
# Environment Configuration
# ================================
export MAX_PIXELS=131072               # Maximum number of pixels allowed for image input
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7      # GPUs to use (multi-GPU training)
export FPS=1                             # Frames per second for video extraction
export FPS_MAX_FRAMES=20                 # Maximum number of frames to load per video
export VIDEO_MAX_PIXELS=12288            # Maximum pixel budget for video frames
export AUDIO_MEMMAP=true                 # Enable memory-mapped audio loading for efficiency
export AUDIO_CHUNK_SIZE=4000             # Size of audio chunks in samples
export AUDIO_NUM_WORKERS=4               # Number of parallel workers for audio processing
export VIDEO_READER_BACKEND=torchvision  # Video decoding backend (torchvision is stable)
export TOKENIZERS_PARALLELISM=false      # Disable tokenizer multithreading to reduce overhead
export OMP_NUM_THREADS=1                 # Limit CPU threads (avoid oversubscription)
export SWIFT_DEVICE_MAP=auto             # Automatically map model across available GPUs

# ================================
# Training Command (Swift SFT)
# ================================
torchrun --nproc_per_node=8 /path/to/swift/cli/sft.py \
  --do_train \
  --model /path/to/modelscope/Qwen/Qwen2.5-Omni-7B \   # Base Qwen2.5-Omni-7B model
  --dataset /path/to/dataset/audio/final_sft_data.jsonl \  # Audio SFT dataset
            /path/to/dataset/video/final_sft_data.jsonl \  # Video SFT dataset
            /path/to/dataset/image/final_sft_data.jsonl \  # Image SFT dataset
  --lora_rank 8 \                           # LoRA rank (smaller → lower memory usage)
  --lora_alpha 32 \                         # LoRA scaling factor
  --freeze_vit true \                       # Freeze Vision Transformer (saves memory)
  --target_modules all-linear \             # Apply LoRA to all linear layers
  --output_dir /path/to/outputs \           # Directory to save checkpoints
  --num_train_epochs 2 \                    # Number of training epochs
  --per_device_train_batch_size 1 \         # Training batch size per GPU
  --per_device_eval_batch_size 1 \          # Evaluation batch size per GPU
  --gradient_accumulation_steps 12 \        # Accumulate gradients to simulate larger batch
  --learning_rate 5e-6 \                    # Learning rate
  --warmup_ratio 0.03 \                     # Warmup ratio for LR scheduler
  --max_length 3072 \                       # Maximum sequence length
  --eval_steps 100 \                        # Evaluate every 100 steps
  --save_steps 100 \                        # Save checkpoint every 100 steps
  --save_total_limit 4 \                    # Keep only the latest 4 checkpoints
  --logging_steps 5 \                       # Log every 5 steps
  --dataloader_num_workers 2 \              # Number of dataloader workers
  --torch_dtype bfloat16 \                  # Use BF16 for better performance on modern GPUs
  --deepspeed zero2 \                       # Enable DeepSpeed ZeRO stage 2 for memory optimization
  2>&1 | tee /path/to/logs/log_file_3b_lora.txt   # Save logs to file
