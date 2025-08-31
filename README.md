# Installation

```bash
# 1) Create the environment from the project spec
conda env create -n sgp_gen -f environment.yml

# 2) Activate the environment
conda activate sgp_gen

# 3) Install Python dependencies
pip install -r requirements.txt
```

# Running RL Experiments

```bash
python train_zero_svg.py \
    --critic_type grpo \
    --gpus 8 \
    --seed 32 \
    --enable_prefix_caching \
    --collocate \
    --vllm_sleep \
    --vllm_gpu_ratio 0.35 \
    --gradient-checkpointing \
    --flash-attn \
    --bf16 \
    --learning_rate 0.000001 \
    --lr_scheduler constant \
    --num_ppo_epochs 1 \
    --beta 0 \
    --oracle_type reward \
    --oracle svg \
    --pretrain Qwen/Qwen2.5-3B \
    --prompt_template r1_svg \
    --zero-stage 2 \
    --ref_offload \
    --train_split train \
    --input_key solution \
    --output_key image_path \
    --max-train 100000 \
    --num_prompt_epoch 10 \
    --prompt_max_length 512 \
    --num_samples 8 \
    --temperature 1 \
    --top_p 1 \
    --generate_max_length 3000 \
    --log_completion_steps 10 \
    --save_steps 30 \
    --train_batch_size 128 \
    --train_batch_size_per_device 1 \
    --mini_train_batch_size_per_device 1 \
    --rollout_batch_size 128 \
    --rollout_batch_size_per_device 16 \
    --pi_buffer_maxlen_per_device 128 \
    --eval_batch_size 50 \
    --eval_steps -1 \
    --eval_temperature 1 \
    --eval_generate_max_length 3000 \
    --eval_data ./datasets/evaluation_suite \
    --eval_input_key input \
    --use-wb \
    --wb_project oat-zero-svg \
    --wb-run-name reward_siglipsmall \
    --prompt_data_svg coco_mix \
    --clip_model clip_small \
    --dino_coeff 0 
```

# Evaluating the Model

```bash
python evaluate_svg_model.py 
    --model_path YOUR_MODEL_PATH  
```
To get the VQA and HPS metrics, check [eval_tools](eval_tools/)

# SGP-CompBench

[sgp-compbench](sgp-compbench/)


