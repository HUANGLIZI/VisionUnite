#!/usr/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=8 --use_env \
 main_pretrain.py --batch_size 4 --epochs 30 --warmup_epochs 1 --blr 5e-4 --weight_decay 0.02 \
 --output_dir /llama-adapter/imagebind-llm/output_dir/ \
 --resume /llama-adapter/imagebind-llm/output_dir/checkpoint-9_4-29.pth \
 --llama_path /llama-adapter/llama_model_weights