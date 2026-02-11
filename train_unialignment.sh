
torchrun --nnodes 1 \
    --node_rank 0 \
    --nproc_per_node 8 \
    --master_addr localhost \
    --master_port 29501 \
    train.py  \
        --config configs/config_unify.py \
        --block_depth 8 \
        --results_dir results \
        --model_parallel_size 1 \
        --data_parallel h_sdp \
        --precision bf16 --grad_precision fp32