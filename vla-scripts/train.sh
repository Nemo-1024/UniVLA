


# Run your training script with torchrun
torchrun --nproc_per_node 8 train.py \
                                 --run_root_dir "vla_log" \