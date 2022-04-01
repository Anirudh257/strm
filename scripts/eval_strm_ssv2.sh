#!/bin/bash
#SBATCH --job-name=strm
#SBATCH --gres gpu:4
#SBATCH --nodes 1
#SBATCH --cpus-per-task 24
#SBATCH --time=2-00:00:00

NEW_HOME="./strm_ssv2/"
COMMAND_TO_RUN="python3 run.py -c checkpoint_dir_ssv2/ --query_per_class 4 --shot 5 --way 5 --trans_linear_out_dim 1152 --tasks_per_batch 16 --test_iters 75000 --dataset ssv2 --split 7 -lr 0.001 --method resnet50 --img_size 224 --scratch new --num_gpus 4 --print_freq 1 --save_freq 75000 --training_iterations 75010 --temp_set 2 --test_model_only True --test_model_path checkpoints/checkpoint75000.pt"

echo ""
echo "Date = $(date)"
echo "Hostname = $(hostname -s)"
echo "Working Directory = $NEW_HOME"
echo "Command = $COMMAND_TO_RUN"
echo ""

cd $NEW_HOME
srun $COMMAND_TO_RUN

