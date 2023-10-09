#！/bin/bash 
clear
export CUDA_VISIBLE_DEVICES=0
save_dir="forward_C3D_32_aug2_O_no_under_AUCH_clt0_alp_001_lambda_1_0_712/"
python TrainGGO-712.py --model 'c3d' --sample_size 32 --epochs 150 --n_classes 1 --sample_duration 16 --model_depth 121 --batch_size 32 --lr 0.001 --num_valid 1 --save_dir $save_dir --aug 2 --sample 'over' --clt 0 --alpha 1 --lamb 1


