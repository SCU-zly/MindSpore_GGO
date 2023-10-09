#！/bin/bash 

clear
export CUDA_VISIBLE_DEVICES=0

## No alpha No lamb parameters, then using defaut (0.1, 1)
# clear
# save_dir="back_10_32_aug2_no_over_no_under_clt2_CWAUCH_alp_001_lambda_1_0_s/"
# data_dir="./data_dir/20200106/"$save_dir

# if [ ! -d $data_dir ];then
# mkdir -p $data_dir 
# else
# echo $data_dir" 存在！" 
# fi

# python tools/preprocess_GGOMM_20191214.py $data_dir

# for valid_num in 0 1 2 3 4 5 6 7 8 9
# do
# python TrainGGO.py --model 'resnet' --sample_size 32 --epochs 120 --n_classes 1 --sample_duration 16 --model_depth 10 --batch_size 32 --lr 0.001 --num_valid $valid_num --save_dir $save_dir --aug 2 --sample 'none' --clt 2 --data_dir $data_dir --alpha 0.01 --lamb 1
# done


save_dir="forward_10_32_aug2_no_over_U_AUCH_clt0_alp_001_lambda_1_0_fold10/"

python TrainGGO-10fold.py --model 'resnet' --sample_size 32 --epochs 300 --n_classes 1 --sample_duration 16 --model_depth 10 --batch_size 32 --lr 0.001 --num_valid 1 --save_dir $save_dir --aug 2 --sample 'under' --clt 0 --alpha 1 --lamb 1

# Just for evaluating the performance of PLTI comparing with human
# save_dir="back_18_32_aug2_no_over_no_under_clt2_CWAUCH_alp_1_lambda_1_0_X/"
# data_dir="./data_dir/TESTING/"

# if [ ! -d $data_dir ];then
# mkdir -p $data_dir 
# else
# echo $data_dir" 存在！" 
# fi

# python tools/preprocess_GGOMM_20191214.py $data_dir

# python TrainGGO.py --model 'resnet' --sample_size 32 --epochs 300 --n_classes 1 --sample_duration 16 --model_depth 18 --batch_size 32 --lr 0.001 --num_valid 0 --save_dir $save_dir --aug 2 --sample 'none' --clt 2 --data_dir $data_dir --alpha 1 --lamb 1 --tf 0

