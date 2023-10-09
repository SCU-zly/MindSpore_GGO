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


save_dir="random_hx_forward"
test_path='/data/zly/dataset/hx_forward_test.npy'
# test_path='/data/zly/dataset/other_center_test.npy'
# pre_model='/data/zly/DeepGGO/saved_models/forward_resnet50_32_aug2_O_no_under_AUCH_clt0_alp_001_lambda_1_0_712/size_32/resnet_50/019.ckpt'
# pre_model='/data/zly/DeepGGO/saved_models/forward_resnet50_32_aug2_O_no_under_AUCH_clt0_alp_001_lambda_1_0_712/size_32/resnet_50/019.ckpt'
# pre_model='/data/zly/DeepGGO/saved_models/forward_resnet34_32_aug2_O_no_under_AUCH_clt0_alp_001_lambda_1_0_712/size_32/resnet_34/023.ckpt'
pre_model='random'
python Test_SNE.py --model 'resnet' --sample_size 32 --n_classes 1 \
 --sample_duration 16 --model_depth 34 --save_dir $save_dir \
 --batch_size 1 --clt 0 --model_path $pre_model --test_path $test_path

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

