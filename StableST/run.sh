## predefined super parameters
# lr_set=(0.005 0.001 0.0005 0.0001)
# bs_set=(32 64 128)
# d_set=(32 64 128)
# max_epoch=(200 400)
# lr_mode_set=('only' 'add')
# seed_set=(31 42 131)

## tuning
# for lr in 0.005 0.001 0.0005 0.0001
# do
#     for bs in 32 64 128
#     do
#         for d in 32 64 128
#         do
#             for lr_mode in 'only' 'add'
#             do
#                 for seed in 31 42 131
#                 do
#                     for epoch in 200 400
#                     do
#                         /home/zhangwt/miniconda3/envs/torch/bin/python /home/zhangwt/StableST/run.py --lr $lr --bs $bs --d $d --lr_mode $lr_mode --seed $seed --max_epoch $epoch --config_filename=configs/$1.yaml
#                     done
#                 done
#             done
#         done
#     done
# done

## only run 
# python run.py --config_filename=configs/$1.yaml

# 首先找到最好的設置
# NYCBike2 0.01,64,32,42,add,400
# NYCTaxi 0.01,32,64,42,only,400
# BJTaxi  0.05,32,32,131,add,400
# ablation
#/home/zhangwt/miniconda3/envs/torch/bin/python /home/zhangwt/StableST/run.py --config_filename=configs/$1.yaml --ablation='causal'
#/home/zhangwt/miniconda3/envs/torch/bin/python /home/zhangwt/StableST/run.py --config_filename=configs/$1.yaml --ablation='regularizer'
/home/zhangwt/miniconda3/envs/torch/bin/python /home/zhangwt/StableST/run.py --config_filename=configs/$1.yaml --bs 16
/home/zhangwt/miniconda3/envs/torch/bin/python /home/zhangwt/StableST/run.py --config_filename=configs/$1.yaml --d 16
#/home/zhangwt/miniconda3/envs/torch/bin/python /home/zhangwt/StableST/run.py --config_filename=configs/$1.yaml --ablation='only'
