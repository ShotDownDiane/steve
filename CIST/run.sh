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
#                         python run.py --lr $lr --bs $bs --d $d --lr_mode $lr_mode --seed $seed --max_epoch $epoch --config_filename=configs/$1.yaml
#                     done
#                 done
#             done
#         done
#     done
# done

## only run 
# python run.py --config_filename=configs/$1.yaml

