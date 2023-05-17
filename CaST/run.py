import os

import warnings
from lib.metrics import test_metrics

warnings.filterwarnings('ignore')

import yaml
import argparse
import train
import test

from lib.utils import get_project_path

def text2args(text,args):
    temp=text.split(",")
    args=argparse.Namespace()
    for s in temp:
        key,value=s.split("=")
        if '\'' in value:
            args[key] = value
        else:
            args[key] = int(value)
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train',
                        type=str, help='the configuration to use')
    parser.add_argument('--config_filename', default='configs/NYCBike1.yaml',
                        type=str, help='the configuration to use')
    parser.add_argument('--lr', default=None,
                    type=float, help='init learning rate')
    parser.add_argument('--bs', default=None,
                    type=int, help='batch size')
    parser.add_argument('--d', default=None,
                        type=int, help='the dimition of encoder')
    parser.add_argument('--seed', default=None,
                    type=int, help='random seed')
    parser.add_argument('--lr_mode', default=None,
                    type=str, help='random seed')
    parser.add_argument('--max_epoch', default=None,
                    type=int, help='random seed')
    
    parser.add_argument('--ablation', default='all',
                    type=str, help='ablation study')
    
    args = parser.parse_args()

    args.config_filename = os.path.join(get_project_path(),args.config_filename)

    print(f'Starting experiment with configurations in {args.config_filename}...')
    configs = yaml.load(
        open(args.config_filename),
        Loader=yaml.FullLoader
    )
    if args.lr:
        configs['lr_init']=args.lr

    if args.bs:
        configs['batch_size']=args.bs
    
    if args.seed:
        configs['seed']=args.seed

    if args.d:
        configs['d_model']=args.d
    
    if args.lr_mode:
        configs['lr_mode']=args.lr_mode
    
    if args.max_epoch:
        configs['epochs']=args.max_epoch
    
    configs['ablation']=args.ablation

    
    

    args = argparse.Namespace(**configs)

    args.graph_file = os.path.join(get_project_path(), args.graph_file)
    args.data_dir = os.path.join(get_project_path(), args.data_dir)

    if args.mode=="train":
        train.main(args)
    elif args.mode=="gat":
        # cross domain call
        pass
    elif args.mode=="test":
        # best_paths=[]
        # for best_path in best_paths:
        #     config_file_path=os.join(best_path,'run.log')
        #     config_file=open(config_file_path)
        #     config=config_file.readlines()
        #     config=config[55:-2]
        pass
            