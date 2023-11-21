import os
from datetime import datetime

import warnings

from lib.metrics import test_metrics

warnings.filterwarnings('ignore')

import torch

from lib.utils import get_project_path

from lib.utils import (
    init_seed,
    get_model_params,
    load_graph, get_log_dir,
)

from lib.dataloader import get_dataloader
from lib.logger import get_logger, PD_Stats
from lib.utils import dwa
import numpy as np
from models.our_model import STEVE

from munch import DefaultMunch


def text2args(text):
    args_dict={}
    temp=text.split(", ")
    for s in temp:
        key,value=s.split("=")
        if '\'' in value:
            args_dict[key] = value.replace('\'','')
        elif '.' in value:
            args_dict[key] =float(value)
        elif 'False' in value:
            args_dict[key] = False
        elif 'True' in value:
            args_dict[key] =True
        else:
            args_dict[key] = int(value)
    args=DefaultMunch.fromDict(args_dict)
    return args


def test(model, dataloader, scaler):
    model.eval()
    invariant_pred=[]
    variant_pred=[]
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_idx, (data, target, c) in enumerate(dataloader):
            repr1, repr2 = model(data)
            # c_hat=model.predict_con(data)
            invariant,variant,pred_output = model.predict(repr1, repr2,data)
            target = target.squeeze(1)
            invariant_pred.append(invariant)
            variant_pred.append(variant)
            y_true.append(target)
            y_pred.append(pred_output)
    invariant_pred = scaler.inverse_transform(torch.cat(invariant_pred, dim=0)).cpu()
    variant_pred = scaler.inverse_transform(torch.cat(variant_pred, dim=0)).cpu()
    y_true = scaler.inverse_transform(torch.cat(y_true, dim=0)).cpu()
    y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0)).cpu()

    return invariant_pred,variant_pred,y_true, y_pred


def make_one_hot(labels, classes):
    # labels=labels.to('cuda:1')
    labels = labels.unsqueeze(dim=-1)
    one_hot = torch.FloatTensor(labels.size()[0], classes).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

def main(args):
    
    A = load_graph(args.graph_file, device=args.device)  

    init_seed(args.seed)

    dataloader = get_dataloader(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        device=args.device
    )

    # current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    # current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    # log_dir = os.path.join(current_dir, 'experiments', 'NYCBike1', current_time)
    model = STEVE(args=args, adj=A, in_channels=args.d_input, embed_size=args.d_model, time_num=288, num_layers=args.layers,
                T_dim=args.input_length, output_T_dim=1, output_dim=args.d_output, heads=2,device=args.device).to(args.device)
    
    best_path=os.path.join(args.log_dir,'best_model.pth')
    print('load model from {}.'.format(best_path))
    state_dict = torch.load(
                best_path,
                map_location=torch.device(args.device)
            )
    model.load_state_dict(state_dict['model'])


    invariant,variant,y_true,y_pred = test(model, dataloader['test'], dataloader['scaler'])
    result_path=os.path.join(args.log_dir,'result.npz')
    print('save result in {}.'.format(result_path))
    np.savez(result_path,invariant=invariant,variant=variant,y_true=y_true,y_pred=y_pred)

if __name__ == '__main__':

    d='/home/zhangwt/StableST/experiments/BJTaxi'
    best_paths = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    file_list=[]
    mae_list=[]
    for best_path in best_paths:
        config_file_path=os.path.join(best_path,'run.log')
        config_file=open(config_file_path)
        config_str=config_file.readlines()
        config=config_str[1]
        config=config[55:-2]
        args=text2args(config)
        
        if args.batch_size== 32 and args.d_model==32 and args.seed==31:
            temp=config_str[-2]
            try:
                print(temp[34:39]) 
                mae=float(temp[34:39])
            except Exception:
                mae=100
            file_list.append(best_path)
            mae_list.append(mae)
    index=np.argsort(mae_list)
    print(file_list[index[0]])
    














