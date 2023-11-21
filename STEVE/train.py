import os
import traceback
from datetime import datetime

import warnings

from lib.metrics import test_metrics

warnings.filterwarnings('ignore')

import yaml
import argparse
import time
import torch

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


class Trainer(object):
    def __init__(self, model, optimizer, dataloader, graph, lr_scheduler,args, load_state=None):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_loader = dataloader['train']
        self.val_loader = dataloader['val']
        self.test_loader = dataloader['test']
        self.scaler = dataloader['scaler']
        self.graph = graph
        self.lr_scheduler=lr_scheduler
        self.args = args


        self.train_per_epoch = len(self.train_loader)
        if self.val_loader != None:
            self.val_per_epoch = len(self.val_loader)

        # log
        args.log_dir = get_log_dir(args)
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.log_dir, debug=args.debug)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')

        # create a panda object to log loss and acc
        self.training_stats = PD_Stats(
            os.path.join(args.log_dir, 'stats.pkl'),
            ['epoch', 'train_loss', 'val_loss'],
        )
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info('Experiment configs are: {}'.format(args))

    def train_epoch(self, epoch, loss_weights):
        self.model.train()

        total_loss = 0
        total_sep_loss = np.zeros(3)
        start_time=datetime.now()
        for batch_idx, (data, target, time_label,c) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            # input shape: n,l,v,c; graph shape: v,v;
            t1=datetime.now()
            repr1, repr2 = self.model(data)  # nvc
            loss, sep_loss = self.model.calculate_loss(data,repr1, repr2, target, c, time_label, self.scaler, loss_weights,True)
            t2=datetime.now()
            assert not torch.isnan(loss)

            loss.backward()
            # gradient clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    get_model_params([self.model]),
                    self.args.max_grad_norm)
            t3=datetime.now()
            self.optimizer.step()
            total_loss += loss.item()
            total_sep_loss += sep_loss
        endtime=datetime.now()
        print(endtime-start_time)
        print(t1-start_time)
        print(t2-start_time)
        print(t3-start_time)

        train_epoch_loss = total_loss / self.train_per_epoch
        total_sep_loss = total_sep_loss / self.train_per_epoch
        self.logger.info('*******Train Epoch {}: averaged Loss : {:.6f}'.format(epoch, train_epoch_loss))

        return train_epoch_loss, total_sep_loss

    def val_epoch(self, epoch, val_dataloader, loss_weights):
        self.model.eval()

        total_val_loss = 0
        total_sep_loss = np.zeros(3)
        with torch.no_grad():
            for batch_idx, (data, target,time_label,c) in enumerate(val_dataloader):
                repr1, repr2 = self.model(data)
                # c_hat=self.model.predict_con(data)
                loss, sep_loss = self.model.calculate_loss(data,repr1, repr2, target, c, time_label,self.scaler, loss_weights)
                # loss = self.model.pred_loss(repr1, repr1, target, self.scaler)
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                total_sep_loss += sep_loss
        val_loss = total_val_loss / len(val_dataloader)
        total_sep_loss = total_sep_loss /len(val_dataloader)
        self.logger.info('*******Val Epoch {}: averaged Loss : {:.6f} sep loss : {}'.format(epoch, val_loss, total_sep_loss))
        return val_loss

    def train(self):
        best_loss = float('inf')
        best_epoch = 0
        not_improved_count = 0
        start_time = time.time()

        loss_tm1 = loss_t = np.ones(3)  # (1.0, 1.0, 1.0)
        for epoch in range(1, self.args.epochs + 1):
            # dwa mechanism to balance optimization speed for different tasks

            if self.args.use_dwa:
                loss_tm2 = loss_tm1
                loss_tm1 = loss_t
                if (epoch == 1) or (epoch == 2):
                    loss_weights = dwa(loss_tm1, loss_tm1, self.args.temp)
                else:
                    loss_weights = dwa(loss_tm1, loss_tm2, self.args.temp)
            self.logger.info('loss weights: {}'.format(loss_weights))
            train_epoch_loss, loss_t = self.train_epoch(epoch, loss_weights)
            # train_epoch_loss = self.train_epoch(epoch, loss_weights)

            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            val_dataloader = self.val_loader if self.val_loader != None else self.test_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader, loss_weights)

            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                best_epoch = epoch
                not_improved_count = 0
                # save the best state
                save_dict = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }
                if not self.args.debug:
                    self.logger.info('**************Current best model saved to {}'.format(self.best_path))
                    torch.save(save_dict, self.best_path)
            else:
                not_improved_count += 1
            
            self.lr_scheduler.step(val_epoch_loss)

            #early stopping
            if self.args.early_stop and not_improved_count == self.args.early_stop_patience:
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                 "Training stops.".format(self.args.early_stop_patience))
                break


        training_time = time.time() - start_time
        self.logger.info("== Training finished.\n"
                         "Total training time: {:.2f} min\t"
                         "best loss: {:.4f}\t"
                         "best epoch: {}\t".format(
            (training_time / 60),
            best_loss,
            best_epoch))

        # test
        state_dict = save_dict if self.args.debug else torch.load(
            self.best_path, map_location=torch.device(self.args.device))
        self.model.load_state_dict(state_dict['model'])
        self.logger.info("== Test results.")
        test_results = self.test(self.model, self.test_loader, self.scaler,
                                 self.graph, self.logger, self.args)
        results = {
            'best_val_loss': best_loss,
            'best_val_epoch': best_epoch,
            'test_results': test_results,
        }

        return results

    @staticmethod
    def test(model, dataloader, scaler, graph, logger, args):
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target, c) in enumerate(dataloader):
                repr1, repr2 = model(data)
                # c_hat=model.predict_con(data)
                pred_output = model.predict(repr1, repr2,data)
                target = target.squeeze(1)
                y_true.append(target)
                y_pred.append(pred_output)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))

        test_results = []
        # inflow
        mae, mape = test_metrics(y_pred[..., 0], y_true[..., 0])
        logger.info("INFLOW, MAE: {:.2f}, MAPE: {:.4f}%".format(mae, mape * 100))
        test_results.append([mae, mape])
        # outflow
        mae, mape = test_metrics(y_pred[..., 1], y_true[..., 1])
        logger.info("OUTFLOW, MAE: {:.2f}, MAPE: {:.4f}%".format(mae, mape * 100))
        test_results.append([mae, mape])

        return np.stack(test_results, axis=0)


def make_one_hot(labels, classes):
    # labels=labels.to('cuda:1')
    labels = labels.unsqueeze(dim=-1)
    one_hot = torch.FloatTensor(labels.size()[0], classes).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

def main(args):

    A = load_graph(args.graph_file, device=args.device)  # �ڽӾ���

    init_seed(args.seed)

    dataloader = get_dataloader(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        device=args.device
    )

    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    log_dir = os.path.join(current_dir, 'experiments', 'NYCBike1', current_time)
    model = STEVE(args=args, adj=A, in_channels=args.d_input, embed_size=args.d_model,
                T_dim=args.input_length, output_T_dim=1, output_dim=args.d_output,device=args.device).to(args.device)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lr_init,
        eps=1.0e-8,
        weight_decay=0,
        amsgrad=False
    )
    lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.lr_patience, verbose=True, threshold=0.0001, threshold_mode='rel', min_lr=0.000005, eps=1e-08)

    # start training
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        graph=A,
        lr_scheduler=lr_scheduler,
        args=args
    )

    results = None
    try:
        if args.mode == 'train':
            results = trainer.train() # best_eval_loss, best_epoch
        elif args.mode == 'test':
            # test
            state_dict = torch.load(
                args.best_path,
                map_location=torch.device(args.device)
            )
            model.load_state_dict(state_dict['model'])
            print("Load saved model")
            results = trainer.test(model, dataloader['test'], dataloader['scaler'],
                        A, trainer.logger, trainer.args)
        else:
            raise ValueError
    except:
        trainer.logger.info(traceback.format_exc())












