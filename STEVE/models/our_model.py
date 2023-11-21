import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import masked_mae_loss
from models.module import ST_encoder,CLUB
from models.layers import RevGradLayer


class STEVE(nn.Module):
    def __init__(
            self,
            args,
            adj,
            in_channels=1,
            embed_size=64,
            T_dim=12,
            output_T_dim=1,
            output_dim=2,
            device="cuda"
    ):
        super(STEVE, self).__init__()

        self.args = args
        self.adj = adj

        self.time_labels = 48

        self.embed_size = embed_size

        
        T_dim = args.input_length-4*(3-1)

        temp_spatial_label=list(range(args.num_nodes))

        self.spatial_label=torch.tensor(temp_spatial_label,device=args.device)

        # encoder
        self.st_encoder4variant = ST_encoder(args.num_nodes, args.d_input, args.d_model, 3, 3,
                               [[args.d_model, args.d_model // 2, args.d_model],
                                [args.d_model, args.d_model // 2, args.d_model]], args.input_length, args.dropout,
                               args.device)

        self.st_encoder4invariant = ST_encoder(args.num_nodes, args.d_input, args.d_model, 3, 3,
                        [[args.d_model, args.d_model // 2, args.d_model],
                        [args.d_model, args.d_model // 2, args.d_model]], args.input_length, args.dropout,
                        args.device)

        # dynamic adj metric
        self.node_embeddings_1 = nn.Parameter(torch.randn(3, args.num_nodes, embed_size), requires_grad=True)
        self.node_embeddings_2 = nn.Parameter(torch.randn(3, embed_size, args.num_nodes), requires_grad=True)


        # predict

        self.variant_predict_conv_1 = nn.Conv2d(T_dim, output_T_dim, 1)
        
        self.variant_predict_conv_2 = nn.Conv2d(embed_size, output_dim, 1)

        self.invariant_predict_conv_1 = nn.Conv2d(T_dim, output_T_dim, 1)
        
        self.invariant_predict_conv_2 = nn.Conv2d(embed_size, output_dim, 1)

        self.relu = nn.ReLU()

        #variant 

        self.variant_tconv = nn.Conv2d(in_channels=T_dim,
                                       out_channels=1,
                                       kernel_size=(1, 1),
                                       bias=True)
        self.variant_end_temproal = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Linear(embed_size * 2, self.time_labels)
        )
        self.variant_end_spacial = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Linear(embed_size * 2, args.num_nodes)
        )

        self.variant_end_congest = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.ReLU(),
            nn.Linear(embed_size // 2 , 2)
        )

        # invariant
        self.invariant_tconv = nn.Conv2d(in_channels=T_dim,
                                       out_channels=1,
                                       kernel_size=(1, 1),
                                       bias=True)
        self.invariant_end_temporal = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Linear(embed_size*2, self.time_labels)
        )
        self.invariant_end_spatial = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.ReLU(),
            nn.Linear(embed_size*2, args.num_nodes)
        )

        self.invariant_end_congest = nn.Sequential(
            nn.Linear(embed_size, embed_size // 2),
            nn.ReLU(),
            nn.Linear(embed_size // 2 , 2)
        )

        self.alpha_linear=nn.Linear(2, 2)
        self.beta_linear=nn.Linear(2, 2)

        self.revgrad=RevGradLayer()

        self.mask=torch.zeros([args.batch_size,args.d_input,args.input_length,args.num_nodes],dtype=torch.float).to(device)
        self.receptive_field = args.input_length + 8

        self.mse_loss=torch.nn.MSELoss()

        # self.mi_net=CLUB(embed_size,embed_size,embed_size*2) # regularizer

        # self.optimizer_mi_net=torch.optim.Adam(self.mi_net.parameters(),lr=args.lr_init)

        self.mae = masked_mae_loss(mask_value=5.0)
        
        # self.alpha=nn.Parameter(torch.rand(1,args.num_nodes,2))


    def forward(self, x):

        x = x.permute(0, 3, 1, 2) #NCLV
        # fix me
        # encoder_inputs = self.conv1(x)
        invariant_output = self.st_encoder4invariant(x,self.adj)
        invariant_output = invariant_output.permute(0, 2, 1, 3)

        adaptive_adj = F.softmax(F.relu(torch.bmm(self.node_embeddings_1, self.node_embeddings_2)), dim=1)
        variant_output = self.st_encoder4variant.variant_encode(x,adaptive_adj)
        variant_output = variant_output.permute(0, 2, 1, 3)


        # return out shape: [v, output_dim]->[B,V,output_dim]
        return invariant_output, variant_output
        



    def predict(self, z1, z2,x):
        out_1 = self.relu(self.invariant_predict_conv_1(z1))  # out shape: [1, output_T_dim, N, C]
        out_1 = out_1.permute(0, 2, 3, 1)  # out shape: [1, C, N, output_T_dim] 64 128 64 1
        out_1 = self.invariant_predict_conv_2(out_1)  # out shape: [b, 1, N, output_T_dim]
        out_1=out_1.permute(0,3,2,1)

        out_2 = self.relu(self.variant_predict_conv_1(z2))  # out shape: [1, output_T_dim, N, C]
        out_2 = out_2.permute(0, 2, 3, 1)  # out shape: [1, C, N, output_T_dim]
        out_2 = self.variant_predict_conv_2(out_2)  # out shape: [b, c, N, t]
        out_2 = out_2.permute(0,3,2,1)


        # todolist change
        # x=x.permute(0,2,3,1)#nclv
        # c=self.generator_conv(x) # b，1，n，2
        # out_c = self.linear_c(c)
        # out_2 = out_2*out_c
        
        # self.out_c=out_c

        # out =torch.cat([out_1,out_2],dim=1).permute(0,2,3,1)
        # out = self.out_mlp(out).squeeze()
        alpha=self.alpha_linear(out_1)
        beta=self.beta_linear(out_2)
        
        temp=torch.stack([alpha,beta],dim=-1) # b t n c 2
        
        temp=F.softmax(temp,dim=-1)
        alpha,beta=temp[...,0],temp[...,1]

        out_1=out_1*alpha
        out_2=out_2*beta
        
        out = out_2 + out_1
        out = out.squeeze(1)
        out_1=out_1.squeeze(1)
        out_2=out_2.squeeze(1)
        return out
    
    def variant_loss(self, z2, date, c):
        # z2.shape=[b,t,c,n]
        # date = [b,time_intervals]
        z2 = self.variant_tconv(z2).squeeze(1)  # [b,c,n]
        z_temporal = z2.mean(2).squeeze()
        
        y_temporal = self.variant_end_temproal(z_temporal)  # b,time_num
        loss_temproal = F.cross_entropy(y_temporal, date)

        z_spatial = z2.transpose(2,1) # b n c 
        y_spatial = self.variant_end_spacial(z_spatial)
        y_spatial = y_spatial.mean(0)
        loss_spacial = F.cross_entropy(y_spatial,self.spatial_label)

        z2_congest = z2.transpose(2,1).unsqueeze(1)
        y_congest = self.variant_end_congest(z2_congest)
        loss_congest = self.mse_loss(y_congest,c)

        loss = (loss_spacial+loss_temproal+loss_congest)/3.      

        return loss
    

    def invariant_loss(self, z1, date,c):
        # z1.shape=[b,t,c,n]
        # recover_loss
        #revgrad loss
        z1_r = self.revgrad(z1)

        z1_r = self.invariant_tconv(z1_r).squeeze(1)  # [b,1,n,c]
        z1_temporal = z1_r.mean(2).squeeze()
        
        y_temporal = self.invariant_end_temporal(z1_temporal)  # b,time_num
        loss_temporal = F.cross_entropy(y_temporal, date) 

        z1_spatial = z1_r.transpose(2,1)
        y_spatial = self.invariant_end_spatial(z1_spatial)# b,num_nodes
        y_spatial = y_spatial.mean(0)# num_nodes
        loss_spatial = F.cross_entropy(y_spatial,self.spatial_label)


        z1_congest = z1_r.transpose(2,1).unsqueeze(1)
        y_congest = self.invariant_end_congest(z1_congest)
        loss_congest = self.mse_loss(y_congest,c)
        
        loss = (loss_spatial+loss_temporal+loss_congest)/3.
        return loss # shape=[]

    def pred_loss(self, z1, z2, x, y_true, scaler):
        y_pred = self.predict(z1, z2, x)
        y_pred = scaler.inverse_transform(y_pred)
        y_true = scaler.inverse_transform(y_true)
        y_true = y_true.squeeze(1)

        loss = self.args.yita * self.mae(y_pred[..., 0], y_true[..., 0])
        loss += (1 - self.args.yita) * self.mae(y_pred[..., 1], y_true[..., 1])
        return loss #[1]


    def calculate_loss(self, x, z1, z2, target, c, time_label,scaler,loss_weights,training=False):
        # z1.shape=btcv
        l1 = self.pred_loss(z1, z2, x, target, scaler)
        loss=0 
        l4=0

        sep_loss = [l1.item()]

        # mi_net train
        if training and self.args.MMI:
            z1_temp=z1.transpose(3,2).mean(1).mean(1)
            z2_temp=z2.transpose(3,2).mean(1).mean(1)
            self.mi_net.train()
            temp1=z1_temp.detach()
            temp2=z2_temp.detach()
            for i in range(5):
                self.optimizer_mi_net.zero_grad()
                mi_loss=self.mi_net.learning_loss(temp1,temp2)
                mi_loss.backward()
                self.optimizer_mi_net.step()
            self.mi_net.eval()
            l4 = 0.1*self.mi_net(z1_temp,z2_temp)
            loss += l4

        loss += loss_weights[0] * l1

        l2 = self.variant_loss(z2, time_label, c)
        sep_loss.append(l2.item())
        loss += loss_weights[1] * l2


        l3 = self.invariant_loss(z1, time_label ,c)
        sep_loss.append(l3.item())
        loss += loss_weights[2] * l3

        if training == False:
            if self.args.lr_mode=='only':
                loss = l1
            elif self.args.lr_mode=='add':
                loss = l1+l2

        return loss, sep_loss
