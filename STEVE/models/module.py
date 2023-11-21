import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models.layers import STConvBlock, cal_cheb_polynomial,cal_laplacian

#Minimum Mutual information
# CLUB: Mutual Information Contrastive Learning Upper Bound
class CLUB(nn.Module):  
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

class ST_encoder(nn.Module):
    def __init__(self, num_nodes, d_input, d_output, Ks, Kt, blocks, input_window, drop_prob, device):
        super().__init__()
        self.num_nodes = num_nodes
        self.feature_dim = d_output
        self.output_dim = d_output

        self.Ks = Ks
        self.Kt = Kt
        self.blocks = blocks
        self.input_window = input_window
        self.output_window = 1
        self.drop_prob = drop_prob


        self.blocks[0][0] = self.feature_dim
        if self.input_window - len(self.blocks) * 2 * (self.Kt - 1) <= 0:
            raise ValueError('Input_window must bigger than 4*(Kt-1) for 2 STConvBlock'
                             ' have 4 kt-kernel convolutional layer.')
        self.device = device
        self.input_conv=nn.Conv2d(d_input, d_output, 1)

        self.st_conv1 = STConvBlock(self.Ks, self.Kt, self.num_nodes,
                                    self.blocks[0], self.drop_prob, self.device)
        self.st_conv2 = STConvBlock(self.Ks, self.Kt, self.num_nodes,
                                    self.blocks[1], self.drop_prob, self.device)

    def forward(self, x ,graph):
        # (batch_size, input_length, num_nodes, feature_dim)nclv

        lap_mx = cal_laplacian(graph)
        Lk = cal_cheb_polynomial(lap_mx, self.Ks)

        x=self.input_conv(x)
        x_st1 = self.st_conv1(x,Lk)   # (batch_size, c[2](64), input_length-kt+1-kt+1, num_nodes)
        x_st2 = self.st_conv2(x_st1,Lk)  # (batch_size, c[2](128), input_length-kt+1-kt+1-kt+1-kt+1, num_nodes)

        return x_st2

    def variant_encode(self,x,graph):
        x=self.input_conv(x)
        x_st1 = self.st_conv1(x, graph)   # (batch_size, c[2](64), input_length-kt+1-kt+1, num_nodes)
        x_st2 = self.st_conv2(x_st1, graph)  # (batch_size, c[2](128), input_length-kt+1-kt+1-kt+1-kt+1, num_nodes)
        return x_st2