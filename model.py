import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.optim as optim
import csv
import random
import numpy as np

from DataPreprocess import cal_distance

class GGLR(nn.Module):
    def __init__(self, embed_dim, layer_num):
        super(GGLR, self).__init__()
        self.embed_dim = embed_dim
        self.k = layer_num
        self.a = nn.Parameter(torch.FloatTensor(1).uniform_(-1,1))
        self.b = nn.Parameter(torch.FloatTensor(1).uniform_(-1,1))
        self.c = nn.Parameter(torch.FloatTensor(1).uniform_(-1,1))

        self.decode_layer = nn.Linear(embed_dim, embed_dim)
        self.layers = []
        self.mse_loss = nn.MSELoss()
        for i in range(layer_num):
            self.layers.append(nn.Linear(embed_dim, embed_dim))
            
    def forward(self, p_outgoing, q_incoming, adjacency_matrix, distance_matrix):
        e_ij= adjacency_matrix
        e_ji = adjacency_matrix.transpose(0, 1)

        D_outgoing = torch.sum(adjacency_matrix, dim=0)
        D_incoming = torch.sum(adjacency_matrix.transpose(0, 1), dim=0)

        p_k = []
        q_k = []
        
        for i in range(self.k):
            new_p = F.leaky_relu(torch.sum(e_ij * self.layers[i](p_outgoing),dim=0) / D_outgoing)
            p_k.append(new_p)
            new_q = F.leaky_relu(torch.sum(e_ji * self.layers[i](q_incoming),dim=0) / D_incoming)
            q_k.append(new_q)

        fx_ij = self.a*distance_matrix**self.b * torch.exp(self.c*distance_matrix)
        e_ij_hat = torch.dot(self.decode_layer(p_k[self.k-1]), q_k[self.k-1]) * fx_ij

        return p_k, q_k, e_ij_hat
    
    def loss_function(self, ground, predict):
        self.mse_loss(ground, predict)
class GPR(nn.Module):
    def __init__(self, user_num, poi_num, embed_dim, layer_num, adjacency_matrix, distance_matrix):
        super(GPR, self).__init__()
        self.adjacency_matrix = adjacency_matrix
        self.distance_matrix = distance_matrix
        self.embed_dim = embed_dim
        self.k = layer_num
        self.user_num = user_num
        self.poi_num = poi_num
        self.sigmoid = nn.Sigmoid()
        self.gglr = GGLR(embed_dim, layer_num)

        self.user_embed = nn.Embedding(user_num,embed_dim) # t
        self.p_outgoing_embed = nn.Embedding(poi_num,embed_dim) # t
        self.q_incoming_embed = nn.Embedding(poi_num,embed_dim) # t

       
        self.user_layers = []
        self.outgoing_layers = []
        for i in range(layer_num):
            self.user_layers.append(nn.Linear(embed_dim, embed_dim,bias=False))
            self.outgoing_layers.append(nn.Linear(embed_dim, embed_dim))
        self.init_emb()
    def init_emb(self):
        nn.init.xavier_normal_(self.user_embed.weight)
        nn.init.xavier_normal_(self.p_outgoing_embed.weight)
        nn.init.xavier_normal_(self.q_incoming_embed.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self,users, user_histories, user_negatives):
        #gglr result
        p_outgoing = self.p_outgoing_embed(torch.LongTensor(range(self.poi_num)))
        q_incoming = self.q_incoming_embed(torch.LongTensor(range(self.poi_num)))
        p_k, q_k, e_ij_hat = self.gglr(p_outgoing, q_incoming, self.adjacency_matrix, self.distance_matrix)

        u_k = []
        newu = self.user_embed(users)
        for i in range(self.k):
            newu = self.user_layers[i](newu) + torch.sum(self.outgoing_layers[i](p_k[i]),dim=0)
            u_k.append(newu)
        result_u = u_k[0]
        result_q = q_k[0]
        for i in range(1,self.k):
            result_u = torch.cat(result_u, u_k[i], dim = -1)
            result_q = torch.cat(result_q, q_k[i], dim = -1)

        rating_ul = torch.dot(result_u, result_q)
        return rating_ul
    
    def loss_function(self, rating_ul, rating_ul_p):
        loss = -torch.sum(torch.log(self.sigmoid(rating_ul - rating_ul_p)))
        return loss