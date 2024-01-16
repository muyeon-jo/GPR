import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.optim as optim
import csv
import random
import numpy as np
from torch_geometric.nn import GCNConv

from DataPreprocess import cal_distance

class GGLR(nn.Module):
    def __init__(self, embed_dim, layer_num):
        super(GGLR, self).__init__()
        self.embed_dim = embed_dim
        self.k = layer_num
        self.a = nn.Parameter(torch.FloatTensor(1).uniform_(-1,1))
        self.b = nn.Parameter(torch.FloatTensor(1).uniform_(-1,1))
        self.c = nn.Parameter(torch.FloatTensor(1).uniform_(-1,1))
        self.ingoing_conv1 = GCNConv(embed_dim,embed_dim)
        self.ingoing_conv2 = GCNConv(embed_dim,embed_dim)

        self.outgoing_conv1 = GCNConv(embed_dim,embed_dim)
        self.outgoing_conv2 = GCNConv(embed_dim,embed_dim)

        self.decode_layer = nn.Linear(embed_dim, embed_dim, bias=False)

        self.mse_loss = nn.MSELoss()
        self.leaky_relu = nn.LeakyReLU()
            
    def forward(self, p_outgoing, q_ingoing, adjacency_matrix, distance_matrix):
        # adj_mat = adjacency_matrix

        no = adjacency_matrix.clone()
        no[adjacency_matrix > 0.0] = 1
        D_outgoing = torch.sum(no, dim=-1) + 0.0000001
        D_ingoing = torch.sum(no.transpose(0, 1), dim=-1)+ 0.0000001
        
        e_ij_hat = []
        p_k = []
        q_k = []
        # t =adjacency_matrix.nonzero().T
        outgoing1 = self.outgoing_conv1(p_outgoing, adjacency_matrix.nonzero().T)
        outgoing1 = torch.mm(adjacency_matrix, outgoing1) #(poi * poi) (dot) (poi * emb)
        # tt = D_outgoing.reshape(-1,1)
        # ttt = D_ingoing.reshape(-1,1)
        outgoing1 = torch.div(outgoing1, D_outgoing.reshape(-1,1))
        outgoing1 = self.leaky_relu(outgoing1)

        outgoing2 = self.outgoing_conv2(outgoing1, adjacency_matrix.nonzero().T)
        outgoing2 = torch.mm(adjacency_matrix,outgoing2)
        outgoing2 = torch.div(outgoing2, D_outgoing.reshape(-1,1))
        outgoing2 = self.leaky_relu(outgoing2)
        
        ingoing1 = self.ingoing_conv1(q_ingoing, adjacency_matrix.T.nonzero().T)
        ingoing1 = torch.mm(adjacency_matrix.T, ingoing1) #(poi * poi) (dot) (poi * emb)
        ingoing1 = torch.div(ingoing1,  D_ingoing.reshape(-1,1))
        ingoing1 = self.leaky_relu(ingoing1)

        ingoing2 = self.ingoing_conv2(ingoing1, adjacency_matrix.T.nonzero().T)
        ingoing2 = torch.mm(adjacency_matrix.T,ingoing2)
        ingoing2 = torch.div(ingoing2, D_ingoing.reshape(-1,1))
        ingoing2 = self.leaky_relu(ingoing2)

        fx_ij = torch.mul(torch.mul(distance_matrix**self.b,self.a), torch.exp(torch.mul(distance_matrix,self.c)))
        e_ij_hat = torch.mul(torch.mm(self.decode_layer(outgoing2), ingoing2.T), fx_ij) 

        return [outgoing1,outgoing2], [ingoing1,ingoing2], e_ij_hat
    
    def loss_function(self, ground, predict):
        ground = ground.reshape(-1,1)
        predict = predict.reshape(-1,1)
        return self.mse_loss(ground, predict)
class GPR(nn.Module):
    def __init__(self, user_num, poi_num, embed_dim, layer_num, POI_POI_Graph, distance_matrix,user_POI_Graph, lambda1=0.2):
        super(GPR, self).__init__()
        self.POI_POI_Graph = torch.tensor(POI_POI_Graph, dtype=torch.float32).to("cuda")
        self.distance_matrix = distance_matrix
        self.user_POI_Graph = torch.tensor(user_POI_Graph, dtype= torch.float32).to("cuda")
        self.embed_dim = embed_dim
        self.k = layer_num
        self.user_num = user_num
        self.poi_num = poi_num
        self.sigmoid = nn.Sigmoid()
        self.lambda1 = lambda1
        self.gglr = GGLR(embed_dim, layer_num).to("cuda")

        self.user_embed = nn.Embedding(user_num,embed_dim) # t
        self.p_outgoing_embed = nn.Embedding(poi_num,embed_dim) # t
        self.q_incoming_embed = nn.Embedding(poi_num,embed_dim) # t

       
        
        self.user_layer1 = nn.Linear(embed_dim,embed_dim,bias=False)
        self.user_layer2 = nn.Linear(embed_dim,embed_dim,bias=False)

        self.outgoing_layer1 = GCNConv(embed_dim,embed_dim)
        self.outgoing_layer2 = GCNConv(embed_dim,embed_dim)

        self.init_emb()
    def init_emb(self):
        nn.init.xavier_normal_(self.user_embed.weight)
        nn.init.xavier_normal_(self.p_outgoing_embed.weight)
        nn.init.xavier_normal_(self.q_incoming_embed.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self,user_ids, train_positives, train_negatives ):
        #gglr result
        
        p1 = self.p_outgoing_embed(torch.LongTensor(range(self.poi_num)).to("cuda"))
        q1 = self.q_incoming_embed(torch.LongTensor(range(self.poi_num)).to("cuda"))
        u1 = self.user_embed(torch.LongTensor(range(self.user_num)).to("cuda"))

        p_k, q_k, e_ij_hat = self.gglr(p1, q1, self.POI_POI_Graph, self.distance_matrix)
        temp_user_emb = torch.tensor(np.zeros([self.user_num,self.embed_dim]), dtype= torch.float32, requires_grad = False).to("cuda")
        p_k[0] = torch.cat([p_k[0],temp_user_emb],dim = 0)
        p_k[1] = torch.cat([p_k[1],temp_user_emb],dim = 0)

        user1 = self.user_layer1(u1)
        edge_list = torch.stack([self.user_POI_Graph.nonzero().T[0].add(self.poi_num),self.user_POI_Graph.nonzero().T[1]])
        # p1 = torch.sum(self.outgoing_layer1(p_k[0],edge_list),dim=0)
        user1 = self.sigmoid(user1 + torch.sum(self.outgoing_layer1(p_k[0],edge_list),dim=0))

        user2 = self.user_layer2(user1)
        # p2 = torch.sum(self.outgoing_layer2(p_k[1],edge_list),dim=0)
        user2 = self.sigmoid(user2 + torch.sum(self.outgoing_layer2(p_k[1],edge_list),dim=0))


        result_u = torch.cat((user1,user2), dim=-1)
        result_q = torch.cat((q_k[0],q_k[1]),dim=-1)
        
        tt = result_u[user_ids]
        pp = result_q[train_positives]
        qq = result_q[train_negatives]

        rating_ul = torch.mm(tt, pp.T).diag()
        rating_ul_prime = torch.mm(tt,qq.T).diag()
        return rating_ul, rating_ul_prime, e_ij_hat 
    
    def loss_function(self, rating_ul, rating_ul_p, e_ij_hat):
        loss1 = self.gglr.loss_function(self.POI_POI_Graph,e_ij_hat)
        loss2 = -torch.sum(torch.log(self.sigmoid(rating_ul - rating_ul_p)+ 0.0000001))
        loss = loss2 + loss1*self.lambda1
        return loss