from argparse import ArgumentParser
from datetime import datetime
import numpy as np
import os
import pandas as pd
import eval_metrics
import datasets
import torch
from powerLaw import PowerLaw, dist
from model import GGLR, GPR
from batches import get_GPR_batch
import time
import random
import multiprocessing as mp
import torch.cuda as T
import pickle
import validation as val
def pickle_load(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

def pickle_save(data, name):
    with open(name, 'wb') as f:
	    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
class Args:
    def __init__(self):
        self.lr = 0.001# learning rate
        self.lamda = 0.02 # model regularization rate
        self.gglr_control = 0.1
        self.scaling = 10
        self.batch_size = 4096 # batch size for training
        self.epochs = 50 # training epoches
        self.topk = 50 # compute metrics@top_k
        self.factor_num = 32 # predictive factors numbers in the model
        self.num_ng = 1 # sample negative items for training


def train(train_matrix, test_positive, test_negative, val_positive, val_negative, dataset):
    now = datetime.now()
    model_directory = "./model/"+now.strftime('%Y-%m-%d %H_%M_%S')+"GPR"
    result_directory = "./result/"+now.strftime('%Y-%m-%d %H_%M_%S')+"GPR"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    max_recall = 0.0
    k_list=[5, 10, 15, 20, 25, 30]
    args = Args()
    num_users = dataset.user_num
    num_items = dataset.poi_num
    dist_mat = distance_mat(num_items, G.poi_coos)
    pickle_save(dist_mat,"dist_mat_Tokyo.pkl")
    dist_mat = pickle_load("dist_mat_Tokyo.pkl")
    dist_mat = torch.tensor(dist_mat,dtype=torch.float32).to(DEVICE)
    model = GPR(num_users, num_items, args.factor_num, 2, dataset.POI_POI_Graph,dist_mat, dataset.user_POI_Graph).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lamda)
    for e in range(args.epochs):
        model.train()
        train_loss = 0.0
        start_time = int(time.time())

        idx = list(range(num_users))
        
        random.shuffle(idx)
        user_id, train_positives, train_negatives = get_GPR_batch(train_matrix,test_negative,num_items,idx,args.num_ng)
        batches = []
        nnn = int(len(user_id)/100)
        for i in range(99):
            batches.append((nnn*(i),nnn*(i+1)))
        
        batches.append((nnn*99,len(user_id)))
        for idx_range in batches:
            optimizer.zero_grad() 
            
            
            rating_ul, rating_ul_prime, e_ij_hat = model(user_id[idx_range[0]:idx_range[1]], train_positives[idx_range[0]:idx_range[1]], train_negatives[idx_range[0]:idx_range[1]])
            loss = model.loss_function(rating_ul, rating_ul_prime, e_ij_hat )
            loss.backward()

            train_loss += loss.item()
            
            optimizer.step() 
        end_time = int(time.time())
        print("Train Epoch: {}; time: {} sec; loss: {:.4f}".format(e+1, end_time-start_time,train_loss))
        
        model.eval() 
        with torch.no_grad():
            start_time = int(time.time())
            val_precision, val_recall, val_hit = val.GeoIE_validation(model,args,num_users,val_positive,val_negative,train_matrix,True,[10],dist_mat)
            end_time = int(time.time())
            print("eval time: {} sec".format(end_time-start_time))
            if(max_recall < val_recall[0]):
                max_recall = val_recall[0]
                torch.save(model, model_directory+"/model")
                precision, recall, hit = val.GeoIE_validation(model,args,num_users,test_positive,test_negative,train_matrix,False,k_list,dist_mat)
                f=open(result_directory+"/results.txt","w")
                f.write("epoch:{}\n".format(e))
                f.write("@k: " + str(k_list)+"\n")
                f.write("prec:" + str(precision)+"\n")
                f.write("recall:" + str(recall)+"\n")
                f.write("hit:" + str(hit)+"\n")
                f.close()
            end_time = int(time.time())
            print("eval time: {} sec".format(end_time-start_time))
def distance_mat(poi_num,poi_coos):
    dist_mat = np.zeros([poi_num,poi_num])
    for i in range(poi_num):
        for j in range(poi_num):
            dist_mat[i][j] = min(max(0.01,dist(poi_coos[i], poi_coos[j])), 100)
    
    return dist_mat

def create_weighted_adjacency_matrix(edge_list, num_nodes):
    # Initialize the adjacency matrix with zeros
    adjacency_matrix = [[0] * num_nodes for _ in range(num_nodes)]

    # Fill in the matrix based on the edge list
    for edge in edge_list:
        start, end = edge
        if int(start) != int(end):
            adjacency_matrix[int(start)][int(end)] += 1  # 중복된 엣지는 중복된 만큼 값을 추가

    return adjacency_matrix

def create_poi_sequence(user_data):
    poi_sequence = []
    sorted_data = user_data.sort_values('timestamp')
    for i in range(len(sorted_data) - 1):
        poi_tuple = (sorted_data.iloc[i]['poi_id'], sorted_data.iloc[i + 1]['poi_id'])
        poi_sequence.append(poi_tuple)
    return poi_sequence

def main():
    print("data loading")
    # dataset_ = datasets.Dataset(15359,14586,"./data/Yelp/")
    # train_matrix, test_positive, test_negative, val_positive, val_negative, place_coords = dataset_.generate_data(0)
    # pickle_save((train_matrix, test_positive, test_negative, val_positive, val_negative, place_coords,dataset_),"dataset_Yelp.pkl")
    train_matrix, test_positive, test_negative, val_positive, val_negative, place_coords, dataset_ = pickle_load("dataset_Tokyo.pkl")
    print("train data generated")
    
    G.fit_distance_distribution(train_matrix, place_coords)
    
    print("train start")
    train(train_matrix, test_positive, test_negative, val_positive, val_negative, dataset_)

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    G = PowerLaw()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DEVICE = 'cpu'
    main()
