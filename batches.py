import time
import random
import numpy as np
import torch
import math
def dist(loc1, loc2):
    lat1, long1 = loc1[0], loc1[1]
    lat2, long2 = loc2[0], loc2[1]
    if abs(lat1 - lat2) < 1e-6 and abs(long1 - long2) < 1e-6:
        return 0.0
    degrees_to_radians = math.pi/180.0
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
           math.cos(phi1)*math.cos(phi2))
    arc = math.acos( cos )
    earth_radius = 6371
    return arc * earth_radius

def get_GeoIE_batch(train_matrix,test_negative, num_poi, uid, negative_num, dist_mat):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    item_list = np.arange(num_poi).tolist()

    positives = train_matrix.getrow(uid).indices.tolist()
    ttt = train_matrix.getrow(uid).data
    random.shuffle(positives)
    histories = np.array([positives]).repeat(len(positives)*(negative_num+1),axis=0)

   

    negative = list(set(item_list)-set(positives) - set(test_negative[uid]))
    random.shuffle(negative)

    negative = negative[:len(positives)*negative_num]
    negatives = np.array(negative).reshape([-1,negative_num])

    a= np.array(positives).reshape(-1,1)
    data = np.concatenate((a, negatives),axis=-1)
    data = data.reshape(-1)
    distances = []
    for t in data:
        temp = []
        for hi in positives:
            temp.append(dist_mat[t][hi])
        distances.append(temp)

    positive_label = np.array([1]).repeat(len(positives)).reshape(-1,1)
    negative_label = np.array([0]).repeat(len(positives)*negative_num).reshape(-1,negative_num)
    labels = np.concatenate((positive_label,negative_label),axis=-1).reshape(-1)

    user_history = torch.LongTensor(histories).to(DEVICE)
    train_data = torch.LongTensor(data).to(DEVICE)
    train_label = torch.tensor(labels,dtype=torch.float32).to(DEVICE)
    user_id = torch.LongTensor(np.array([uid]).repeat(len(train_data))).to(DEVICE)
    freq = np.array(train_matrix.getrow(uid).data).repeat((negative_num+1),axis=0)
    freq = torch.LongTensor(freq).reshape(-1,1).to(DEVICE)
    distances = torch.tensor(distances,dtype=torch.float32).to(DEVICE)

    return user_id, user_history, train_data, train_label, freq, distances

def get_GeoIE_batch_test(train_matrix, test_positive, test_negative, uid, dist_mat):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = []
    history = train_matrix.getrow(uid).indices.tolist()
    negative = test_negative[uid]
    positive = test_positive[uid]
    histories = np.array([history]).repeat(len(positive)+len(negative),axis=0)

    data = np.concatenate((negative,positive))

    distances = []
    for t in data:
        temp = []
        for hi in history:
            temp.append(dist_mat[t][hi])
        distances.append(temp)

    positive_label = np.array([1]).repeat(len(positive))
    negative_label = np.array([0]).repeat(len(negative))
    labels = np.concatenate((negative_label,positive_label))

    user_history = torch.LongTensor(histories).to(DEVICE)
    train_data = torch.LongTensor(data).to(DEVICE)
    train_label = torch.tensor(labels, dtype=torch.float32).to(DEVICE)
    user_id = torch.LongTensor(np.array([uid]).repeat(len(train_data))).to(DEVICE)
    freq = np.ones([len(positive)+len(negative)])
    freq = torch.LongTensor(freq).reshape(-1,1).to(DEVICE)
    distances = torch.tensor(distances,dtype=torch.float32).to(DEVICE)

    return user_id, user_history, train_data, train_label, freq, distances
