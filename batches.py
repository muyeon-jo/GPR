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

def get_GPR_batch(train_matrix,test_negative, num_poi, uids, negative_num):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_positives = []
    train_negatives = []
    user_id = []
    for uid in uids:
        item_list = np.arange(num_poi).tolist()

        positives = train_matrix.getrow(uid).indices.tolist()
        random.shuffle(positives)
        
        user_id.extend(np.array([uid]).repeat(len(positives)).reshape(-1,1).tolist())
        # for ui in range(len(positives)):
        #     user_id.append(uid)

        negative = list(set(item_list)-set(positives) - set(test_negative[uid]))
        random.shuffle(negative)

        negative = negative[:len(positives)*negative_num]
        # negatives = np.array(negative).reshape([-1,negative_num])

        # a= np.array(positives).reshape(-1,1)
        train_positives.extend(positives)
        # for po in a:
        #     train_positives.append(po.item())
        train_negatives.extend(negative)
        # for ne in negatives:
        #     train_negatives.append(ne.item())
    
    train_positives = np.array(train_positives).reshape(-1,1)
    train_negatives = np.array(train_negatives).reshape(-1,negative_num)
    user_id = np.array(user_id).reshape(-1,1)

    train_positives = torch.LongTensor(train_positives).squeeze().to(DEVICE)
    train_negatives = torch.LongTensor(train_negatives).squeeze().to(DEVICE)
    user_id = torch.LongTensor(user_id).squeeze().to(DEVICE)

    return user_id, train_positives, train_negatives

def get_GPR_batch_test(train_matrix, test_positive, test_negative, uid):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datas = []
    negative = test_negative[uid]
    positive = test_positive[uid]
    user_id = []

    user_id.extend(np.array([uid]).repeat(len(positive) + len(negative)).tolist())
    # for ui in range(len(positive) + len(negative)):
    #     user_id.append(uid)

    datas.extend(negative)
    datas.extend(positive)
    # for i in negative:
    #     datas.append(i)

    # for i in positive:
    #     datas.append(i)

    datas = torch.LongTensor(datas).to(DEVICE)
    user_id = torch.LongTensor(user_id).to(DEVICE)

    return user_id, datas
