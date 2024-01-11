import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.optim as optim
import csv
import random
import numpy as np
import pickle

def read_files(file_path_coos, file_path_checkin):
    # 파일에서 POI 정보 읽어오기
    POI_info = {}
    with open(file_path_coos, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for line in reader:
            # line[0]: business_id, line[1]: latitude, line[2]: longitude
            poi_id = line[0]
            lat_long = [float(line[1]), float(line[2])]
            POI_info.setdefault(poi_id, lat_long)
            
    # Initialize empty lists
    user_id = []
    history_total = []
    # Read the file and process data
    with open(file_path_checkin, 'r') as file:
        lines = file.readlines()
    # Process each line in the file
    current_user = None
    user_history = []
    for line in lines:
        # Split the line into columns
        columns = line.strip().split('\t')
        # Extract user_id, poi_id, and timestamp
        user, poi, timestamp = map(int, columns)
        # Check if user has changed
        if current_user is None:
            current_user = user
        if user != current_user: # new user, update list
            user_id.append(current_user) 
            history_total.append(user_history)
            # Reset user-specific variables
            current_user = user
            user_history = []
        # Add poi_id to the user's history
        user_history.append(poi)

    # Append the last user's history to the lists
    if current_user is not None:
        user_id.append(current_user)
        history_total.append(user_history)

    label_rate = 0.6
    test_rate = 0.2
    history = []
    label = []
    test_data = []
    for pois in history_total:
        hist = pois[:int(len(pois) * label_rate)]
        tg = pois[int(len(pois) * label_rate):int(len(pois) * (label_rate+test_rate))]
        t = pois[int(len(pois) * (label_rate+test_rate)):]
        history.append(hist)
        label.append(tg)
        test_data.append(t)
        
    return test_data, label, history, POI_info

def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    r = 6371  # Radius of the Earth in kilometers. You can use 3959 for miles.

    # Calculate the distance
    distance = r * c
    return distance

def cal_distance(pid, hid, POI_info):
    result = []
    for id in hid:
        lat1, lon1 = map(float, POI_info[str(pid)])
        lat2, lon2 = map(float, POI_info[str(id)])
        d = haversine(lat1, lon1, lat2, lon2)
        result.append(d if d != 0 else 0.01)
    return result

def cal_all_poi_distance(poi_num, POI_info):
    result = []
    poi = [i for i in range(poi_num)]
    for i in poi:
        result.append(cal_distance(i, poi, POI_info))
        
    pkl_file_path = 'GEOIE\poi_distances.pkl'
    with open(pkl_file_path, 'wb') as pkl_file:
        pickle.dump(result, pkl_file)
    return result
    

def get_train_data(label, history, POI_info, ng_num):
    data = {}
    
    for user_id, lb in enumerate(label):
        targets = lb
        neg_samples = []
        count = []
        for target in lb:
            c = history[user_id].count(target) #+ lb.count(target)
            count.append(c)
            
        for _ in range(ng_num):
            neg = [] # target의 개수만큼 neg list를 ngnum개 만큼 생성
            while len(neg) < len(targets):
                n = random.randint(0, 14585)
                if n not in lb and n not in history[user_id]:
                    neg.append(n)
            neg_samples.append(neg)
            
            

        distances = [cal_distance(t, history[user_id], POI_info) for t in targets]
        neg_distances = [[cal_distance(n, history[user_id], POI_info) for n in neg] for neg in neg_samples]

        # User별로 데이터 집계
        data[user_id] = {
            'user_id' : user_id,
            'targets': targets,
            'neg_samples': neg_samples,
            'distances': distances,
            'neg_distances': neg_distances,
            'history' : history[user_id],
            'count' : count
        }
    return data