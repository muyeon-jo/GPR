from sklearn.model_selection import train_test_split
import scipy.sparse as sparse
import numpy as np
import random
import csv
from haversine import haversine
def train_test_split_with_time(place_list, freq_list, time_list, test_size):
    li = []
    for i in range(len(place_list)):
        li.append((place_list[i],time_list[i], freq_list[i]))
    
    li.sort(key=lambda x:-x[1])
    test = li[:int(len(li)*test_size)]
    train = li[int(len(li)*test_size):]
    random.shuffle(train)

    test_place = []
    test_freq = []
    for i in test:
        test_place.append(i[0])
        test_freq.append(i[2])

    train_place=[]
    train_freq=[]
    for i in train:
        train_place.append(i[0])
        train_freq.append(i[2])

    return train_place, test_place, train_freq, test_freq
def train_test_val_split_with_time(place_list, freq_list, time_list, test_size, val_size):
    li = []
    for i in range(len(place_list)):
        li.append((place_list[i], time_list[i], freq_list[i]))
    li.sort(key=lambda x:-x[1])
    test = li[:int(len(li)*test_size)]
    train_ = li[int(len(li)*test_size):]

    val_num = int(len(li)*val_size)
    if val_num == 0:
        val_num=1
    val = train_[:val_num]
    train = train_[val_num:]

    # random.shuffle(train)
    test_place = []
    test_freq = []
    for i in test:
        test_place.append(i[0])
        test_freq.append(i[2])

    train_place=[]
    train_freq=[]
    for i in train:
        train_place.append(i[0])
        train_freq.append(i[2])

    val_place = []
    val_freq = []

    for i in val:
        val_place.append(i[0])
        val_freq.append(i[2])
    return train_place, test_place, val_place, train_freq, test_freq, val_freq

def train_test_val_split(place_list, freq_list, test_size, val_size):
    li = []
    for i in range(len(place_list)):
        li.append((place_list[i], freq_list[i]))
    
    random.shuffle(li)
    test = li[:int(len(li)*test_size)]
    train_ = li[int(len(li)*test_size):]
    val_num = int(len(li)*val_size)
    if val_num == 0:
        val_num=1
    val = train_[:val_num]
    train = train_[val_num:]

    test_place = []
    test_freq = []
    for i in test:
        test_place.append(i[0])
        test_freq.append(i[1])

    train_place=[]
    train_freq=[]
    for i in train:
        train_place.append(i[0])
        train_freq.append(i[1])

    val_place = []
    val_freq = []

    for i in val:
        val_place.append(i[0])
        val_freq.append(i[1])
    return train_place, test_place, val_place, train_freq, test_freq, val_freq

class Yelp(object):
    def __init__(self):
        self.user_num = 15359
        # self.user_num = 10000
        self.poi_num = 14586
        self.directory_path = './data/Yelp/'
        self.poi_file = 'Yelp_poi_coos.txt'
        self.checkin_file = 'Yelp_checkins.txt'
        # self.checkin_file = 'sample.txt'
    def read_raw_data(self):
        all_data = open(self.directory_path + self.checkin_file, 'r').readlines()
        sparse_raw_matrix = sparse.dok_matrix((self.user_num, self.poi_num))
        sparse_raw_time_matrix= sparse.dok_matrix((self.user_num, self.poi_num))
        for eachline in all_data:
            uid, lid, time = eachline.strip().split()
            uid, lid, time = int(uid), int(lid), float(time)
            sparse_raw_matrix[uid, lid] = sparse_raw_matrix[uid, lid] + 1
            if sparse_raw_time_matrix[uid,lid] ==0 or sparse_raw_time_matrix[uid,lid] >= time:
                sparse_raw_time_matrix[uid, lid] = time
        return sparse_raw_matrix.tocsr(), sparse_raw_time_matrix.tocsr()

    def split_data(self, raw_matrix, time_matrix, random_seed=0):
        test_size = 0.2
        val_size = 0.1
        train_matrix = sparse.dok_matrix((self.user_num, self.poi_num))
        
        test_positive = []
        test_negative = []
        pois = np.arange(self.poi_num)
        for user_id in range(self.user_num):
            place_list = raw_matrix.getrow(user_id).indices
            freq_list = raw_matrix.getrow(user_id).data
            time_list = time_matrix.getrow(user_id).data
            train_place, test_place, train_freq, test_freq = train_test_split(place_list, freq_list, test_size=test_size, random_state=random_seed)
            # train_place, test_place, train_freq, test_freq = train_test_split_with_time(place_list, freq_list, time_list, test_size)

            for i in range(len(train_place)):
                train_matrix[user_id, train_place[i]] = train_freq[i]
            test_positive.append(test_place.tolist())

            negative = list(set(pois) - set(raw_matrix.getrow(user_id).indices))
            random.shuffle(negative)
            # train_negative.append(negative[int(len(negative)*test_size):])
            test_negative.append(negative[:int(len(negative)*test_size)])
        # sparse.save_npz('./data/Foursquare/train_matrix.npz', train_matrix)

        return train_matrix.tocsr(), test_positive, test_negative

    def read_poi_coos(self):
        poi_coos = {}
        poi_data = open(self.directory_path + self.poi_file, 'r').readlines()
        for eachline in poi_data:
            lid, lat, lng = eachline.strip().split()
            lid, lat, lng = int(lid), float(lat), float(lng)
            poi_coos[lid] = (lat, lng)

        place_coords = []
        for k, v in poi_coos.items():
            place_coords.append([v[0], v[1]])

        return place_coords

    def generate_data(self, random_seed=0):
        raw_matrix, time_matrix = self.read_raw_data()
        train_matrix, test_positive, test_negative = self.split_data(raw_matrix, time_matrix, random_seed)
        place_coords =self.read_poi_coos()
        return train_matrix,  test_positive, test_negative, place_coords

class Foursquare(object):
    def __init__(self):
        self.user_num = 24941
        self.poi_num = 28593
        self.directory_path = './data/Foursquare/'
        self.checkin_file = 'Foursquare_checkins.txt'
        self.poi_file = 'Foursquare_poi_coos.txt'
    def read_raw_data(self):
        all_data = open(self.directory_path + self.checkin_file, 'r').readlines()
        sparse_raw_matrix = sparse.dok_matrix((self.user_num, self.poi_num))
        sparse_raw_time_matrix= sparse.dok_matrix((self.user_num, self.poi_num))
        for eachline in all_data:
            uid, lid, time = eachline.strip().split()
            uid, lid, time = int(uid), int(lid), float(time)
            sparse_raw_matrix[uid, lid] = sparse_raw_matrix[uid, lid] + 1
            if sparse_raw_time_matrix[uid,lid] ==0 or sparse_raw_time_matrix[uid,lid] >= time:
                sparse_raw_time_matrix[uid, lid] = time
        return sparse_raw_matrix.tocsr(), sparse_raw_time_matrix.tocsr()

    def split_data(self, raw_matrix, time_matrix, random_seed=0):
        test_size = 0.2
        train_matrix = sparse.dok_matrix((self.user_num, self.poi_num))
        
        test_positive = []
        test_negative = []
        pois = np.arange(self.poi_num)
        for user_id in range(self.user_num):
            place_list = raw_matrix.getrow(user_id).indices
            freq_list = raw_matrix.getrow(user_id).data
            time_list = time_matrix.getrow(user_id).data
            train_place, test_place, train_freq, test_freq = train_test_split(place_list, freq_list, test_size=test_size, random_state=random_seed)
            # train_place, test_place, train_freq, test_freq = train_test_split_with_time(place_list, freq_list, time_list, test_size)

            for i in range(len(train_place)):
                train_matrix[user_id, train_place[i]] = train_freq[i]
            test_positive.append(test_place.tolist())

            negative = list(set(pois) - set(raw_matrix.getrow(user_id).indices))
            random.shuffle(negative)
            # train_negative.append(negative[int(len(negative)*test_size):])
            test_negative.append(negative[:int(len(negative)*test_size)])
        # sparse.save_npz('./data/Foursquare/train_matrix.npz', train_matrix)

        return train_matrix.tocsr(), test_positive, test_negative

    def read_poi_coos(self):
        poi_coos = {}
        poi_data = open(self.directory_path + self.poi_file, 'r').readlines()
        for eachline in poi_data:
            lid, lat, lng = eachline.strip().split()
            lid, lat, lng = int(lid), float(lat), float(lng)
            poi_coos[lid] = (lat, lng)

        place_coords = []
        for k, v in poi_coos.items():
            place_coords.append([v[0], v[1]])

        return place_coords

    def generate_data(self, random_seed=0):
        raw_matrix, time_matrix = self.read_raw_data()
        train_matrix, test_positive, test_negative = self.split_data(raw_matrix, time_matrix, random_seed)
        place_coords =self.read_poi_coos()
        return train_matrix,  test_positive, test_negative, place_coords

class Dataset(object):
    def __init__(self,user_num,_poi_num,directory_path):
        self.user_num = user_num
        self.poi_num = _poi_num
        self.directory_path = directory_path
        self.checkin_file = 'checkins.txt'
        self.poi_file = 'poi_coos.txt'
    def read_raw_data(self):
        all_data = open(self.directory_path + self.checkin_file, 'r').readlines()
        sparse_raw_matrix = sparse.dok_matrix((self.user_num, self.poi_num))
        sparse_raw_time_matrix= sparse.dok_matrix((self.user_num, self.poi_num))
        for eachline in all_data:
            uid, lid, time = eachline.strip().split()
            uid, lid, time = int(uid), int(lid), float(time)
            sparse_raw_matrix[uid, lid] = sparse_raw_matrix[uid, lid] + 1
            if sparse_raw_time_matrix[uid,lid] < time:
                sparse_raw_time_matrix[uid, lid] = time
        return sparse_raw_matrix.tocsr(), sparse_raw_time_matrix.tocsr()

    def split_data(self, raw_matrix, time_matrix, random_seed=0):
        test_size = 0.2
        val_size = 0.1
        train_matrix = sparse.dok_matrix((self.user_num, self.poi_num))
        
        val_positive = []
        val_negative = []
        test_positive = []
        test_negative = []
        self.POI_POI_Graph = np.zeros([self.poi_num,self.poi_num])
        self.user_POI_Graph = np.zeros([self.user_num,self.poi_num])

        pois = set(range(self.poi_num))
        for user_id in range(self.user_num):
            place_list = raw_matrix.getrow(user_id).indices
            freq_list = raw_matrix.getrow(user_id).data
            time_list = time_matrix.getrow(user_id).data
            train_place, test_place, val_place, train_freq, test_freq, val_freq = train_test_val_split_with_time(place_list, freq_list, time_list, test_size, val_size)
            # train_place, test_place, train_freq, test_freq = train_test_split_with_time(place_list, freq_list, time_list, test_size)
            # train_place, test_place, val_place, train_freq, test_freq, val_freq = train_test_val_split(place_list, freq_list, test_size, val_size)

            for i in range(len(train_place)):
                train_matrix[user_id, train_place[i]] = train_freq[i]

                self.user_POI_Graph[user_id][train_place[i]] = 1

                if i <len(train_place)-1:
                    self.POI_POI_Graph[train_place[i]][train_place[i+1]] +=1
                
            test_positive.append(test_place)
            val_positive.append(val_place)

            negative = list(pois - set(raw_matrix.getrow(user_id).indices))
            random.shuffle(negative)
            # train_negative.append(negative[int(len(negative)*test_size):])
            ln = len(negative[int(len(negative)*test_size):])
            test_ln = len(negative[:int(len(negative)*test_size)])
            test_negative.append(negative[:test_ln])
            val_negative.append(negative[test_ln:test_ln + int(ln*val_size)])
        # sparse.save_npz('./data/Foursquare/train_matrix.npz', train_matrix)

        return train_matrix.tocsr(), test_positive, test_negative,val_positive,val_negative

    def read_poi_coos(self):
        poi_coos = {}
        poi_data = open(self.directory_path + self.poi_file, 'r').readlines()
        for eachline in poi_data:
            lid, lat, lng = eachline.strip().split()
            lid, lat, lng = int(lid), float(lat), float(lng)
            poi_coos[lid] = (lat, lng)

        place_coords = []
        for k, v in poi_coos.items():
            place_coords.append([v[0], v[1]])
        self.place_coos = place_coords
        return place_coords

    def generate_data(self, random_seed=0):
        raw_matrix, time_matrix = self.read_raw_data()
        train_matrix, test_positive, test_negative, val_positive, val_negative = self.split_data(raw_matrix, time_matrix, random_seed)
        place_coords =self.read_poi_coos()
        return train_matrix,  test_positive, test_negative, val_positive, val_negative, place_coords

if __name__ == '__main__':
    # train_matrix, test_positive, test_negative, val_positive, val_negative, place_coords= Dataset(9902,6427,"./data/philadelphia_downtown/").generate_data()
    train_matrix, test_positive, test_negative, val_positive, val_negative, place_coords= Dataset(15359,14586,"./data/Yelp/").generate_data()
    print(train_matrix.shape, len(test_positive), len(place_coords))


