import sys
import numpy as np
import _pickle as cPickle
import argparse
from scipy.io import loadmat 


def load_data(dataset, path, ratio =0.9, seed =0):
    if dataset == 'freyface':
        data_train, data_test = load_data_freyface(path, ratio, seed)
    return data_train, data_test

def load_data_freyface(path, ratio= 0.9, seed =0):
    print("loading data")
    f= open(path+ '/freyface/freyfaces.pkl' , 'rb')
    data = cPickle.load(f, encoding='latin1')
    data = np.array(data, dtype= 'f')
    f.close()
    np.random.seed(seed)
    np.random.shuffle(data)
    num_train = int(ratio * data.shape[0])
    data_train = data[:num_train]
    data_test = data[num_train:]      
    return data_train, data_test   

def get_next_batch(Train_set, batch_size):
    batch_list = np.random.randint(Train_set.shape[0], size=batch_size)
    next_batch = Train_set[batch_list]
    return next_batch