import numpy as np
import cPickle
import scipy.io as sio


def load_data(normal_stat=False):
    if normal_stat:
        filepath = "../data/data_normal.p"
    else:
        filepath = "../data/data_seq.p" 
    x = cPickle.load(open(filepath,"rb"))
    return [x[0], x[1], x[2], x[3]]  # retrun train_x, train_y, test_x, test_y

if __name__ == '__main__':
    t = load_data(True)
    print t[2].shape
    print t[3].shape
    t = load_data()
    print t[0].shape
    
