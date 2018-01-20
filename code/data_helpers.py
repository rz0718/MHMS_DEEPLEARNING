import numpy as np
import cPickle
import scipy.io as sio
from sklearn import preprocessing
def transform_data(x, mode='equal'):
    # x is num_sample * len_time * dim
    num_sample, timestep, data_dim = x.shape[0], x.shape[1],x.shape[2]
    x = np.mean(x,axis=1)
    return np.squeeze(x)


def normal_data(x):
    #preprocess data
    x_transform = np.swapaxes(x,1,2)
    num_sample,timestep, data_dim = x_transform.shape[0], x_transform.shape[1],x_transform.shape[2]
    x_transform = x.reshape(num_sample*timestep, data_dim)
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(x_transform)
    x_transform = scaler.transform(x_transform) 
    x_transform = x_transform.reshape(num_sample,timestep,data_dim)
    return  x_transform


def load_data(num_test=1, normal_stat=False):
    fold_names = ['c1', 'c4', 'c6']
    x = cPickle.load(open("/data/cwru/data/%s_20.p" %fold_names[num_test],"rb"))
    test, test_y= x[0][3:303], x[1][3:303]
    fold_names.pop(num_test)
    x = cPickle.load(open("/data/cwru/data/%s_20.p" %fold_names[0],"rb"))
    train, train_y = x[0][3:303], x[1][3:303]
    x = cPickle.load(open("/data/cwru/data/%s_20.p" %fold_names[1],"rb"))
    train = np.concatenate([train,x[0][3:303]], axis=0)
    train_y = np.concatenate([train_y,x[1][3:303]], axis=0)
    len_train, timesteps, data_dim = train.shape[0], train.shape[1], train.shape[2]
    len_test = test.shape[0]
    x_train1 = train.reshape((len_train*timesteps, data_dim))
    x_test1 = test.reshape((len_test*timesteps, data_dim))
    data_whole = np.concatenate((x_train1,x_test1),axis=0)
    if np.isnan(data_whole).any():
        idx = np.argwhere(np.isnan(data_whole))
        for idice in idx:
            data_whole[idice[0],idice[1]] = data_whole[idice[0]-1,idice[1]]
    data_whole = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(data_whole)
    x_train1 = data_whole[:len_train*timesteps]  
    x_train = x_train1.reshape((len_train, timesteps, data_dim))
    x_test1 = data_whole[len_train*timesteps:]
    x_test = x_test1.reshape((len_test, timesteps, data_dim))
    with open("/data/cwru/data/data_seq.p",'wb') as fp:
        cPickle.dump([x_train, train_y, x_test, test_y], fp)
    if normal_stat:
        x_train = transform_data(x_train)
        x_test = transform_data(x_test)
    with open("/data/cwru/data/data_normal.p",'wb') as fp:
        cPickle.dump([x_train, train_y, x_test, test_y], fp)
    return [x_train, train_y, x_test, test_y]


if __name__ == '__main__':
    t = load_data(1, True)
    
