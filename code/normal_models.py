from numpy.random import seed
seed(10132017)
from tensorflow import set_random_seed
set_random_seed(18071991)
from keras.models import Sequential
from sklearn.svm import SVR
from keras.layers.core import Flatten, Dense, Dropout, Activation
from sklearn.ensemble import RandomForestRegressor


dropout_rate = 0.2
FINAL_DIM = 900
def build_SVR(kernel_func='rbf', C_value=1.0):
    return SVR(kernel=kernel_func, C=C_value)

def build_RF(num_estimator):
    return RandomForestRegressor(n_estimators=num_estimator)

def build_NN(data_dim, hidDim=[140,280]):
    model = Sequential()
    model.add(Dense(hidDim[0],activation='tanh', input_shape=(data_dim, )))
    model.add(Dense(hidDim[1], activation='tanh'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(FINAL_DIM,activation='tanh'))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")   
    return model
