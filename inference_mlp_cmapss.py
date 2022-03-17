

## Import libraries in python
import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import importlib
from scipy.stats import randint, expon, uniform

import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt

import scipy.stats as stats
from scipy.stats import ttest_1samp
import tensorflow as tf
print(tf.__version__)
# import keras.backend as K
import tensorflow.keras.backend as K
from tensorflow.keras import backend
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Embedding
from tensorflow.keras.layers import BatchNormalization, Activation, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

seed = 0

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

piecewise_lin_ref = 125

current_dir = os.path.dirname(os.path.abspath(__file__))
data_filedir = os.path.join(current_dir, 'cmapss')

train_FD001_path = data_filedir +'/train_FD001.csv'
test_FD001_path = data_filedir +'/test_FD001.csv'
RUL_FD001_path = data_filedir+'/RUL_FD001.txt'
FD001_path = [train_FD001_path, test_FD001_path, RUL_FD001_path]

### Data load (use only FD001 first)
## Assign columns name
cols = ['unit_nr', 'cycles', 'os_1', 'os_2', 'os_3']
cols += ['sensor_{0:02d}'.format(s + 1) for s in range(26)]
col_rul = ['RUL_truth']
print (cols)
## Define function for data reading
def load_data (data_path_list, columns_ts, columns_rul):
  train_FD = pd.read_csv(data_path_list[0], sep= ' ', header=None, 
                            names=columns_ts, index_col=False)
  test_FD = pd.read_csv(data_path_list[1], sep= ' ', header=None, 
                            names=columns_ts, index_col=False)
  RUL_FD = pd.read_csv(data_path_list[2], sep= ' ', header=None, 
                            names=columns_rul, index_col=False)
  
  return train_FD, test_FD, RUL_FD


## Read csv file to pandas dataframe
train_FD, test_FD, RUL_FD = load_data (FD001_path, cols, col_rul)

## Calculate RUL and append to train data
# get the time of the last available measurement for each unit
mapper = {}
for unit_nr in train_FD['unit_nr'].unique():
    mapper[unit_nr] = train_FD['cycles'].loc[train_FD['unit_nr'] == unit_nr].max()
    
# calculate RUL = time.max() - time_now for each unit
train_FD['RUL'] = train_FD['unit_nr'].apply(lambda nr: mapper[nr]) - train_FD['cycles']
# piecewise linear for RUL labels
train_FD['RUL'].loc[(train_FD['RUL'] > piecewise_lin_ref)] = piecewise_lin_ref

## Excluse columns which only have NaN as value
# nan_cols = ['sensor_{0:02d}'.format(s + 22) for s in range(5)]
cols_nan = train_FD.columns[train_FD.isna().any()].tolist()
# print('Columns with all nan: \n' + str(cols_nan) + '\n')
cols_const = [ col for col in train_FD.columns if len(train_FD[col].unique()) <= 2 ]
# print('Columns with all const values: \n' + str(cols_const) + '\n')

## Drop exclusive columns
# train_FD = train_FD.drop(columns=cols_const + cols_nan)
# test_FD = test_FD.drop(columns=cols_const + cols_nan)

train_FD = train_FD.drop(columns=cols_const + cols_nan + ['sensor_01','sensor_05','sensor_06',
                                                          'sensor_10','sensor_16','sensor_18','sensor_19'])

test_FD = test_FD.drop(columns=cols_const + cols_nan + ['sensor_01','sensor_05','sensor_06',
                                                          'sensor_10','sensor_16','sensor_18','sensor_19'])


## Check loaded data
pd.set_option('display.max_rows', 500)
print (train_FD.head(1000))
# print (test_FD)
# print (RUL_FD)


### function to reshape features into (samples, time steps, features)
def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    # for one id I put all the rows in a single matrix
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    # Iterate over two lists in parallel.
    # For example id1 have 192 rows and sequence_length is equal to 50
    # so zip iterate over two following list of numbers (0,142),(50,192)
    # 0 50 -> from row 0 to row 50
    # 1 51 -> from row 1 to row 51
    # 2 52 -> from row 2 to row 52
    # ...
    # 142 192 -> from row 142 to 192
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]


        
        
def gen_labels(id_df, seq_length, label):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    # For one id I put all the labels in a single matrix.
    # For example:
    # [[1]
    # [4]
    # [1]
    # [5]
    # [9]
    # ...
    # [200]]
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    # I have to remove the first seq_length labels
    # because for one id the first sequence of seq_length size have as target
    # the last label (the previus ones are discarded).
    # All the next id's sequences will have associated step by step one label as target.
    return data_matrix[seq_length:num_elements, :]
        

### Normalize sensor measurement data 
def df_preprocessing(df, train=True):
    if train==True:
        cols_normalize = df.columns.difference(['unit_nr', 'cycles', 'os_1', 'os_2', 'RUL'])
    else : 
        cols_normalize = df.columns.difference(['unit_nr', 'cycles', 'os_1', 'os_2'])
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_df = pd.DataFrame(min_max_scaler.fit_transform(df[cols_normalize]),
                                 columns=cols_normalize,
                                 index=df.index)
    join_df = df[df.columns.difference(cols_normalize)].join(norm_df)
    df = join_df.reindex(columns=df.columns)
    if train==True:
        pass
    else :
        df = df.reset_index(drop=True)
    
    return df

### Generate training set / Test set / Labels (ground truth of test set)

## Set parameters (sequence and window length etc..)
sequence_length = 30
n_channel = 1
stride = 1
window_length = 10
n_window = int((sequence_length - window_length)/(stride) + 1)
print ("n_window: ",n_window)

n_filters = 5
strides_len = 1
input_features = n_channel
kernel_size = 3
n_conv_layer = 3
n_outputs = 1

model_path = data_filedir +'/model_s_%s_n_%s_l_%s.h5' %(sequence_length, window_length, n_window )
bs = 512




## preprocessing(normailization for the neural networks)
min_max_scaler = preprocessing.MinMaxScaler()
# for the training set
# train_FD['cycles_norm'] = train_FD['cycles']
cols_normalize = train_FD.columns.difference(['unit_nr', 'cycles', 'os_1', 'os_2', 'RUL'])

norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_FD[cols_normalize]),
                             columns=cols_normalize,
                             index=train_FD.index)
join_df = train_FD[train_FD.columns.difference(cols_normalize)].join(norm_train_df)
train_FD_norm = join_df.reindex(columns=train_FD.columns)


# for the test set
# test_FD['cycles_norm'] = test_FD['cycles']
cols_normalize_test = test_FD.columns.difference(['unit_nr', 'cycles','os_1', 'os_2' ])
# print ("cols_normalize_test", cols_normalize_test)
norm_test_df = pd.DataFrame(min_max_scaler.transform(test_FD[cols_normalize_test]), columns=cols_normalize_test,index=test_FD.index)
test_join_df = test_FD[test_FD.columns.difference(cols_normalize_test)].join(norm_test_df)
test_FD = test_join_df.reindex(columns=test_FD.columns)
test_FD = test_FD.reset_index(drop=True)
test_FD_norm = test_FD

## or use function
# train_FD_norm = df_preprocessing(train_FD)
# test_FD_norm = df_preprocessing(test_FD, train=False)

print (train_FD_norm)
print (test_FD_norm)

# pick the feature columns
sequence_cols_train =  train_FD_norm.columns.difference(['unit_nr', 'cycles' , 'os_1', 'os_2',  'RUL'])
sequence_cols_test =  test_FD_norm.columns.difference(['unit_nr', 'os_1', 'os_2', 'cycles'])
print("sequence_cols_train: ", sequence_cols_train)
print ("sequence_cols_test: ", sequence_cols_test)
                                                          
## generator for the sequences
# transform each id of the train dataset in a sequence
seq_gen = (list(gen_sequence(train_FD_norm[train_FD_norm['unit_nr'] == id], sequence_length, sequence_cols_train))
           for id in train_FD_norm['unit_nr'].unique())

# generate sequences and convert to numpy array
seq_array_train = np.concatenate(list(seq_gen)).astype(np.float32)
print("seq_array_train.shape", seq_array_train.shape) 


# seq_gen_test = (list(gen_sequence(test_FD001_norm[test_FD001_norm['unit_nr'] == id], sequence_length, sequence_cols_test))
#            for id in test_FD001_norm['unit_nr'].unique())

# # generate sequences and convert to numpy array
# seq_array_test = np.concatenate(list(seq_gen_test)).astype(np.float32)
# print("seq_array_test.shape", seq_array_test.shape) 



# for id in test_FD001_norm['unit_nr'].unique():
#     print (len(test_FD001_norm[test_FD001_norm['unit_nr'] == id]))


# #Test dataset
# seq_gen_test = (list(gen_sequence(test_FD001_norm[test_FD001_norm['unit_nr'] == id], sequence_length, sequence_cols_test))
#            for id in test_FD001_norm['unit_nr'].unique())

# # generate sequences and convert to numpy array
# seq_array_test = np.concatenate(list(seq_gen_test)).astype(np.float32)
# print("seq_array_test.shape", seq_array_test.shape) 


# generate labels
label_gen = [gen_labels(train_FD_norm[train_FD_norm['unit_nr'] == id], sequence_length, ['RUL'])
             for id in train_FD_norm['unit_nr'].unique()]

label_array_train = np.concatenate(label_gen).astype(np.float32)




# ## Data for first engine 
# for unit_nr in train_FD001_norm['unit_nr'].unique():
#     train_FD001_unit.append(train_FD001_norm[train_FD001_norm['unit_nr'] == unit_nr])


# unit0_sensor2 = train_FD001_unit[0]['sensor_02']
# print (unit0_sensor2)    
'''
Define the function for generating CNN braches(heads)

'''

def one_dcnn(input_array):

    cnn = Sequential(name='one_d_cnn')
    cnn.add(Conv1D(filters=8, kernel_size=12, padding='same', input_shape=(input_array.shape[1],input_array.shape[2])))
    # cnn.add(BatchNormalization())
    cnn.add(Activation('sigmoid'))
    cnn.add(AveragePooling1D(pool_size=2))
    cnn.add(Conv1D(filters=14, kernel_size=4, padding='same'))
    # cnn.add(BatchNormalization())
    cnn.add(Activation('sigmoid'))
    cnn.add(AveragePooling1D(pool_size=2))
    cnn.add(Flatten())
    cnn.add(Dense(50))
    cnn.add(Activation('sigmoid'))
    cnn.add(Dense(1))
    cnn.add(Activation("linear"))
    return cnn


def mlps(vec_len):
    '''

    '''
    # model = Sequential()
    # model.add(Dense(h1, activation='relu', input_shape=(vec_len,)))
    # model.add(Dense(h4, activation='relu'))
    # model.add(Dense(1))

    # model = Sequential()
    # model.add(Dense(30, activation='relu', input_shape=(vec_len,)))
    # # model.add(Dense(100, activation='relu'))
    # model.add(Dense(1))

    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(vec_len,)))
    model.add(Dense(1))

    # model = Sequential()
    # model.add(Dense(20, activation='relu', input_shape=(vec_len,)))
    # model.add(Dense(20, activation='relu'))
    # model.add(Dense(20, activation='relu'))
    # model.add(Dense(1))
    


    return model    


def r2_keras(y_true, y_pred):
    """Coefficient of Determination
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

train_direct_array = seq_array_train[:,29,:]
print ("train_direct_array.shape", train_direct_array.shape)

### Imigrate keras model to tensorflow 2.0 keras



# one_d_cnn_model = one_dcnn(seq_array_train)
# print(one_d_cnn_model.summary())




mlp_net = mlps(train_direct_array.shape[1])
print(mlp_net.summary())


# adm = optimizers.Adam(learning_rate=0.01)
# adm = optimizers.Adam(learning_rate=0.001)
rp = optimizers.RMSprop(learning_rate=0.001, rho=0.9)

# model.compile(optimizer=adm, loss='categorical_crossentropy', metrics=['acc'])


# cnnlstm.compile(loss='mean_squared_error', optimizer=adm, metrics=[rmse, 'mae', r2_keras])
#     cnnlstm.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=[rmse, 'mae', r2_keras,'cosine_proximity'])
# one_d_cnn_model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=[rmse, 'mae', r2_keras,'cosine_proximity'])

lr = 0.01
# amsgrad = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True, name='Adam')
mlp_net.compile(loss='mean_squared_error', optimizer=rp, metrics='mae')

print(mlp_net.summary())

# fit the network
history = mlp_net.fit(train_direct_array, label_array_train, epochs=30, batch_size=bs, validation_split=0.2, verbose=1, 
                      callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min'),
                                    ModelCheckpoint(model_path,monitor='val_loss', save_best_only=False, mode='min', verbose=1)]
                      )


#     history = cnnlstm.fit(train_FD001_sensor, label_array_train, epochs=60, batch_size=bs, verbose=1 )

# list all data in history
print(history.history.keys())
mlp_net.save(model_path)

'''
EVALUATE ON TEST DATA

'''

# We pick the last sequence for each id in the test data
seq_array_test_last = [test_FD[test_FD['unit_nr'] == id][sequence_cols_test].values[-sequence_length:]
                       for id in test_FD['unit_nr'].unique() if len(test_FD[test_FD['unit_nr'] == id]) >= sequence_length]
print (seq_array_test_last[0].shape)

seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
print("seq_array_test_last")
# print(seq_array_test_last)
print(seq_array_test_last.shape)



test_direct_array = seq_array_test_last[:,29,:]
print ("train_direct_array.shape", test_direct_array.shape)


# Similarly, we pick the labels
# print("y_mask")
y_mask = [len(test_FD_norm[test_FD_norm['unit_nr'] == id]) >= sequence_length for id in test_FD_norm['unit_nr'].unique()]
print ("y_mask", y_mask)
label_array_test_last = RUL_FD['RUL_truth'][y_mask].values

label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0], 1).astype(np.float32)
print (label_array_test_last)
print(label_array_test_last.shape)


estimator = load_model(model_path, custom_objects={'rmse':rmse, 'r2_keras': r2_keras})

# test metrics
# scores_test = estimator.evaluate(seq_array_test_last, label_array_test_last, verbose=2)
scores_test = estimator.evaluate(test_direct_array, label_array_test_last, verbose=2)

print ("estimator.metrics_names", estimator.metrics_names)
print ("scores_test", scores_test)
print('\nLOSS: {}'.format(scores_test[0]))
print('\nRMSE: {}'.format(scores_test[1]))
# y_pred_test = estimator.predict(seq_array_test_last)
import time
import math
start = time.time()

y_pred_test = estimator.predict(test_direct_array)

end = time.time()
eval_time = end - start
print("eval_time:", eval_time)

y_true_test = label_array_test_last


rms = sqrt(mean_squared_error(y_pred_test, y_true_test))



print(rms)
rms = round(rms, 2)

h_array = y_pred_test - y_true_test
# print (h_array)
s_array = np.zeros(len(h_array))
for j, h_j in enumerate(h_array):
    if h_j < 0:
        s_array[j] = math.exp(-(h_j / 13)) - 1
    else:
        s_array[j] = math.exp(h_j / 10) - 1

score = np.sum(s_array)
print ("score", score)


# pd.set_option('display.max_rows', 1000)
# test_print = pd.DataFrame()
# test_print['y_pred']  = y_pred_test.flatten()
# test_print['y_truth'] = y_true_test.flatten()
# test_print['diff'] = abs(y_pred_test.flatten() - y_true_test.flatten())
# test_print['diff(ratio)'] = abs(y_pred_test.flatten() - y_true_test.flatten())/y_true_test.flatten()
# test_print['diff(%)'] = (abs(y_pred_test.flatten() - y_true_test.flatten())/y_true_test.flatten())*100
# print (test_print)
