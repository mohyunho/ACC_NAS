import time
import json
import logging as log
import sys

import os
import math
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
# import keras

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
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
# from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform

from utils.archt_scoring import scorefunc_slogdet, tf_net_kmatrix
import cupy as cp

# os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC']='1'

from utils.dnn import one_dcnn
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'



# Ignore tf err log
pd.options.mode.chained_assignment = None  # default='warn'

def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def train_params_count(model):
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    return trainableParams



# def scheduler(epoch, lr):
#     if epoch == 30:
#         print("lr decay by 10")
#         return lr * 0.1
#     elif epoch == 70:
#         print("lr decay by 10")
#         return lr * 0.1
#     else:
#         return lr

def rmse(y_true, y_pred):
    """Metrics for evaluation"""
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

initializer = GlorotNormal(seed=0)
# initializer = GlorotUniform(seed=0)


# def scheduler(epoch, lr):
#     if epoch == 10:
#         return lr * 0.1
#     elif epoch == 20:
#         return lr * tf.math.exp(-0.1)
#     else:
#         return lr


def scheduler(epoch, lr):
    if epoch == 17:
        return lr * 0.1
    # elif epoch == 27:
    #     return lr * tf.math.exp(-0.1)
    else:
        return lr


class network_fit(object):
    '''
    class for network
    '''

    def __init__(self, sample_array, n_layers,
                 n_filters, kernel_size, n_mlp, batch, epoch, patience, val_split, model_path, device):
        '''
        Constructor
        Generate a NN and train
        @param none
        '''
        # self.__logger = logging.getLogger('data preparation for using it as the network input')
        self.sample_array = sample_array
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_mlp = n_mlp
        self.batch = batch
        self.epoch = epoch
        self.patience = patience
        self.val_split = val_split
        self.model_path = model_path
        self.device = device

        # self.model= gen_net(self.feat_len, self.l2_parm, self.lin_check,
        #                     self.num_neurons_lst, self.type_lst, self.device)

    def trained_model(self):
        model = one_dcnn(self.n_layers, self.n_filters, self.kernel_size, self.n_mlp, self.sample_array, initializer)
        return model


    def train_net(self, model, lr, train_sample_array, train_label_array, val_sample_array, val_label_array):
        '''
        specify the optimizers and train the network
        :param epochs:
        :param batch_size:
        :param lr:
        :return:
        '''
        print("Initializing network...")
        start_itr = time.time()
        keras_rmse = tf.keras.metrics.RootMeanSquaredError()


        amsgrad = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True, name='Adam')
        rmsop = optimizers.RMSprop(learning_rate=lr, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
                                   name='RMSprop')

        lr_scheduler = LearningRateScheduler(scheduler)

        model.compile(loss='mean_squared_error', optimizer=amsgrad, metrics=['mae', keras_rmse ])
        history = model.fit(train_sample_array, train_label_array, epochs=self.epoch, batch_size=self.batch,
                            validation_data=(val_sample_array, val_label_array), verbose=0,
                            callbacks=[lr_scheduler, EarlyStopping(monitor='val_loss', min_delta=0, patience=self.patience, verbose=0,
                                                               mode='min'),
                                       ModelCheckpoint(self.model_path, monitor='val_loss',
                                                                 save_best_only=True, mode='min', verbose=0)]
                                      )


   


        val_loss_history = history.history['val_loss']
        # print ("val_loss_history", val_loss_history)
        val_loss_history = [round(num, 4) for num in val_loss_history]

        val_rmse_hist = history.history['val_root_mean_squared_error']


        import csv   
        with open('val_rmse_hist.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(val_rmse_hist)    



        val_pred = model.predict(val_sample_array)
        val_pred = val_pred.flatten()
        rms = sqrt(mean_squared_error(val_pred, val_label_array))
        rms = round(rms, 4)
        val_net = (rms,)
        end_itr = time.time()
        train_time = end_itr - start_itr

        num_tran_params = train_params_count(model)
        print("number of trainable parameters: ", num_tran_params )
        print("training network is successfully completed, time: ", train_time)
        print("val_net in rmse: ", val_net[0])
        print ("val_rmse_hist[-1]", val_rmse_hist[-1])


        model = None
        val_pred = None
        del model, val_pred
 
        with open('train_time_hist.csv', 'a') as fq:
            writer = csv.writer(fq)
            writer.writerow([train_time])   


        return val_net, num_tran_params, train_time

    
    def score_net(self, model, lr, train_sample_array, train_label_array, val_sample_array, val_label_array):
        '''
        specify the optimizers and train the network
        :param epochs:
        :param batch_size:
        :param lr:
        :return:
        '''
        print("Initializing network...")
        start_itr = time.time()
        keras_rmse = tf.keras.metrics.RootMeanSquaredError()

        ###### Calculate score #############

        # Calculate model's score

        kmatrix = tf_net_kmatrix(model, self.batch, train_sample_array)

        # print ("output kmatrix: ", kmatrix)

        sign, archt_score = scorefunc_slogdet (kmatrix)
        print ("archt_score", archt_score)
        if int(sign) == 0:
            archt_score = 0

        print ("sign", sign)
        print ("archt_score", archt_score)

        if archt_score == 0:
            archt_score_inverse = 1000.0
        else:
            archt_score_div = archt_score / 10000.0
            archt_score_inverse = 1/archt_score_div

        if archt_score_inverse <= 0:
            archt_score_inverse = archt_score_inverse * -1

        print ("archt_score_inverse", archt_score_inverse)
        archt_score_inverse = cp.asnumpy(archt_score_inverse)
        archt_score_inverse = float(archt_score_inverse)
        print ("type(archt_score_inverse)", type(archt_score_inverse))

        ####################################

        archt_score_inverse = round(archt_score_inverse, 6)
        val_net = (archt_score_inverse,)
        end_itr = time.time()
        train_time = end_itr - start_itr

        num_tran_params = train_params_count(model)
        print("number of trainable parameters: ", num_tran_params )
        print("training network is successfully completed, time: ", train_time)
        print("val_net in rmse: ", val_net[0])


        model = None
        del model


        return val_net, num_tran_params, train_time

