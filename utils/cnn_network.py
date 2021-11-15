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
np.random.seed(0)

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
from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform

from utils.dnn import one_dcnn

# Ignore tf err log
pd.options.mode.chained_assignment = None  # default='warn'

def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def train_params_count(model):
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    return trainableParams

def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops

# def scheduler(epoch, lr):
#     if epoch == 30:
#         print("lr decay by 10")
#         return lr * 0.1
#     elif epoch == 70:
#         print("lr decay by 10")
#         return lr * 0.1
#     else:
#         return lr

initializer = GlorotNormal(seed=0)
# initializer = GlorotUniform(seed=0)


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

        amsgrad = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True, name='Adam')
        rmsop = optimizers.RMSprop(learning_rate=lr, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
                                   name='RMSprop')

        model.compile(loss='mean_squared_error', optimizer=amsgrad, metrics='mae')
        history = model.fit(train_sample_array, train_label_array, epochs=self.epoch, batch_size=self.batch,
                            validation_data=(val_sample_array, val_label_array), verbose=2,
                            callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=self.patience, verbose=1,
                                                               mode='min'),
                                       ModelCheckpoint(self.model_temp_path, monitor='val_loss',
                                                                 save_best_only=True, mode='min', verbose=1)]
                                      )



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


        model = None
        val_pred = None
        del model, val_pred


        return val_net, num_tran_params, train_time


