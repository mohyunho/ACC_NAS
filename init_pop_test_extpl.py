'''
DL models (FNN, 1D CNN and CNN-LSTM) evaluation on N-CMAPSS
12.07.2021
Hyunho Mo
hyunho.mo@unitn.it
'''
## Import libraries in python
import gc
import argparse
import os
import json
import logging
import sys
import h5py
import time
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt


from utils.pareto import pareto

import matplotlib.figure
import matplotlib.backends.backend_agg as agg
import matplotlib.backends.backend_svg as svg

from matplotlib.pyplot import cm


from matplotlib import gridspec
import math
import random
from random import shuffle
from tqdm.keras import TqdmCallback

import statistics


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
from tqdm import tqdm
import scipy.stats as stats
# from sklearn.utils.testing import ignore_warnings
# from sklearn.exceptions import ConvergenceWarning
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

from pygmo import *

import cvxpy as cp
from scipy.optimize import curve_fit

# from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph

from tensorflow.keras.initializers import GlorotNormal, GlorotUniform

initializer = GlorotNormal(seed=0)
# initializer = GlorotUniform(seed=0)

from utils.data_preparation_unit import df_all_creator, df_train_creator, df_test_creator, Input_Gen
from utils.dnn import one_dcnn, one_dcnn_baseline, one_dcnn_cmapss


seed = 0

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# Ignore tf err log
pd.options.mode.chained_assignment = None  # default='warn'


# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# tf.get_logger().setLevel(logging.ERROR)



# tf.config.set_visible_devices([], 'GPU')

current_dir = os.path.dirname(os.path.abspath(__file__))
data_filedir = os.path.join(current_dir, 'N-CMAPSS')
data_filepath = os.path.join(current_dir, 'N-CMAPSS', 'N-CMAPSS_DS02-006.h5')
sample_dir_path = os.path.join(data_filedir, 'Samples_whole')

model_temp_path = os.path.join(current_dir, 'Models', 'oned_cnn_rep.h5')
tf_temp_path = os.path.join(current_dir, 'TF_Model_tf')

pic_dir = os.path.join(current_dir, 'PF')

log_dir_path = os.path.join(current_dir, 'EA_log')

if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)

all_models = {}

#def pow3(x, c, a, alpha):
#    return  c - a * x**(-alpha)
#all_models["pow3"] = pow3

# def loglog_linear(x, a, b, c):
#     # x = x+1
#     x = np.log(x)
#     return -1*np.log(np.abs(a*x - b))+c
# all_models["loglog_linear"] = loglog_linear

# def pow4(x, c, a, b, alpha):
#     return c - (a*x+b)**-alpha
# all_models["pow4"] = pow4

# def exp(x, a,b,c,d):
#     return a + np.exp(b*(x**(-c)))+d
# all_models["exp"] = exp


def mmf(x, alpha, beta, kappa, delta):
    return alpha - (alpha - beta) / (1. + np.abs(kappa * x)**delta)
all_models["mmf"] = mmf

def janoschek(x, a, beta, k, delta):
    return a - (a - beta) * np.exp(-k*x**delta)
all_models["janoschek"] = janoschek

def weibull(x, alpha, beta, kappa, delta):
    x = 1 + x
    return alpha - (alpha - beta) * np.exp(-(kappa * x)**delta)
all_models["weibull"] = weibull

# def ilog2(x, c, a, b):
#     x = 1 + x
#     assert(np.all(x>1))
#     return -c + a / np.log(b*x)
# all_models["ilog2"] = ilog2

def reverse_gompertz(x, a, c, b, d):

    return a + (c-a)*(1-np.exp(-np.exp(-b*(x-d))))
    #return a + b * np.exp(np.exp(-k*(x-i)))
all_models["reverse_gompertz"] = reverse_gompertz


def hill_custom(x, a, b, c, d):
    return a + (b-a)/(1 + (10**(x-c))**d)
all_models["hill_custom"] = hill_custom


'''
load array from npz files
'''
def load_part_array (sample_dir_path, unit_num, win_len, stride, part_num):
    filename =  'Unit%s_win%s_str%s_part%s.npz' %(str(int(unit_num)), win_len, stride, part_num)
    filepath =  os.path.join(sample_dir_path, filename)
    loaded = np.load(filepath)
    return loaded['sample'], loaded['label']

def load_part_array_merge (sample_dir_path, unit_num, win_len, win_stride, partition):
    sample_array_lst = []
    label_array_lst = []
    print ("Unit: ", unit_num)
    for part in range(partition):
      print ("Part.", part+1)
      sample_array, label_array = load_part_array (sample_dir_path, unit_num, win_len, win_stride, part+1)
      sample_array_lst.append(sample_array)
      label_array_lst.append(label_array)
    sample_array = np.dstack(sample_array_lst)
    label_array = np.concatenate(label_array_lst)
    sample_array = sample_array.transpose(2, 0, 1)
    print ("sample_array.shape", sample_array.shape)
    print ("label_array.shape", label_array.shape)
    return sample_array, label_array


def load_array (sample_dir_path, unit_num, win_len, stride):
    filename =  'Unit%s_win%s_str%s.npz' %(str(int(unit_num)), win_len, stride)
    filepath =  os.path.join(sample_dir_path, filename)
    loaded = np.load(filepath)

    return loaded['sample'].transpose(2, 0, 1), loaded['label']


# def load_array (sample_dir_path, unit_num, win_len, stride, sampling):
#     filename =  'Unit%s_win%s_str%s_smp%s.npz' %(str(int(unit_num)), win_len, stride, sampling)
#     filepath =  os.path.join(sample_dir_path, filename)
#     loaded = np.load(filepath)

#     return loaded['sample'].transpose(2, 0, 1), loaded['label']


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


def train_params_count(model):
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    return trainableParams


def shuffle_array(sample_array, label_array):
    ind_list = list(range(len(sample_array)))
    print("ind_list befor: ", ind_list[:10])
    print("ind_list befor: ", ind_list[-10:])
    ind_list = shuffle(ind_list)
    print("ind_list after: ", ind_list[:10])
    print("ind_list after: ", ind_list[-10:])
    print("Shuffeling in progress")
    shuffle_sample = sample_array[ind_list, :, :]
    shuffle_label = label_array[ind_list,]
    return shuffle_sample, shuffle_label

def figsave(history, win_len, win_stride, bs, lr, sub):
    fig_acc = plt.figure(figsize=(15, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training', fontsize=24)
    plt.ylabel('loss', fontdict={'fontsize': 18})
    plt.xlabel('epoch', fontdict={'fontsize': 18})
    plt.legend(['Training loss', 'Validation loss'], loc='upper left', fontsize=18)
    plt.show()
    print ("saving file:training loss figure")
    fig_acc.savefig(pic_dir + "/training_w%s_s%s_bs%s_sub%s_lr%s.png" %(int(win_len), int(win_stride), int(bs), int(sub), str(lr)))
    return







def scheduler(epoch, lr):
    if epoch == 17:
        return lr * 0.1
    # elif epoch == 27:
    #     return lr * tf.math.exp(-0.1)
    else:
        return lr


def release_list(a):
   del a[:]
   del a

units_index_train = [2.0, 5.0, 10.0, 16.0, 18.0, 20.0]
units_index_test = [11.0, 14.0, 15.0]



def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='sample creator')
    parser.add_argument('-w', type=int, default=50, help='sequence length')
    parser.add_argument('-s', type=int, default=50, help='stride of filter')
    parser.add_argument('-f', type=int, default=10, help='number of filter')
    parser.add_argument('-k', type=int, default=10, help='size of kernel')
    parser.add_argument('-bs', type=int, default=512, help='batch size')
    parser.add_argument('-ep', type=int, default=30, help='max epoch')
    parser.add_argument('-pt', type=int, default=30, help='patience')
    parser.add_argument('-vs', type=float, default=0.2, help='validation split')
    parser.add_argument('-lr', type=float, default=10**(-1*4), help='learning rate')
    parser.add_argument('-sub', type=int, default=1, help='subsampling stride')
    parser.add_argument('--sampling', type=int, default=1, help='sub sampling of the given data. If it is 10, then this indicates that we assumes 0.1Hz of data collection')
    parser.add_argument('-t', type=int, required=True, help='trial')
    parser.add_argument('--pop', type=int, default=50, required=False, help='population size of EA')
    parser.add_argument('--gen', type=int, default=50, required=False, help='generations of evolution')
    parser.add_argument('--device', type=str, default="GPU", help='Use "basic" if GPU with cuda is not available')
    parser.add_argument('--obj', type=str, default="moo", help='Use "soo" for single objective and "moo" for multiobjective')
    parser.add_argument('--abl', type=str, default="ori", help='"ori or extpl"')
    parser.add_argument('-obep', type=int, default=15, help='ob ep')
    parser.add_argument('-start', type=int, default=2, help='start ep')
    args = parser.parse_args()

    win_len = args.w
    win_stride = args.s
    partition = 3
    n_filters = args.f
    kernel_size = args.k
    lr = args.lr
    bs = args.bs
    ep = args.ep
    pt = args.pt
    vs = args.vs
    sub = args.sub
    sampling = args.sampling
    ablation = args.abl
    ob_ep = args.obep
    st_ep = args.start

    lr = 10**(-1*4)

    device = args.device
    obj = args.obj
    trial = args.t
    pop_size = args.pop
    n_generations = args.gen


    log_filename = 'mute_log_%s_%s_%s_%s_%s_%s.csv'  %(ablation, pop_size, n_generations, obj, trial, ob_ep)

    mutlog_df = pd.read_csv(os.path.join(log_dir_path, log_filename))


    prft_log_df = mutlog_df.loc[mutlog_df['gen']==0]

    # mutate_filename = os.path.join(log_dir_path, 'mute_log_%s_%s_%s_%s_%s_%s.csv' % (ablation, pop_size, n_generations, obj, trial, ep))

    # # mutate_filename = 'EA_log/mute_log_%s_%s_%s_%s.csv' % (pop_size, n_generations, obj, trial)
    # ea_log_df = pd.read_csv(mutate_filename)

    # last_gen_df = ea_log_df.loc[ea_log_df['gen'] == n_generations]
    # print ("last_gen_df", last_gen_df)


    print (prft_log_df)



    train_units_samples_lst =[]
    train_units_labels_lst = []


    for index in units_index_train:
        print("Load data index: ", index)
        sample_array, label_array = load_array (sample_dir_path, index, win_len, win_stride)
        #sample_array, label_array = shuffle_array(sample_array, label_array)
        print("sample_array.shape", sample_array.shape)
        print("label_array.shape", label_array.shape)
        sample_array = sample_array[::sub]
        label_array = label_array[::sub]

        sample_array = sample_array.astype(np.float32)
        label_array = label_array.astype(np.float32)

        print("sub sample_array.shape", sample_array.shape)
        print("sub label_array.shape", label_array.shape)
        train_units_samples_lst.append(sample_array)
        train_units_labels_lst.append(label_array)

    sample_array = np.concatenate(train_units_samples_lst)
    label_array = np.concatenate(train_units_labels_lst)
    print ("samples are aggregated")

    release_list(train_units_samples_lst)
    release_list(train_units_labels_lst)
    train_units_samples_lst =[]
    train_units_labels_lst = []
    print("Memory released")

    sample_array, label_array = shuffle_array(sample_array, label_array)
    print("samples are shuffled")
    print("sample_array.shape", sample_array.shape)
    print("label_array.shape", label_array.shape)

    # sample_array = sample_array.reshape(sample_array.shape[0], sample_array.shape[2])
    print("sample_array_reshape.shape", sample_array.shape)
    print("label_array_reshape.shape", label_array.shape)
    window_length = sample_array.shape[1]
    feat_len = sample_array.shape[2]
    num_samples = sample_array.shape[0]
    print ("window_length", window_length)
    print("feat_len", feat_len)

    train_sample_array = sample_array[:int(num_samples*(1-vs))]
    train_label_array = label_array[:int(num_samples*(1-vs))]
    val_sample_array = sample_array[int(num_samples*(1-vs))+1:]
    val_label_array = label_array[int(num_samples*(1-vs))+1:]

    print ("train_sample_array.shape", train_sample_array.shape)
    print ("train_label_array.shape", train_label_array.shape)
    print ("val_sample_array.shape", val_sample_array.shape)
    print ("val_label_array.shape", val_label_array.shape)

    sample_array = []
    label_array = []

    release_list(train_units_samples_lst)
    release_list(train_units_labels_lst)
    train_units_samples_lst =[]
    train_units_labels_lst = []
    print("Memory released")



    # input_temp = Input(shape=(sample_array.shape[1], sample_array.shape[2]),name='kernel_size%s' %str(int(kernel_size)))
    # #------
    # one_d_cnn = one_dcnn(n_filters, kernel_size, sample_array, initializer)
    # cnn_out = one_d_cnn(input_temp)
    # x = cnn_out
    # # x = Dropout(0.5)(x)
    # main_output = Dense(1, activation='linear', kernel_initializer=initializer, name='main_output')(x)
    # one_d_cnn_model = Model(inputs=input_temp, outputs=main_output)

    # model = Model(inputs=[input_1, input_2], outputs=main_output)

    last_gen_rmse_lst = []
    last_gen_score_lst = []
    last_gen_numparams_lst = []
    last_gen_traintime_lst = []
    last_gen_infertime_lst = []
    last_gen_val_lst = []

    points_val = []
    points_test = []

    for idx, row in prft_log_df.iterrows():

        

        n_layers = int(row['params_1'] )
        n_filters = int(row['params_2'] ) 
        kernel_size = int(row['params_3'])  
        n_mlp = int(row['params_4'] ) *10
        one_d_cnn_model = one_dcnn(n_layers, n_filters, kernel_size, n_mlp, train_sample_array, initializer)

        geno_lst = [n_layers, n_filters, kernel_size, n_mlp]

        print (n_layers, n_filters, kernel_size, n_mlp)
        # one_d_cnn_model = one_dcnn_cmapss(sample_array)

        print(one_d_cnn_model.summary())
        # one_d_cnn_model.compile(loss='mean_squared_error', optimizer=amsgrad, metrics=[rmse, 'mae'])


        start = time.time()

        lr_scheduler = LearningRateScheduler(scheduler)

        keras_rmse = tf.keras.metrics.RootMeanSquaredError()

        amsgrad = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True, name='Adam')
        rmsop = optimizers.RMSprop(learning_rate=lr, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
                            name='RMSprop')

        one_d_cnn_model.compile(loss='mean_squared_error', optimizer=amsgrad, metrics=['mae', keras_rmse ])
        # history = one_d_cnn_model.fit(sample_array, label_array, epochs=ep, batch_size=bs, validation_split=vs, verbose=2,
        #                 callbacks = [lr_scheduler, EarlyStopping(monitor='val_loss', min_delta=0, patience=pt, verbose=1, mode='min'),
        #                                 ModelCheckpoint(model_temp_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)]
        #                 )

        history = one_d_cnn_model.fit(train_sample_array, train_label_array, epochs=ep, batch_size=bs,
                validation_data=(val_sample_array, val_label_array), verbose=2,
                callbacks=[lr_scheduler, EarlyStopping(monitor='val_loss', min_delta=0, patience=pt, verbose=1,
                                                    mode='min'),
                            ModelCheckpoint(model_temp_path, monitor='val_loss',
                                                        save_best_only=True, mode='min', verbose=1)]
                            )

        # TqdmCallback(verbose=2)
        # one_d_cnn_model.save(tf_temp_path,save_format='tf')
        figsave(history, win_len, win_stride, bs, lr, sub)

        # print("The FLOPs is:{}".format(get_flops(one_d_cnn_model)), flush=True)
        num_train = train_sample_array.shape[0]
        end = time.time()
        training_time = end - start
        print("Training time: ", training_time)


        val_rmse_hist = history.history['val_root_mean_squared_error']
        print ("val_rmse_hist[-1]", val_rmse_hist[-1])

        num_tran_params = train_params_count(one_d_cnn_model)

        ### Test (inference after training)
        start = time.time()

        output_lst = []
        truth_lst = []

        for index in units_index_test:
            print ("test idx: ", index)
            # sample_array, label_array = load_array(sample_dir_path, index, win_len, win_stride, sampling)
            test_sample_array, test_label_array = load_array(sample_dir_path, index, win_len, win_stride)
            # estimator = load_model(tf_temp_path, custom_objects={'rmse':rmse})
            print("sample_array.shape", test_sample_array.shape)
            print("label_array.shape", test_label_array.shape)
            test_sample_array = test_sample_array[::sub]
            test_label_array = test_label_array[::sub]
            print("sub sample_array.shape", test_sample_array.shape)
            print("sub label_array.shape", test_label_array.shape)

            estimator = load_model(model_temp_path)

            y_pred_test = estimator.predict(test_sample_array)
            output_lst.append(y_pred_test)
            truth_lst.append(test_label_array)

        print(output_lst[0].shape)
        print(truth_lst[0].shape)

        print(np.concatenate(output_lst).shape)
        print(np.concatenate(truth_lst).shape)

        output_array = np.concatenate(output_lst)[:, 0]
        trytg_array = np.concatenate(truth_lst)

        test_rms = sqrt(mean_squared_error(output_array, trytg_array))
        print(test_rms)
        test_rms = round(test_rms, 2)

        end = time.time()
        inference_time = end - start
        num_test = output_array.shape[0]


        h_array = output_array - trytg_array
        # print (h_array)
        s_array = np.zeros(len(h_array))
        for j, h_j in enumerate(h_array):
            if h_j < 0:
                s_array[j] = math.exp(-(h_j / 13)) - 1
            else:
                s_array[j] = math.exp(h_j / 10) - 1

        score = np.sum(s_array)
        score = round(score, 2)

        training_time = round(training_time, 2)
        inference_time = round(inference_time, 2)

        print("Training time: ", training_time)
        print("Inference time: ", inference_time)
        print("number of trainable parameters: ", num_tran_params )
        print ("score", score)
        print("Result in RMSE: ", test_rms)


        # Validation RMSE observation history
        numb_obeservation = ob_ep
        val_rmse_hist = history.history['val_root_mean_squared_error']  
        x_max = np.arange(1,31)
        x_observation = x_max[:numb_obeservation]
        y_obesrvation = val_rmse_hist[:numb_obeservation]

        fig = matplotlib.figure.Figure(figsize=(8, 6))
        # agg.FigureCanvasAgg(fig)
        ax = fig.add_subplot(1, 1, 1)
        # Plot actual curve with solid line
        ax.plot(x_observation, y_obesrvation, label="observation", color="black", linewidth=2, zorder=4)
        color = iter(cm.rainbow(np.linspace(0, 1, len(all_models)+1)))


        y_func_lst = []
        curve_y_lst = []
        y_func_end_lst = []


        x_observation = x_max[st_ep:numb_obeservation]
        print ("len(x_observation)", len(x_observation))
        y_obesrvation = val_rmse_hist[st_ep:numb_obeservation]
  
        for index, (key, value) in enumerate(all_models.items()):
            print ("value", value)
            next_x = np.arange(numb_obeservation+1,31)
            if key == "loglog_linear":
                fitting_parameters, covariance = curve_fit(value, x_observation, y_obesrvation, p0=[10,1,10], maxfev=10000000)
            elif key == "pow3":
                fitting_parameters, covariance = curve_fit(value, x_observation, y_obesrvation, maxfev=10000000)
            elif key == "ilog2":
                fitting_parameters, covariance = curve_fit(value, x_observation, y_obesrvation, maxfev=10000000)
            else:
                fitting_parameters, covariance = curve_fit(value, x_observation, y_obesrvation, maxfev=10000000)
            
            if len(fitting_parameters)==2:
                y_func = value(x_observation, fitting_parameters[0], fitting_parameters[1])
                next_y = value(next_x, fitting_parameters[0], fitting_parameters[1])
            elif len(fitting_parameters)==3:
                y_func = value(x_observation, fitting_parameters[0], fitting_parameters[1], fitting_parameters[2])
                next_y = value(next_x, fitting_parameters[0], fitting_parameters[1], fitting_parameters[2])
            elif len(fitting_parameters)==4:
                y_func = value(x_observation, fitting_parameters[0], fitting_parameters[1], fitting_parameters[2], fitting_parameters[3])
                next_y = value(next_x, fitting_parameters[0], fitting_parameters[1], fitting_parameters[2], fitting_parameters[3])
            
            y_func_end_lst.append(y_func[-1])
            y_func_lst.append(y_func)
            curve_y_lst.append(np.append(y_func, next_y))
            # next_y_lst.append(next_y)

            
            # ax.plot(np.append(y, next_y), 'ro')
            c = next(color)
            # ax.plot(next_x, next_y, color=c, marker='o', label=key)

            # ax.plot(x_max, np.append(y_func, next_y), color=c, marker='o', linestyle='dashed', linewidth=2, label=key, zorder=1)
            ax.plot(x_max[st_ep:], np.append(y_func, next_y), color=c, marker='o', linestyle='dashed', linewidth=2, label=key, zorder=1)



            

        # list of 1d arrays to 2d array
        print ("y_func_lst", y_func_lst)
        print ("y_func_end_lst", y_func_end_lst)
        input_arrays = np.transpose(np.stack(y_func_lst, axis=0))
        print ("input_arrays.shape", input_arrays.shape)
        # Find coefficient of linear combination of vectors with least square (olve a least-squares problem with CVXPY) 
        # https://www.cvxpy.org/examples/basic/least_squares.html
        # m: length of vector, n: numb of curves
        # Shape of A:(m, n), length of b: (m)
        n = len(all_models)
        coeff = cp.Variable(n)
        cost = cp.sum_squares(input_arrays @ coeff - y_obesrvation)
        prob = cp.Problem(cp.Minimize(cost))
        prob.solve()

        print("The optimal coeff is")
        print(coeff.value)

        if coeff.value is None:
            print ("cannot fit the curve, assign rms 20")
            rms = sum(y_func_end_lst) / len(y_func_end_lst)
            if rms >= y_obesrvation[-1]:
                rms = y_obesrvation[-1]
                rms = round(rms, 4)

        else:
            # Combine (linear combination) of curves
            curves_arrays = np.transpose(np.stack(curve_y_lst, axis=0))
            print("curves_arrays.shape", curves_arrays.shape)
            combined_y = curves_arrays  @ coeff.value 
            print ("combined_y", combined_y)



            c = next(color)
            # ax.plot(x_max, combined_y, color=c, marker='D', linewidth=1.5, label="combined", zorder=2)
            ax.plot(x_max[st_ep:], combined_y, color=c, marker='D', linewidth=1.5, label="combined", zorder=2)


            ax.legend(loc='upper right', fontsize=12)
            ax.set_xlabel('Epoch', fontsize=15)
            ax.set_ylabel('Validation RMSE', fontsize=15)
            x_epoch = np.arange(1,31)
            ax.set_xticks(x_epoch)
            ax.set_xticklabels(x_epoch, rotation=60)
            ymax_plot = 25
            ax.set_ylim(0, ymax_plot)
            ax.vlines(numb_obeservation,  0, ymax_plot, colors=(0.7, 0.7, 0.7), linestyle='-.',linewidth=1, zorder=3)

            extpl_rmse = combined_y[-1]
            print ("combined_y[-1]", combined_y[-1])
            # if (extpl_rmse <= 5.0) or (abs(y_obesrvation[-1] - combined_y[-1])>=3):
            #     rms = 30.0
            # else:
            #     rms = round(extpl_rmse, 4)

            if abs(y_obesrvation[-1] - combined_y[-1])>=1:
                rms = sum(y_func_end_lst) / len(y_func_end_lst)
                rms = round(rms, 4)
            else:
                rms = round(extpl_rmse, 4)

            if rms >= y_obesrvation[-1]:
                rms = y_obesrvation[-1]
                rms = round(rms, 4)

            fig.suptitle('Last ob: %s, Estimated validation RSME: %s' %(round(y_obesrvation[-1],4),rms), fontsize=12)
            fig.savefig(os.path.join(pic_dir, 'curve_extpl_%s.png' %str(geno_lst)), bbox_inches='tight')

            # val_rmse_extpl = round(combined_y[-1], 4)
            val_rmse_extpl = rms

            if val_rmse_extpl >=15:
                val_rmse_extpl = 15




        last_gen_rmse_lst.append(test_rms)
        last_gen_score_lst.append(score)
        last_gen_numparams_lst.append(num_tran_params / 10000)
        last_gen_traintime_lst.append(training_time)
        last_gen_infertime_lst.append(inference_time)
        last_gen_val_lst.append(val_rmse_extpl)


        
        points_val.append([val_rmse_extpl, num_tran_params / 10000])
        points_test.append([test_rms, num_tran_params / 10000])
    
    prft_infer_df = pd.DataFrame([])

    prft_infer_df['p1']  = prft_log_df['params_1'].values
    prft_infer_df['p2']  = prft_log_df['params_2'].values
    prft_infer_df['p3']  = prft_log_df['params_3'].values
    prft_infer_df['p4']  = prft_log_df['params_4'].values

    print ("last_gen_numparams_lst", last_gen_numparams_lst)

    prft_infer_df['train_time'] = last_gen_traintime_lst
    prft_infer_df['infer_time'] = last_gen_infertime_lst
    prft_infer_df['val_rmse'] = last_gen_val_lst
    prft_infer_df['num_params'] = last_gen_numparams_lst
    prft_infer_df['RMSE'] = last_gen_rmse_lst
    prft_infer_df['score'] = last_gen_score_lst

    print ("BEST ind test RMSE", min(last_gen_rmse_lst))
    print ("Average test RMSE", statistics.mean(last_gen_rmse_lst) )
        
    print ("BEST ind test score", min(last_gen_score_lst))
    print ("Average test score", statistics.mean(last_gen_score_lst) )

    output_filepath = os.path.join(log_dir_path, 'inference_prft_%s_%s_0_%s_%s_%s.csv' % (ablation, pop_size,  obj, trial, ob_ep))
    prft_infer_df.to_csv(output_filepath, index=False)


    print ("points_val", points_val)
    print ("points_test", points_test)
    print ("csv saved")

    ref = [15.0, 15.0]


    hv_val = hypervolume(points = points_val)
    result_val = hv_val.compute(ref_point = ref)
    print ("hypervolume_validation", result_val)

    hv_test = hypervolume(points = points_test)
    result_test = hv_test.compute(ref_point = ref)
    print ("hypervolume_test", result_test)



if __name__ == '__main__':
    main()
