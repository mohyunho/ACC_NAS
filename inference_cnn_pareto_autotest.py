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

from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph

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

pic_dir = os.path.join(current_dir, 'Figures')

log_dir_path = os.path.join(current_dir, 'EA_log')



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

    lr = 10**(-1*4)

    device = args.device
    obj = args.obj
    trial = args.t
    pop_size = args.pop
    n_generations = args.gen


    prft_filename = 'prft_out_%s_%s_%s_%s_%s.csv'  %(ablation, pop_size, n_generations, trial, ep)

    prft_log_df = pd.read_csv(os.path.join(log_dir_path, prft_filename), header=0, names=["p1", 'p2', 'p3', 'p4'])


    mutate_filename = os.path.join(log_dir_path, 'mute_log_%s_%s_%s_%s_%s_%s.csv' % (ablation, pop_size, n_generations, obj, trial, ep))

    # mutate_filename = 'EA_log/mute_log_%s_%s_%s_%s.csv' % (pop_size, n_generations, obj, trial)
    ea_log_df = pd.read_csv(mutate_filename)

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


        n_layers = int(row['p1'] )
        n_filters = int(row['p2'] ) 
        kernel_size = int(row['p3'])  
        n_mlp = int(row['p4'] ) *10
        one_d_cnn_model = one_dcnn(n_layers, n_filters, kernel_size, n_mlp, train_sample_array, initializer)

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

        print("The FLOPs is:{}".format(get_flops(one_d_cnn_model)), flush=True)
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

        rms = sqrt(mean_squared_error(output_array, trytg_array))
        print(rms)
        rms = round(rms, 2)

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
        print("Result in RMSE: ", rms)

        last_gen_rmse_lst.append(rms)
        last_gen_score_lst.append(score)
        last_gen_numparams_lst.append(num_tran_params)
        last_gen_traintime_lst.append(training_time)
        last_gen_infertime_lst.append(inference_time)
        last_gen_val_lst.append(val_rmse_hist[-1])

        
        points_val.append([val_rmse_hist[-1], num_tran_params])
        points_test.append([rms, num_tran_params])
    
    prft_infer_df = pd.DataFrame([])

    prft_infer_df['p1']  = prft_log_df['p1'].values
    prft_infer_df['p2']  = prft_log_df['p2'].values
    prft_infer_df['p3']  = prft_log_df['p3'].values
    prft_infer_df['p4']  = prft_log_df['p4'].values

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

    output_filepath = os.path.join(log_dir_path, 'inference_prft_%s_%s_%s_%s_%s_%s.csv' % (ablation, pop_size, n_generations, obj, trial, ep))
    prft_infer_df.to_csv(output_filepath, index=False)


################### pareto front plot ###############
    # Draw scatter plot
    fig = matplotlib.figure.Figure(figsize=(3, 3))
    agg.FigureCanvasAgg(fig)
    # cmap = get_cmap(10)
    ax = fig.add_subplot(1, 1, 1)
    # Draw scatter plot

    prft_rmse = prft_infer_df['val_rmse'].values
    prft_params = prft_infer_df['num_params'].values
    

    # x_min = int(min(prft_rmse)) - 0.5
    # x_max = int(max(prft_rmse)) + 0.5
    # x_sp = 0.25
    # x_range = np.arange(x_min, x_max, 2 * x_sp)

    # ax.scatter(mute_log_df['fitness_1'], mute_log_df['test_rmse'], facecolor=(1.0, 1.0, 0.4), edgecolors=(0.0, 0.0, 0.0), zorder=1,
    #            c=cmap(0), s=20 )

    ax.scatter(prft_rmse, prft_params, facecolor=(1.0, 1.0, 0.4),
               edgecolors=(0.0, 0.0, 0.0), zorder=1, s=20,  label="Solutions" )

    ax.scatter(8.23, 5722, marker="D",facecolor=(0.0, 1.0, 0.0), edgecolors=(0.0, 0.0, 0.0), zorder=1,
           s=60, label="Handcrafted CNN")

    x_range = np.arange(5.0, 10.5, 0.5)
    ax.set_xticks(x_range)
    ax.set_xticklabels(x_range, rotation=60)
    # ax.set_yticks(np.arange(y_min, y_max, 2 * y_sp))
    ax.set_xlim(5, 10)
    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)
    # ax.set_title("Solutions and pareto front", fontsize=15)
    ax.set_xlabel('Validation RMSE', fontsize=12)
    ax.set_ylabel('Trainable parameters', fontsize=12)
    ax.legend(fontsize=8, loc='upper right')

    # Save figure
    # ax.set_rasterized(True)
    fig.savefig(os.path.join(pic_dir, 'infer_prft_val_%s_%s_%s_%s_%s.png' % (ablation, pop_size, n_generations, trial, ep)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.eps' % (pop, gen, trial)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.pdf' % (pop, gen, trial)), bbox_inches='tight')

##############################################

    # Draw scatter plot
    fig = matplotlib.figure.Figure(figsize=(3, 3))
    agg.FigureCanvasAgg(fig)
    # cmap = get_cmap(10)
    ax = fig.add_subplot(1, 1, 1)
    # Draw scatter plot

    prft_rmse = prft_infer_df['RMSE'].values
    prft_params = prft_infer_df['num_params'].values
    

    # x_min = int(min(prft_rmse)) - 0.5
    # x_max = int(max(prft_rmse)) + 0.5
    # x_sp = 0.25
    # x_range = np.arange(x_min, x_max, 2 * x_sp)

    # ax.scatter(mute_log_df['fitness_1'], mute_log_df['test_rmse'], facecolor=(1.0, 1.0, 0.4), edgecolors=(0.0, 0.0, 0.0), zorder=1,
    #            c=cmap(0), s=20 )

    ax.scatter(prft_rmse, prft_params, facecolor=(1.0, 1.0, 0.4),
               edgecolors=(0.0, 0.0, 0.0), zorder=1, s=20,  label="Solutions" )

    ax.scatter(8.04, 5722, marker="D",facecolor=(0.0, 1.0, 0.0), edgecolors=(0.0, 0.0, 0.0), zorder=1,
           s=60, label="Handcrafted CNN")

    x_range = np.arange(5.0, 10.5, 0.5)
    ax.set_xticks(x_range)
    ax.set_xticklabels(x_range, rotation=60)
    # ax.set_yticks(np.arange(y_min, y_max, 2 * y_sp))
    ax.set_xlim(5, 10)
    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)
    # ax.set_title("Solutions and pareto front", fontsize=15)
    ax.set_xlabel('Test RMSE', fontsize=12)
    ax.set_ylabel('Trainable parameters', fontsize=12)
    ax.legend(fontsize=8, loc='upper right')

    # Save figure
    # ax.set_rasterized(True)
    fig.savefig(os.path.join(pic_dir, 'infer_prft_test_%s_%s_%s_%s_%s.png' % (ablation, pop_size, n_generations, trial, ep)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.eps' % (pop, gen, trial)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.pdf' % (pop, gen, trial)), bbox_inches='tight')



######################################


    ref = [30.0, 200000.0]

    hv_val = hypervolume(points = points_val)
    result_val = hv_val.compute(ref_point = ref)
    print ("hypervolume_validation", result_val)

    hv_test = hypervolume(points = points_test)
    result_test = hv_test.compute(ref_point = ref)
    print ("hypervolume_test", result_test)


if __name__ == '__main__':
    main()
