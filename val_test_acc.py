import argparse
import time
import json
import logging
import sys
import glob
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import importlib
from scipy.stats import randint, expon, uniform
from tqdm import tqdm
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

current_dir = os.path.dirname(os.path.abspath(__file__))
data_filedir = os.path.join(current_dir, 'N-CMAPSS')
data_filepath = os.path.join(current_dir, 'N-CMAPSS', 'N-CMAPSS_DS02-006.h5')
sample_dir_path = os.path.join(data_filedir, 'Samples_whole')

model_temp_path = os.path.join(current_dir, 'Models', 'oned_cnn_rep.h5')
tf_temp_path = os.path.join(current_dir, 'TF_Model_tf')

pic_dir = os.path.join(current_dir, 'Figures')

# directory_path = current_dir + '/EA_log'
directory_path = os.path.join(current_dir, 'EA_log')


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


def release_list(a):
   del a[:]
   del a


def recursive_clean(directory_path):
    """clean the whole content of :directory_path:"""
    if os.path.isdir(directory_path) and os.path.exists(directory_path):
        files = glob.glob(directory_path + '*')
        for file_ in files:
            if os.path.isdir(file_):
                recursive_clean(file_ + '/')
            else:
                os.remove(file_)

units_index_train = [2.0, 5.0, 10.0, 16.0, 18.0, 20.0]
units_index_test = [11.0, 14.0, 15.0]


initializer = GlorotNormal(seed=0)
# initializer = GlorotUniform(seed=0)


def scheduler(epoch, lr):
    if epoch == 15:
        return lr * 0.1
    elif epoch == 25:
        return lr * tf.math.exp(-0.1)
    else:
        return lr



def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='NAS CNN')
    parser.add_argument('-w', type=int, default=50, help='sequence length', required=True)
    parser.add_argument('-s', type=int, default=1, help='stride of filter')
    parser.add_argument('-bs', type=int, default=1000, help='batch size')
    parser.add_argument('-ep', type=int, default=30, help='max epoch')
    parser.add_argument('-pt', type=int, default=20, help='patience')
    parser.add_argument('-vs', type=float, default=0.1, help='validation split')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-sub', type=int, default=1, help='subsampling stride')
    parser.add_argument('-t', type=int, required=True, help='trial')
    parser.add_argument('--pop', type=int, default=50, required=False, help='population size of EA')
    parser.add_argument('--gen', type=int, default=50, required=False, help='generations of evolution')
    parser.add_argument('--device', type=str, default="GPU", help='Use "basic" if GPU with cuda is not available')
    parser.add_argument('--obj', type=str, default="moo", help='Use "soo" for single objective and "moo" for multiobjective')

    args = parser.parse_args()

    win_len = args.w
    win_stride = args.s

    lr = args.lr
    bs = args.bs
    ep = args.ep
    pt = args.pt
    vs = args.vs
    sub = args.sub

    device = args.device
    obj = args.obj
    trial = args.t

    pop = args.pop
    gen = args.gen

    # random seed predictable
    jobs = 1
    seed = trial
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

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
######################################
### Test
    test_units_samples_lst =[]
    test_units_labels_lst = []

    for index in units_index_test:
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
        test_units_samples_lst.append(sample_array)
        test_units_labels_lst.append(label_array)

    test_sample_array = np.concatenate(test_units_samples_lst)
    test_label_array = np.concatenate(test_units_labels_lst)
    print ("samples are aggregated")

    release_list(test_units_samples_lst)
    release_list(test_units_labels_lst)
    test_units_samples_lst =[]
    test_units_labels_lst = []
    sample_array = []
    label_array = []
    print("Memory released")



##### Load save individuals and generate phenotype
    log_file = os.path.join(directory_path, 'mute_log_%s_%s_soo_%s.csv' %(pop,gen,trial))
    mute_log_df = pd.read_csv(log_file)

    # lst
    test_rmse = []
    flops = []
    train_params = []
    train_time = []
    val_rms_lst =[]
    # archt_scores = []


    # Iterows

    # selected_df = mute_log_df.loc[(mute_log_df['idx'] >=0)&(mute_log_df['idx'] <= 100)]
    print (mute_log_df)
    # selected_df = mute_log_df.iloc[0:100,:]
    selected_df = mute_log_df
    print (selected_df)

    # for index, ind in mute_log_df.iterrows():
    for index, ind in tqdm(selected_df.iterrows(), total=selected_df.shape[0]):
        print (ind['params_1'], ind['params_2'], ind['params_3'], ind['params_4'])
        n_layers = int(ind['params_1'])
        n_filters = int(ind['params_2'])
        kernel_size = int(ind['params_3'])
        n_mlp = 10 * int(ind['params_4'])
        lr = 10**(-1*4)

###################
        # model = one_dcnn(n_layers, n_filters, kernel_size, n_mlp, train_sample_array, initializer)

        # Calculate model's score
        # archt_score =


###################
        model = one_dcnn(n_layers, n_filters, kernel_size, n_mlp, train_sample_array, initializer)

        keras_rmse = tf.keras.metrics.RootMeanSquaredError()
        # print("Initializing network...")
        start_itr = time.time()
        amsgrad = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True, name='Adam')
        rmsop = optimizers.RMSprop(learning_rate=lr, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
                                   name='RMSprop')

        lr_scheduler = LearningRateScheduler(scheduler)

        model.compile(loss='mean_squared_error', optimizer=amsgrad, metrics=['mae', keras_rmse ])
        history = model.fit(train_sample_array, train_label_array, epochs=ep, batch_size=bs,
                            validation_data=(val_sample_array, val_label_array), verbose=0,
                            callbacks=[lr_scheduler, EarlyStopping(monitor='val_loss', min_delta=0, patience=pt, verbose=0,
                                                     mode='min'),
                                       ModelCheckpoint(model_temp_path, monitor='val_loss',
                                                       save_best_only=True, mode='min', verbose=0)]
                            )


        test_pred = model.predict(test_sample_array)
        test_pred = test_pred.flatten()
        rms = sqrt(mean_squared_error(test_pred, test_label_array))
        rms = round(rms, 4)
        end_itr = time.time()
        training_time = end_itr - start_itr
        num_tran_params = train_params_count(model)
        # flop = get_flops(model)

        val_pred = model.predict(val_sample_array)
        val_pred = val_pred.flatten()
        val_rms = sqrt(mean_squared_error(val_pred, val_label_array))
        val_rms = round(val_rms, 4)

        test_rmse.append(rms)
        # val_rms_lst.append(val_rms)
        # flops.append(flop)
        train_params.append(num_tran_params)
        train_time.append(training_time)

        # archt_scores.append(archt_score)

        print ("ind['fitness_1']: ", ind['fitness_1'])
        print ("val_rms: ", val_rms)
        print ("rms: ", rms)



########
    # append columns
    selected_df['test_rmse'] = test_rmse
    # selected_df['val_rmse'] = val_rms_lst
    # mute_log_df['flops'] = flops
    selected_df['train_params'] = train_params
    selected_df['train_time'] = train_time

    # mute_log_df['archt_scores'] = archt_scores

    # Save to csv
    new_file_path = os.path.join(directory_path, 'mute_log_%s_%s_soo_%s_test.csv' %(pop,gen,trial))
    selected_df.to_csv(new_file_path, index=False)

    print ("Test results are saved")

if __name__ == '__main__':
    main()
