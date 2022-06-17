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

pic_dir = os.path.join(current_dir, 'PF')

log_dir_path = os.path.join(current_dir, 'EA_log')

if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)


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


    prft_filename = os.path.join(log_dir_path, 'inference_prft_%s_%s_%s_%s_%s_%s.csv' % (ablation, pop_size, n_generations, obj, trial, ep))

    prft_infer_df = pd.read_csv(prft_filename)

    points_val = []
    points_test = []

    for idx, row in prft_infer_df.iterrows():
        points_val.append([row['val_rmse'], row['num_params']])
        points_test.append([row['RMSE'], row['num_params']])



    # ref = [15.0, 20.0]
    ref = [15, 15.0]

    hv_val = hypervolume(points = points_val)
    result_val = hv_val.compute(ref_point = ref)
    print ("hypervolume_validation", result_val)

    hv_test = hypervolume(points = points_test)
    result_test = hv_test.compute(ref_point = ref)
    print ("hypervolume_test", result_test)



######################
    initial_pop_df = pd.read_csv(os.path.join(log_dir_path, 'inference_prft_%s_%s_0_moo_%s_%s.csv' %(ablation, pop_size, trial,ep)))
    initial_pop_val = initial_pop_df['val_rmse'].values
    initial_pop_params = initial_pop_df['num_params'].values
    initial_pop_rmse = initial_pop_df['RMSE'].values




################### pareto front plot ###############
    # Draw scatter plot
    fig = matplotlib.figure.Figure(figsize=(5, 5))
    agg.FigureCanvasAgg(fig)
    # cmap = get_cmap(10)
    ax = fig.add_subplot(1, 1, 1)
    # Draw scatter plot

    prft_rmse = prft_infer_df['val_rmse'].values
    prft_params = prft_infer_df['num_params'].values 
    
    print ("prft_params", prft_params)

    # x_min = int(min(prft_rmse)) - 0.5
    # x_max = int(max(prft_rmse)) + 0.5
    # x_sp = 0.25
    # x_range = np.arange(x_min, x_max, 2 * x_sp)

    # ax.scatter(mute_log_df['fitness_1'], mute_log_df['test_rmse'], facecolor=(1.0, 1.0, 0.4), edgecolors=(0.0, 0.0, 0.0), zorder=1,
    #            c=cmap(0), s=20 )

    ax.scatter(prft_rmse, prft_params, facecolor=(1.0, 1.0, 0.4),
               edgecolors=(0.0, 0.0, 0.0), zorder=3, s=60,  label="Solutions" )

    ax.scatter(initial_pop_val, initial_pop_params, facecolor=(0.0, 0.5, 0.0),
               edgecolors=(0.0, 0.0, 0.0), zorder=1, s=40,  label="Initial population" )

    ax.scatter(8.03, 1.6126, marker="D",facecolor=(0.0, 1.0, 0.0), edgecolors=(0.0, 0.0, 0.0), zorder=4, s=60, label="Handcrafted CNN")

    x_range = np.arange(5.0, 15.5, 0.5)
    ax.set_xticks(x_range)
    ax.set_xticklabels(x_range, rotation=60)
    ax.set_xlim(5, 16)

    y_range = np.arange(0, 21, 1)
    ax.set_yticks(y_range)
    ax.set_yticklabels(y_range)
    ax.set_ylim(0, 21)


    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)
    # ax.set_title("Solutions and pareto front", fontsize=15)
    ax.set_xlabel('Validation RMSE', fontsize=12)
    ax.set_ylabel(r'Trainable parameters $\times$ ($10^4$)', fontsize=12)
    ax.legend(fontsize=10, loc='center right')
    fig.suptitle('Hypervolume: %s' %result_val, fontsize=12)


    ax.vlines(ref[0],  0, ref[1], colors='black', linestyle='-.',linewidth=1, zorder=3)
    ax.hlines(ref[1],  0, ref[0], colors='black', linestyle='-.',linewidth=1, zorder=3)
    # Save figure
    # ax.set_rasterized(True)
    fig.savefig(os.path.join(pic_dir, 'infer_prft_val_%s_%s_%s_%s_%s.png' % (ablation, pop_size, n_generations, trial, ep)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.eps' % (pop, gen, trial)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.pdf' % (pop, gen, trial)), bbox_inches='tight')

##############################################

    # Draw scatter plot
    fig = matplotlib.figure.Figure(figsize=(5, 5))
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
               edgecolors=(0.0, 0.0, 0.0), zorder=3, s=60,  label="Solutions" )

    ax.scatter(initial_pop_rmse, initial_pop_params, facecolor=(0.0, 0.5, 0.0),
               edgecolors=(0.0, 0.0, 0.0), zorder=1, s=40,  label="Initial population" )

    ax.scatter(7.49, 1.6126, marker="D",facecolor=(0.0, 1.0, 0.0), edgecolors=(0.0, 0.0, 0.0), zorder=4,
           s=60, label="Handcrafted CNN")


    x_range = np.arange(5.0, 15.5, 0.5)
    ax.set_xticks(x_range)
    ax.set_xticklabels(x_range, rotation=60)
    ax.set_xlim(5, 16)

    y_range = np.arange(0, 21, 1)
    ax.set_yticks(y_range)
    ax.set_yticklabels(y_range)
    ax.set_ylim(0, 21)


    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)
    # ax.set_title("Solutions and pareto front", fontsize=15)
    ax.set_xlabel('Test RMSE', fontsize=12)
    ax.set_ylabel(r'Trainable parameters $\times$ ($10^4$)', fontsize=12)
    ax.legend(fontsize=10, loc='center right')
    fig.suptitle('Hypervolume: %s' %result_test, fontsize=12)

    ax.vlines(ref[0],  0, ref[1], colors='black', linestyle='-.',linewidth=1, zorder=3)
    ax.hlines(ref[1],  0, ref[0], colors='black', linestyle='-.',linewidth=1, zorder=3)

    # Save figure
    # ax.set_rasterized(True)
    fig.savefig(os.path.join(pic_dir, 'infer_prft_test_%s_%s_%s_%s_%s.png' % (ablation, pop_size, n_generations, trial, ep)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.eps' % (pop, gen, trial)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.pdf' % (pop, gen, trial)), bbox_inches='tight')



######################################






######################################



if __name__ == '__main__':
    main()
