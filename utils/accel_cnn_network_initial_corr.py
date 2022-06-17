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
import csv 
import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.figure
import cvxpy as cp
from scipy.optimize import curve_fit
import cupy

from math import sqrt
# import keras

from scipy.optimize import curve_fit

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

from utils.dnn import one_dcnn

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


initializer = GlorotNormal(seed=0)

# Ignore tf err log
pd.options.mode.chained_assignment = None  # default='warn'

current_dir = os.path.dirname(os.path.abspath(__file__))
# data_filepath = os.path.join(current_dir, 'dati_bluetensor.xlsx')

pic_dir = os.path.join(current_dir, 'Corr')
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)

def func_lin(x, a, b):
    return x*a+b

def func_sqr(x, a, b, c, d):
    return x*x*x*a + x*x*b + x*c + d


def corr_coeff (individuals, fit_lst, batch_size, train_sample_array):

    archt_score_lst = []
    val_rmse_lst = []

    print ("individuals: ", individuals)
    print ("len(individuals): ", len(individuals))
    print ("fit_lst: ", fit_lst)
    print ("batch_size: ", batch_size)
    print ("train_sample_array.shape: ", train_sample_array.shape)


    for idx, ind in enumerate(individuals):
        print ("ind: ", ind)
        n_layers = ind[0]
        n_filters = ind[1]
        kernel_size = ind[2]
        n_mlp = 10 * ind[3]
        model = one_dcnn(n_layers, n_filters, kernel_size, n_mlp, train_sample_array, initializer)

        # Calculate model's score

        kmatrix = tf_net_kmatrix(model, batch_size, train_sample_array)

        # print ("output kmatrix: ", kmatrix)

        sign, archt_score = scorefunc_slogdet (kmatrix)
        print ("archt_score", archt_score)
        if int(sign) == 0:
            archt_score = 0

        print ("sign", sign)
        print ("archt_score", archt_score)

        if archt_score == 0:
            # continue
            archt_score_inverse = 10.0
        else:
            archt_score_div = archt_score / 10000.0
            archt_score_inverse = 1/archt_score_div

        if archt_score_inverse <= 0:
            continue
        elif archt_score_inverse >= 10:
            archt_score_inverse = 10.0
            # archt_score_inverse = archt_score_inverse * -1

        print ("archt_score_inverse", archt_score_inverse)
        archt_score_inverse = cupy.asnumpy(archt_score_inverse)
        archt_score_inverse = float(archt_score_inverse)
        print ("type(archt_score_inverse)", type(archt_score_inverse))

        #######

        archt_score_inverse = round(archt_score_inverse, 6)

        # Convert 'archt_score_inverse' to 'expected validation RMSE'


        archt_score_lst.append(archt_score_inverse)

        fitness_temp = fit_lst[idx]
        val_rmse_temp = fitness_temp[0]
        print ("val_rmse_temp", val_rmse_temp)
        val_rmse_lst.append(val_rmse_temp)



    print ("archt_score_lst", archt_score_lst)
    print ("val_rmse_lst", val_rmse_lst)

    ydata = np.asarray(val_rmse_lst, dtype=np.float64)
    xdata = np.asarray(archt_score_lst, dtype=np.float64)

    # ydata = val_rmse_lst
    # xdata = archt_score_lst
    # xdata = xdata.astype(np.float64)
    # ydata = ydata.astype(np.float64)

    fig = plt.figure(figsize=(7.2, 4.2))

    plt.plot(xdata, ydata, 'go', label='Data')

    popt, pcov = curve_fit(func_lin, xdata, ydata)
    # print (pcov)
    print (popt)
    # plt.plot(xdata, func_lin(xdata, *popt), 'b-', label='Linear: a=%5.2e, b=%5.2e' % tuple(popt))
    popt, pcov = curve_fit(func_sqr, xdata, ydata)
    # print (pcov)
    print (popt)

    x_sample = np.arange(4,11)
    x_low_sample = np.arange(3,5)
    x_high_sample = np.arange(10,12)

    min_val_rmse = func_sqr (x_sample[0], popt[0], popt[1], popt[2], popt[3])
    max_val_rmse = func_sqr (x_sample[-1], popt[0], popt[1], popt[2], popt[3])


    # min_val_rmse = func_sqr([4.0], *popt)
    # max_val_rmse = func_sqr([10.0], *popt)

    # plt.plot(xdata, func_sqr(xdata, *popt), 'r-', label='Best-fit curve, cubic: a=%5.2e, b=%5.2e, c=%5.2e, d=%5.2e' % tuple(popt))
    plt.plot(x_sample, func_sqr(x_sample, *popt), 'r-', label='Best-fit sample curve, cubic: a=%5.2e, b=%5.2e, c=%5.2e, d=%5.2e' % tuple(popt))
    plt.plot(x_low_sample, [min_val_rmse, min_val_rmse], 'r-')
    plt.plot(x_high_sample, [max_val_rmse,max_val_rmse], 'r-')
    plt.legend(fontsize=10)
    plt.ylabel("Validation RMSE", fontsize=15)
    plt.xlabel("Architecture score", fontsize=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    # plt.ylim([0,500])

    fig.savefig(os.path.join(pic_dir, 'corr_plot.png' ), dpi=1500 ,bbox_inches='tight')


    print ("popt",popt)
    print ("type(popt)", type(popt))




    return popt


