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
from scipy.stats import truncnorm



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
from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform

from utils.dnn import one_dcnn

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'



# Ignore tf err log
pd.options.mode.chained_assignment = None  # default='warn'

current_dir = os.path.dirname(os.path.abspath(__file__))
# data_filepath = os.path.join(current_dir, 'dati_bluetensor.xlsx')

pic_dir = os.path.join(current_dir, 'Curves')
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


all_models = {}

# def pow3(x, c, a, alpha):
#     return  c - a * x**(-alpha)
# all_models["pow3"] = pow3

# def loglog_linear(x, a, b, c):
#     # x = x+1
#     x = np.log(x)
#     return -1*np.log(np.abs(a*x - b))+c
# all_models["loglog_linear"] = loglog_linear

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

def reverse_gompertz(x, a, c, b, d):

    return a + (c-a)*(1-np.exp(-np.exp(-b*(x-d))))
    #return a + b * np.exp(np.exp(-k*(x-i)))
all_models["reverse_gompertz"] = reverse_gompertz


def hill_custom(x, a, b, c, d):
    return a + (b-a)/(1 + (10**(x-c))**d)
all_models["hill_custom"] = hill_custom

def log_custom(x, a, b, c, d):
    # return a * (np.log(x-b) / np.log(c) ) + d
    return a * np.log(x-c+1)  + d
all_models["log_custom"] = log_custom


# def logistic_curve(x, a, b, c,d):
#     """
#         a: asymptote
#         k: 
#         b: inflection point
#         http://www.pisces-conservation.com/growthhelp/logistic_curve.htm
#     """
#     return -1*a / (1+ np.exp(-b*(x-c))) +d
# all_models["logistic_curve"] = logistic_curve



# def reverse_sigmoid(x,a,b,c,d):
#     return a/(1+ np.exp(-b+c*x))+d
# all_models["reverse_sigmoid"] = reverse_sigmoid    



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
                 n_filters, kernel_size, n_mlp, batch, ob_ep, st_ep, epoch, patience, val_split, model_path, device):
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
        self.ob_ep = ob_ep
        self.st_ep = st_ep
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


    def train_net(self, model, lr, geno_lst, train_sample_array, train_label_array, val_sample_array, val_label_array):
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

        # Validation RMSE observation history
        numb_obeservation = self.ob_ep

        print ("self.n_layers", self.n_layers)
        print ("numb_obeservation", numb_obeservation)


        # Train each individual for observation epochs
        model.compile(loss='mean_squared_error', optimizer=amsgrad, metrics=['mae', keras_rmse ])
        history = model.fit(train_sample_array, train_label_array, epochs=numb_obeservation, batch_size=self.batch,
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
        with open('val_rmse_hist.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(val_rmse_hist)  

        x_max = np.arange(1,31)
        x_observation = x_max[:numb_obeservation]
        y_obesrvation = val_rmse_hist

        fig = matplotlib.figure.Figure(figsize=(8, 6))
        # agg.FigureCanvasAgg(fig)
        ax = fig.add_subplot(1, 1, 1)
        # Plot actual curve with solid line
        ax.plot(x_observation, y_obesrvation, label="observation", color="black", linewidth=2, zorder=4)
        color = iter(cm.rainbow(np.linspace(0, 1, len(all_models)+1)))

        y_func_lst = []
        curve_y_lst = []
        y_func_end_lst = []

        numb_obeservation_default = numb_obeservation
        
        # X = get_truncated_normal(mean=0.1, sd=0.02, low=0.0, upp=0.2)
        # X = get_truncated_normal(mean=0.2, sd=0.05, low=0.0, upp=0.3)
        print ("min(val_rmse_hist)", min(val_rmse_hist))
        print ("1/min(val_rmse_hist)", 1/min(val_rmse_hist))
        X = get_truncated_normal(mean=1/min(val_rmse_hist), sd=0.02, low=0.0, upp=0.2)
        noise = X.rvs(2)



        estimation1 = min(val_rmse_hist)- noise[0]
        estimation2 = estimation1 - noise[1]
        # estimation3 = estimation2 - noise[2]
        
        y_obesrvation.append(estimation1)
        y_obesrvation.append(estimation2)
        # y_obesrvation.append(estimation3)

        print (noise)
        print ("estimation1", estimation1)
        print ("estimation2", estimation2)
        # print ("estimation3", estimation3)
        print ("y_obesrvation", y_obesrvation)

        numb_obeservation = numb_obeservation + 2
        # numb_obeservation = numb_obeservation + 3


        x_observation = x_max[self.st_ep:numb_obeservation]
        print ("len(x_observation)", len(x_observation))
        y_obesrvation = y_obesrvation[self.st_ep:]

        ax.scatter([ numb_obeservation-1, numb_obeservation], [estimation1, estimation2], facecolor="black", edgecolors=(0.0, 0.0, 0.0), zorder=4, s=20,  label="Estimation" )
        # ax.scatter([numb_obeservation-2, numb_obeservation-1, numb_obeservation], [estimation1, estimation2, estimation3], facecolor="black", edgecolors=(0.0, 0.0, 0.0), zorder=4, s=20,  label="Estimation" )

  
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
            
            print ("next_y", next_y)

            y_func_end_lst.append(next_y[-1])
            y_func_lst.append(y_func)
            curve_y_lst.append(np.append(y_func, next_y))
            # next_y_lst.append(next_y)
            
            # Plot extrapolation with red circles
            # ax.plot(np.append(y, next_y), 'ro')
            c = next(color)
            # ax.plot(next_x, next_y, color=c, marker='o', label=key)

            # ax.plot(x_max, np.append(y_func, next_y), color=c, marker='o', linestyle='dashed', linewidth=2, label=key, zorder=1)
            ax.plot(x_max[self.st_ep:], np.append(y_func, next_y), color=c, marker='o', linestyle='dashed', linewidth=2, label=key, zorder=1)





        # list of 1d arrays to 2d array
        print ("y_func_lst", y_func_lst)
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
            # rms = 30.0

            c = next(color)
            # ax.plot(x_max, combined_y, color=c, marker='D', linewidth=1.5, label="combined", zorder=2)
            ax.scatter(x_max[-1], rms, color=c, label="combined", zorder=2)

            ax.legend(loc='upper right', fontsize=12)
            ax.set_xlabel('Epoch', fontsize=15)
            ax.set_ylabel('Validation RMSE', fontsize=15)
            x_epoch = np.arange(1,31)
            ax.set_xticks(x_epoch)
            ax.set_xticklabels(x_epoch, rotation=60)
            ymax_plot = 25
            ax.set_ylim(0, ymax_plot)
            ax.vlines(numb_obeservation_default,  0, ymax_plot, colors=(0.7, 0.7, 0.7), linestyle='-.',linewidth=1, zorder=3)

            fig.suptitle('Last ob: %s, Estimated validation RSME: %s' %(round(y_obesrvation[-3],4),rms), fontsize=12)
            # fig.suptitle('Last ob: %s, Estimated validation RSME: %s' %(round(y_obesrvation[-4],4),rms), fontsize=12)
            fig.savefig(os.path.join(pic_dir, 'curve_extpl_%s.png' %str(geno_lst)), bbox_inches='tight')


        else:
            # Combine (linear combination) of curves
            curves_arrays = np.transpose(np.stack(curve_y_lst, axis=0))
            print("curves_arrays.shape", curves_arrays.shape)
            combined_y = curves_arrays  @ coeff.value 
            print ("combined_y", combined_y)



            c = next(color)
            # ax.plot(x_max, combined_y, color=c, marker='D', linewidth=1.5, label="combined", zorder=2)

            extpl_rmse = combined_y[-1]
            print ("combined_y[-1]", combined_y[-1])

            if abs(y_obesrvation[-1] - combined_y[-1])>=1.5:
                rms = sum(y_func_end_lst) / len(y_func_end_lst)
                ax.scatter(x_max[-1], rms, color=c, label="combined", zorder=2)

            else:
                rms = round(extpl_rmse, 4)
                ax.plot(x_max[self.st_ep:], combined_y, color=c, marker='D', linewidth=1.5, label="combined", zorder=2)

            ax.legend(loc='upper right', fontsize=12)
            ax.set_xlabel('Epoch', fontsize=15)
            ax.set_ylabel('Validation RMSE', fontsize=15)
            x_epoch = np.arange(1,31)
            ax.set_xticks(x_epoch)
            ax.set_xticklabels(x_epoch, rotation=60)
            ymax_plot = 25
            ax.set_ylim(0, ymax_plot)
            ax.vlines(numb_obeservation_default,  0, ymax_plot, colors=(0.7, 0.7, 0.7), linestyle='-.',linewidth=1, zorder=3)


            # if (extpl_rmse <= 5.0) or (abs(y_obesrvation[-1] - combined_y[-1])>=3):
            #     rms = 30.0
            # else:
            #     rms = round(extpl_rmse, 4)




            fig.suptitle('Last ob: %s, Estimated validation RSME: %s' %(round(y_obesrvation[-3],4),rms), fontsize=12)
            # fig.suptitle('Last ob: %s, Estimated validation RSME: %s' %(round(y_obesrvation[-4],4),rms), fontsize=12)
            fig.savefig(os.path.join(pic_dir, 'curve_extpl_%s.png' %str(geno_lst)), bbox_inches='tight')


        val_net = (rms,)
        end_itr = time.time()
        train_time = end_itr - start_itr

        num_tran_params = train_params_count(model)
        print("number of trainable parameters: ", num_tran_params )
        print("training network is successfully completed, time: ", train_time)
        print("val_net extrapolation in rmse: ", val_net[0])


        model = None
        val_pred = None
        del model, val_pred
 
        with open('train_time_hist.csv', 'a') as fq:
            writer = csv.writer(fq)
            writer.writerow([train_time])   


        return val_net, num_tran_params, train_time


