'''
Created on April , 2021
@author:
'''

## Import libraries in python
import argparse
import time
import json
import logging
import sys
import os
import math
import pandas as pd
import numpy as np
from itertools import cycle

import seaborn as sns
import random
import importlib
from scipy.stats import randint, expon, uniform
import glob
# import tensorflow as tf
import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt

from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.pareto import pareto
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.backends.backend_agg as agg
import matplotlib.backends.backend_svg as svg

cycol = cycle('bgrcmk')

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

# tf.config.set_visible_devices([], 'GPU')

current_dir = os.path.dirname(os.path.abspath(__file__))
data_filedir = os.path.join(current_dir, 'N-CMAPSS')
data_filepath = os.path.join(current_dir, 'N-CMAPSS', 'N-CMAPSS_DS02-006.h5')
sample_dir_path = os.path.join(data_filedir, 'Samples_whole')

model_temp_path = os.path.join(current_dir, 'Models', 'oned_cnn_rep.h5')
tf_temp_path = os.path.join(current_dir, 'TF_Model_tf')

pic_dir = os.path.join(current_dir, 'PF')

log_dir_path = os.path.join(current_dir, 'EA_log')



def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='sample creator')
    parser.add_argument('-ep', type=int, default=30, help='max epoch')
    parser.add_argument('--pop', type=int, default=20, required=False, help='population size of EA')
    parser.add_argument('--gen', type=int, default=10, required=False, help='generations of evolution')
    parser.add_argument('--seed', type=int, default=7, required=False, help='generations of evolution')


    args = parser.parse_args()
    ep = args.ep
    pop_size = args.pop
    n_generations = args.gen
    num_seeds = args.seed


    fig = matplotlib.figure.Figure(figsize=(3, 3))
    agg.FigureCanvasAgg(fig)
    cmap = get_cmap(num_seeds + 1)
    ax = fig.add_subplot(1, 1, 1)


    for trial in range(num_seeds):


        prft_score_filename = os.path.join(log_dir_path, 'inference_prft_score_%s_%s_moo_%s_30.csv' % (pop_size, n_generations, trial))
        prft_score_df = pd.read_csv(prft_score_filename)
        archt_score_lst = []
        test_rmse_lst = []
        for idx, row in prft_score_df.iterrows():
            # append test rmse
            test_rmse_lst.append(row['num_params'])

            # find archt score and append

            mut_log_filename =  os.path.join(log_dir_path, 'mute_log_score_%s_%s_moo_%s_30.csv' % (pop_size, n_generations, trial))
            mut_log_df = pd.read_csv(mut_log_filename)
            selected_archt_df = mut_log_df.loc[(mut_log_df['params_1']==row['p1'])&(mut_log_df['params_2']==row['p2'])&(mut_log_df['params_3']==row['p3'])&(mut_log_df['params_4']==row['p4'])]
            archt_score_temp = selected_archt_df['fitness_1'].values[-1]
            print (selected_archt_df)
            print ("archt_score_temp", archt_score_temp)

            if archt_score_temp >= 10:
                archt_score_temp = 10

            print ("archt_score_temp", archt_score_temp)  

            archt_score_lst.append(archt_score_temp)



        # ax.scatter(data[col_a], data[col_b], lw=0, facecolor=(0.7, 0.7, 0.7), zorder=-1, label="All solutions")
        ax.scatter(test_rmse_lst, archt_score_lst, facecolor=(1.0, 1.0, 0.4), edgecolors=(0.0, 0.0, 0.0), zorder=1, c=cmap(trial),
                s=40, label="seed %s" %trial, alpha=0.5)



    
    x_range = np.arange(1, 9)
    ax.set_xticks(x_range)
    # ax.set_xticklabels(x_range, rotation=60)
    ax.set_xticklabels(x_range)
    # ax.set_yticks(

    # ax.set_xlim(x_min,x_max)
    # ax.set_ylim(0,y_max)

    # ax.set_title("Solutions and pareto front", fontsize=15)
    ax.set_xlabel(r'Trainable parameters $\times$ ($10^4$)', fontsize=12)
    ax.set_ylabel('Architecture score', fontsize=12)
    ax.legend(fontsize=8)
    ax.vlines(5,  3.0, 11.0, colors='black', linestyle='-.',linewidth=1, zorder=3)
    # ax.set_rasterized(True)
    fig.savefig(os.path.join(pic_dir, 'prft_params-score_%s_%s.png' % (pop_size, n_generations)), dpi=1500, bbox_inches='tight')
    fig.savefig(os.path.join(pic_dir, 'prft_params-score_%s_%s.eps' % (pop_size, n_generations)), dpi=1500, bbox_inches='tight')
    fig.savefig(os.path.join(pic_dir, 'prft_params-score_%s_%s.pdf' % (pop_size, n_generations)), bbox_inches='tight')



if __name__ == '__main__':
    main()
