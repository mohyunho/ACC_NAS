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


cycol = cycle('bgrcmk')

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='NAS CNN')
    parser.add_argument('-t', type=int, required=True, help='trial')
    parser.add_argument('--pop', type=int, default=50, required=False, help='population size of EA')
    parser.add_argument('--gen', type=int, default=50, required=False, help='generations of evolution')

    args = parser.parse_args()

    trial = args.t
    pop = args.pop
    gen = args.gen

    # Load csv file
    new_file_path = os.path.join(directory_path, 'mute_log_%s_%s_soo_%s_test.csv' %(pop,gen,trial))
    mute_log_df = pd.read_csv(new_file_path)

    # Draw scatter plot
    fig = matplotlib.figure.Figure(figsize=(3, 3))
    agg.FigureCanvasAgg(fig)
    cmap = get_cmap(len(prft_trial_lst))
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(prft[col_a], prft[col_b], facecolor=(1.0, 1.0, 0.4), edgecolors=(0.0, 0.0, 0.0), zorder=1,
               c=cmap(0), s=20, )

    x_range = np.arange(x_min, x_max, 2 * x_sp)
    ax.set_xticks(x_range)
    ax.set_xticklabels(x_range, rotation=60)
    ax.set_yticks(np.arange(y_min, y_max, 2 * y_sp))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max)
    # ax.set_title("Solutions and pareto front", fontsize=15)
    ax.set_xlabel('Validation RMSE', fontsize=12)
    ax.set_ylabel('Test RMSE', fontsize=12)
    ax.legend(fontsize=9)

    # Save figure
    # ax.set_rasterized(True)
    fig.savefig(os.path.join(pic_dir, 'prft_aggr_%s_%s.png' % (pop_size, n_generations)), dpi=1500, bbox_inches='tight')
    fig.savefig(os.path.join(pic_dir, 'prft_aggr_%s_%s.eps' % (pop_size, n_generations)), dpi=1500, bbox_inches='tight')
    fig.savefig(os.path.join(pic_dir, 'prft_aggr_%s_%s.pdf' % (pop_size, n_generations)), bbox_inches='tight')



if __name__ == '__main__':
    main()
