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

def roundup(x):
  return int(math.ceil(x / 100.0)) * 100

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
    parser.add_argument('--bs', type=int, default=256, required=False, help='generations of evolution')

    args = parser.parse_args()

    trial = args.t
    pop = args.pop
    gen = args.gen
    bs = args.bs

    # Load csv file
    # new_file_path = os.path.join(directory_path, 'mute_log_%s_%s_soo_%s_score.csv' %(pop,gen,trial))
    new_file_path = os.path.join(directory_path, 'mute_log_%s_%s_%s_soo_%s_score.csv' %(pop,gen,bs,trial))
    test_file_path = os.path.join(directory_path, 'mute_log_%s_%s_soo_%s_test.csv' %(pop,gen,trial))
    mute_log_df = pd.read_csv(new_file_path)
    test_log_df = pd.read_csv(test_file_path)


    y_sp = 100
    ref_avg = 20
####################################
    # Draw scatter plot
    fig = matplotlib.figure.Figure(figsize=(3, 3))
    agg.FigureCanvasAgg(fig)
    cmap = get_cmap(10)
    ax = fig.add_subplot(1, 1, 1)
    # Draw scatter plot

    x_min = int(min(mute_log_df['fitness_1'])) - 0.5
    x_max = int(max(mute_log_df['fitness_1'])) + 0.5
    x_sp = 0.25
    # y_min = min(mute_log_df['archt_scores']) - 100
    sc = mute_log_df['archt_scores'].values
    # print (sc)
    # print (type(sc))
    # print ("np.min(sc[np.nonzero(sc)])", np.min(sc[np.nonzero(sc)]))
    if np.min(sc[np.nonzero(sc)]) - 100 <= 100:
        y_min = 100
    else:
        y_min = np.min(sc[np.nonzero(sc)]) - 100
    # y_min = 400
    y_max = roundup(max(mute_log_df['archt_scores'])) + 100
    # y_sp = 100
    # y_sp = (y_max - y_min)/20
    x_range = np.arange(x_min, x_max, 2 * x_sp)

    # ax.scatter(mute_log_df['fitness_1'], mute_log_df['test_rmse'], facecolor=(1.0, 1.0, 0.4), edgecolors=(0.0, 0.0, 0.0), zorder=1,
    #            c=cmap(0), s=20 )

    ax.scatter(mute_log_df['fitness_1'], mute_log_df['archt_scores'], facecolor=(1.0, 1.0, 0.4),
               edgecolors=(0.0, 0.0, 0.0), zorder=1, s=20 )


    ax.set_xticks(x_range)
    ax.set_xticklabels(x_range, rotation=60)
    ax.set_yticks(np.arange(y_min, y_max, 2 * y_sp))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # ax.set_title("Solutions and pareto front", fontsize=15)
    ax.set_xlabel('Validation RMSE', fontsize=12)
    ax.set_ylabel('Architectuer score', fontsize=12)
    # ax.legend(fontsize=9)

    # Save figure
    # ax.set_rasterized(True)
    fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s_%s.png' % (pop, gen, bs, trial)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.eps' % (pop, gen, trial)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.pdf' % (pop, gen, trial)), bbox_inches='tight')


###############################
    # Draw scatter plot
    fig = matplotlib.figure.Figure(figsize=(3, 3))
    agg.FigureCanvasAgg(fig)
    cmap = get_cmap(10)
    ax = fig.add_subplot(1, 1, 1)
    x_min = int(min(mute_log_df['fitness_1'])) - 0.5
    x_max = int(max(mute_log_df['fitness_1'])) + 0.5
    # x_sp = 0.25
    # y_min = min(mute_log_df['archt_scores']) - 100
    sc = mute_log_df['archt_scores'].values
    # print (sc)
    # print (type(sc))
    # print ("np.min(sc[np.nonzero(sc)])", np.min(sc[np.nonzero(sc)]))
    if np.min(sc[np.nonzero(sc)]) - 100 <= 100:
        y_min = 100
    else:
        y_min = np.min(sc[np.nonzero(sc)]) - 100
    # y_min = 400
    y_max = roundup(max(mute_log_df['archt_scores'])) + 100
    # y_sp = y_sp
    # y_sp = (y_max - y_min)/20
    x_range = np.arange(x_min, x_max, 2 * x_sp)

    # ax.scatter(mute_log_df['fitness_1'], mute_log_df['test_rmse'], facecolor=(1.0, 1.0, 0.4), edgecolors=(0.0, 0.0, 0.0), zorder=1,
    #            c=cmap(0), s=20 )
    # Calculate score avg every 100
    start_value = roundup(max(mute_log_df['archt_scores']))
    mean_scores = []
    mean_vals = []
    std_scores = []
    std_vals = []
    # ref_avg = ref_avg
    interval = y_sp
    for n in range(ref_avg):
        selected_df = mute_log_df.loc[(mute_log_df['archt_scores'] < start_value - n*interval) &
                        (mute_log_df['archt_scores'] > start_value - (n+1)*interval)]
        mean_score = np.mean(selected_df['archt_scores'].values)
        mean_val = np.mean(selected_df['fitness_1'].values)
        std_score = np.std(selected_df['archt_scores'].values)
        std_val = np.std(selected_df['fitness_1'].values)
        mean_scores.append(mean_score)
        mean_vals.append(mean_val)
        std_scores.append(std_score)
        std_vals.append(std_val)


    # ax.scatter(mean_vals, mean_scores, facecolor=(1.0, 0.4, 0.4),
    #            edgecolors=(0.0, 0.0, 0.0), zorder=2, s=30 )

    for i in range(len(mean_vals)):
        ax.errorbar(mean_vals[i], mean_scores[i], xerr=std_vals[i], color='red', ecolor='red', marker="s", ms=3,
                     elinewidth=1, capsize=2, capthick=1)

    ax.hlines(np.arange(start_value - ref_avg*interval, start_value, interval), x_min, x_max, colors=(0.1, 0.1, 0.1, 0.1), zorder=2)

    ax.scatter(mute_log_df['fitness_1'], mute_log_df['archt_scores'], facecolor=(1.0, 1.0, 0.4),
               edgecolors=(0.0, 0.0, 0.0), zorder=1, s=20 )


    ax.set_xticks(x_range)
    ax.set_xticklabels(x_range, rotation=60)
    ax.set_yticks(np.arange(y_min, y_max, 2 * y_sp))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # ax.set_title("Solutions and pareto front", fontsize=15)
    ax.set_xlabel('Validation RMSE', fontsize=12)
    ax.set_ylabel('Architectuer score', fontsize=12)
    # ax.legend(fontsize=9)

    # Save figure
    # ax.set_rasterized(True)
    fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s_%s_avg.png' % (pop, gen, bs, trial)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.eps' % (pop, gen, trial)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.pdf' % (pop, gen, trial)), bbox_inches='tight')
    print ("mean_vals", mean_vals)


############################à

    fig = matplotlib.figure.Figure(figsize=(3, 3))
    agg.FigureCanvasAgg(fig)
    cmap = get_cmap(10)
    ax = fig.add_subplot(1, 1, 1)
    # Draw scatter plot

    x_min = int(min(test_log_df['test_rmse'])) - 0.5
    x_max = int(max(test_log_df['test_rmse'])) + 0.5
    x_sp = 0.25
    # y_min = min(mute_log_df['archt_scores']) - 100
    sc = mute_log_df['archt_scores'].values
    # print (sc)
    # print (type(sc))
    # print ("np.min(sc[np.nonzero(sc)])", np.min(sc[np.nonzero(sc)]))
    if np.min(sc[np.nonzero(sc)]) - 100 <= 100:
        y_min = 100
    else:
        y_min = np.min(sc[np.nonzero(sc)]) - 100
    # y_min = 400
    y_max = roundup(max(mute_log_df['archt_scores'])) + 100
    # y_sp = 100
    # y_sp = (y_max - y_min)/20
    x_range = np.arange(x_min, x_max, 2 * x_sp)

    # ax.scatter(mute_log_df['fitness_1'], mute_log_df['test_rmse'], facecolor=(1.0, 1.0, 0.4), edgecolors=(0.0, 0.0, 0.0), zorder=1,
    #            c=cmap(0), s=20 )

    ax.scatter(test_log_df['test_rmse'], mute_log_df['archt_scores'], facecolor=(1.0, 1.0, 0.4),
               edgecolors=(0.0, 0.0, 0.0), zorder=1, s=20 )


    ax.set_xticks(x_range)
    ax.set_xticklabels(x_range, rotation=60)
    ax.set_yticks(np.arange(y_min, y_max, 2 * y_sp))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # ax.set_title("Solutions and pareto front", fontsize=15)
    ax.set_xlabel('Test RMSE', fontsize=12)
    ax.set_ylabel('Architectuer score', fontsize=12)
    # ax.legend(fontsize=9)

    # Save figure
    # ax.set_rasterized(True)
    fig.savefig(os.path.join(pic_dir, 'test_score_%s_%s_%s_%s.png' % (pop, gen, bs, trial)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.eps' % (pop, gen, trial)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.pdf' % (pop, gen, trial)), bbox_inches='tight')


##################################à
    # Draw scatter plot
    mute_log_df['test_rmse'] = test_log_df['test_rmse']

    fig = matplotlib.figure.Figure(figsize=(3, 3))
    agg.FigureCanvasAgg(fig)
    cmap = get_cmap(10)
    ax = fig.add_subplot(1, 1, 1)
    x_min = int(min(mute_log_df['test_rmse'])) - 0.5
    x_max = int(max(mute_log_df['test_rmse'])) + 0.5
    x_sp = 0.25
    # y_min = min(mute_log_df['archt_scores']) - 100
    sc = mute_log_df['archt_scores'].values
    # print (sc)
    # print (type(sc))
    # print ("np.min(sc[np.nonzero(sc)])", np.min(sc[np.nonzero(sc)]))
    if np.min(sc[np.nonzero(sc)]) - 100 <= 100:
        y_min = 100
    else:
        y_min = np.min(sc[np.nonzero(sc)]) - 100
    # y_min = 400
    y_max = roundup(max(mute_log_df['archt_scores'])) + 100
    # y_sp = y_sp
    # y_sp = (y_max - y_min)/20
    x_range = np.arange(x_min, x_max, 2 * x_sp)

    # ax.scatter(mute_log_df['fitness_1'], mute_log_df['test_rmse'], facecolor=(1.0, 1.0, 0.4), edgecolors=(0.0, 0.0, 0.0), zorder=1,
    #            c=cmap(0), s=20 )
    # Calculate score avg every 100
    start_value = roundup(max(mute_log_df['archt_scores']))
    mean_scores = []
    mean_vals = []
    std_scores = []
    std_vals = []
    # ref_avg = ref_avg
    interval = y_sp



    for n in range(ref_avg):
        selected_df = mute_log_df.loc[(mute_log_df['archt_scores'] < start_value - n*interval) &
                        (mute_log_df['archt_scores'] > start_value - (n+1)*interval)]
        mean_score = np.mean(selected_df['archt_scores'].values)
        mean_val = np.mean(selected_df['test_rmse'].values)
        std_score = np.std(selected_df['archt_scores'].values)
        std_val = np.std(selected_df['test_rmse'].values)
        mean_scores.append(mean_score)
        mean_vals.append(mean_val)
        std_scores.append(std_score)
        std_vals.append(std_val)

    #
    # ax.scatter(mean_vals, mean_scores, facecolor=(1.0, 0.4, 0.4),
    #            edgecolors=(0.0, 0.0, 0.0), zorder=2, s=30 )

    for i in range(len(mean_vals)):
        ax.errorbar(mean_vals[i], mean_scores[i], xerr=std_vals[i], color='red', ecolor='red', marker="s", ms=3,
                     elinewidth=1, capsize=2, capthick=1)

    ax.hlines(np.arange(start_value - ref_avg*interval, start_value, interval), x_min, x_max, colors=(0.1, 0.1, 0.1, 0.1), zorder=2)

    ax.scatter(mute_log_df['test_rmse'], mute_log_df['archt_scores'], facecolor=(1.0, 1.0, 0.4),
               edgecolors=(0.0, 0.0, 0.0), zorder=1, s=20 )


    ax.set_xticks(x_range)
    ax.set_xticklabels(x_range, rotation=60)
    ax.set_yticks(np.arange(y_min, y_max, 2 * y_sp))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # ax.set_title("Solutions and pareto front", fontsize=15)
    ax.set_xlabel('Test RMSE', fontsize=12)
    ax.set_ylabel('Architectuer score', fontsize=12)
    # ax.legend(fontsize=9)

    # Save figure
    # ax.set_rasterized(True)
    fig.savefig(os.path.join(pic_dir, 'test_score_%s_%s_%s_%s_avg.png' % (pop, gen, bs, trial)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.eps' % (pop, gen, trial)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.pdf' % (pop, gen, trial)), bbox_inches='tight')
    print ("mean_vals", mean_vals)




    ############################à


    fig = matplotlib.figure.Figure(figsize=(3, 3))
    agg.FigureCanvasAgg(fig)
    cmap = get_cmap(10)
    ax = fig.add_subplot(1, 1, 1)
    # Draw scatter plot

    x_min = int(min(mute_log_df['test_rmse'])) - 0.5
    x_max = int(max(mute_log_df['test_rmse'])) + 0.5
    x_sp = 0.25

    y_max =int(max(mute_log_df['fitness_1'])) + 1
    y_min =int(min(mute_log_df['fitness_1'])) - 1
    y_sp = 0.25



    # y_sp = 100
    # y_sp = (y_max - y_min)/20
    x_range = np.arange(x_min, x_max, 2 * x_sp)

    # ax.scatter(mute_log_df['fitness_1'], mute_log_df['test_rmse'], facecolor=(1.0, 1.0, 0.4), edgecolors=(0.0, 0.0, 0.0), zorder=1,
    #            c=cmap(0), s=20 )

    ax.scatter(mute_log_df['test_rmse'], mute_log_df['fitness_1'], facecolor=(1.0, 1.0, 0.4),
               edgecolors=(0.0, 0.0, 0.0), zorder=1, s=20 )


    ax.set_xticks(x_range)
    ax.set_xticklabels(x_range, rotation=60)
    ax.set_yticks(np.arange(y_min, y_max, 2 * y_sp))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # ax.set_title("Solutions and pareto front", fontsize=15)
    ax.set_xlabel('Test RMSE', fontsize=12)
    ax.set_ylabel('Validation RMSE', fontsize=12)
    # ax.legend(fontsize=9)

    # Save figure
    # ax.set_rasterized(True)
    fig.savefig(os.path.join(pic_dir, 'val_test_%s_%s_%s.png' % (pop, gen, trial)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.eps' % (pop, gen, trial)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.pdf' % (pop, gen, trial)), bbox_inches='tight')



    #############################


    mute_log_df['test_rmse'] = test_log_df['test_rmse']

    fig = matplotlib.figure.Figure(figsize=(3, 3))
    agg.FigureCanvasAgg(fig)
    cmap = get_cmap(10)
    ax = fig.add_subplot(1, 1, 1)
    x_min = int(min(mute_log_df['test_rmse'])) - 0.5
    x_max = int(max(mute_log_df['test_rmse'])) + 0.5
    x_sp = 0.25

    y_max =int(max(mute_log_df['fitness_1'])) + 1
    y_min =int(min(mute_log_df['fitness_1'])) - 1
    y_sp_rmse = 0.25
    x_range = np.arange(x_min, x_max, 2 * x_sp)

    # ax.scatter(mute_log_df['fitness_1'], mute_log_df['test_rmse'], facecolor=(1.0, 1.0, 0.4), edgecolors=(0.0, 0.0, 0.0), zorder=1,
    #            c=cmap(0), s=20 )
    # Calculate score avg every 100
    start_value = int(max(mute_log_df['fitness_1'])) + 1
    mean_scores = []
    mean_vals = []
    std_scores = []
    std_vals = []
    # ref_avg = ref_avg
    ref_avg = 30
    interval = y_sp_rmse



    for n in range(ref_avg):
        selected_df = mute_log_df.loc[(mute_log_df['fitness_1'] < start_value - n*interval) &
                        (mute_log_df['fitness_1'] > start_value - (n+1)*interval)]
        mean_score = np.mean(selected_df['fitness_1'].values)
        mean_val = np.mean(selected_df['test_rmse'].values)
        std_score = np.std(selected_df['fitness_1'].values)
        std_val = np.std(selected_df['test_rmse'].values)
        mean_scores.append(mean_score)
        mean_vals.append(mean_val)
        std_scores.append(std_score)
        std_vals.append(std_val)


    # ax.scatter(mean_vals, mean_scores, facecolor=(1.0, 0.4, 0.4),
    #            edgecolors=(0.0, 0.0, 0.0), zorder=2, s=30 )

    for i in range(len(mean_vals)):
        ax.errorbar(mean_vals[i], mean_scores[i], xerr=std_vals[i], color='red', ecolor='red', marker="s", ms=3,
                     elinewidth=1, capsize=2, capthick=1)

    ax.hlines(np.arange(start_value - ref_avg*interval, start_value, interval), x_min, x_max, colors=(0.1, 0.1, 0.1, 0.1), zorder=2)

    ax.scatter(mute_log_df['test_rmse'], mute_log_df['fitness_1'], facecolor=(1.0, 1.0, 0.4),
               edgecolors=(0.0, 0.0, 0.0), zorder=1, s=20 )


    ax.set_xticks(x_range)
    ax.set_xticklabels(x_range, rotation=60)
    ax.set_yticks(np.arange(y_min, y_max, 2 * y_sp))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # ax.set_title("Solutions and pareto front", fontsize=15)
    ax.set_xlabel('Test RMSE', fontsize=12)
    ax.set_ylabel('Validation RMSE', fontsize=12)
    # ax.legend(fontsize=9)

    # Save figure
    # ax.set_rasterized(True)
    fig.savefig(os.path.join(pic_dir, 'val_test_%s_%s_%s_avg.png' % (pop, gen, trial)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.eps' % (pop, gen, trial)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.pdf' % (pop, gen, trial)), bbox_inches='tight')
    print ("mean_vals", mean_vals)







    ###########################

    print ("Plot save")


if __name__ == '__main__':
    main()
