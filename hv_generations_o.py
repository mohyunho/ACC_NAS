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

pop_size = 20
n_generations = 10

current_dir = os.path.dirname(os.path.abspath(__file__))

pic_dir = os.path.join(current_dir, 'Figures')
# Log file path of EA in csv
ea_log_path = os.path.join(current_dir, 'EA_log_o')

scale = 100

def roundup(x, scale):
    return int(math.ceil(x / float(scale))) * scale

def rounddown(x, scale):
    return int(math.floor(x / float(scale))) * scale


cycol = cycle('bgrcmk')

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)



cmap = get_cmap(6)


pd.options.mode.chained_assignment = None  # default='warn'



results_lst = []
prft_lst = []
hv_trial_lst = []
prft_trial_lst = []
########################################
for file in sorted(os.listdir(ea_log_path)):
    if file.startswith("mute_log_ori_20_10_"):
        print ("path1: ", file)
        mute_log_df = pd.read_csv(os.path.join(ea_log_path, file))
        results_lst.append(mute_log_df)


for loop_idx in range(len(results_lst)):
    print ("file %s in progress..." %loop_idx)
    mute_log_df = results_lst[loop_idx]

    ####################
    hv_lst = []
    for gen in mute_log_df['gen'].unique():
        if gen == 0:
            continue
        else:
            print ("gen", gen)
            hv_temp = mute_log_df.loc[mute_log_df['gen'] == gen]['hypervolume'].values
            hv_value = sum(hv_temp) / len(hv_temp)
            hv_lst.append(hv_value)
    print (hv_lst)
    offset_hv = [x - min(hv_lst) for x in hv_lst]
    norm_hv = [x / (max(offset_hv) + 1) for x in offset_hv]
    print ("norm_hv", norm_hv)
    hv_trial_lst.append(norm_hv)
    # print(norm_hv)


hv_gen = np.stack(hv_trial_lst)
hv_gen_lst = []
for gen in range(hv_gen.shape[1]):
    hv_temp =hv_gen[:,gen]
    hv_gen_lst.append(hv_temp)

print ("hv_gen", hv_gen)

# print (hv_gen_lst)
# print (len(hv_gen_lst))
fig_verify = plt.figure(figsize=(3, 3))
mean_hv = np.array([np.mean(a) for a in hv_gen_lst])
std_hv = np.array([np.std(a) for a in hv_gen_lst])
x_ref = range(1, n_generations + 1)
for trial in range(5):
    plt.plot(x_ref, hv_trial_lst[trial], color=cmap(trial), linewidth=0.5, zorder=3, label ='seed %s' %trial)


plt.plot(x_ref, mean_hv, color='black', linewidth=2, label = 'Mean', zorder=5)

plt.fill_between(x_ref, mean_hv-std_hv, mean_hv+std_hv,
    zorder=1, alpha=0.2, facecolor=(1.0, 0.8, 0.8))

plt.plot(x_ref, mean_hv-std_hv, color='black', zorder=2, linewidth= 0.5, linestyle='dashed')
plt.plot(x_ref, mean_hv+std_hv, color='black', zorder=2, linewidth= 0.5, linestyle='dashed', label = 'Std')
plt.xticks(x_ref, fontsize=8)
plt.yticks(fontsize=9)
plt.ylabel("Normalized hypervolume", fontsize=9)
plt.xlabel("Generations", fontsize=9)
plt.legend(loc='lower right', fontsize=7)
plt.xlim(0,11)
fig_verify.savefig(os.path.join(pic_dir, 'hv_plot_ori_%s_%s.png' % (pop_size, n_generations)), dpi=1500,
                   bbox_inches='tight')
fig_verify.savefig(os.path.join(pic_dir, 'hv_plot_ori_%s_%s.eps' % (pop_size, n_generations)), dpi=1500,
                   bbox_inches='tight')


