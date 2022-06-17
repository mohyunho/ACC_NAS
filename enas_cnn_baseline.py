'''
Created on Nov , 2021
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
import matplotlib.pyplot as plt
import seaborn as sns
import random
import importlib
from scipy.stats import randint, expon, uniform
import glob
import tensorflow as tf
import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt

from utils.cnn_network import network_fit

from utils.cnn_task import SimpleNeuroEvolutionTask
from utils.ea_multi import GeneticAlgorithm



current_dir = os.path.dirname(os.path.abspath(__file__))
data_filedir = os.path.join(current_dir, 'N-CMAPSS')
data_filepath = os.path.join(current_dir, 'N-CMAPSS', 'N-CMAPSS_DS02-006.h5')
sample_dir_path = os.path.join(data_filedir, 'Samples_whole')

model_temp_path = os.path.join(current_dir, 'Models', 'oned_cnn_rep_ori.h5')
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

def figsave(history, h1,h2,h3,h4, bs, lr, sub):
    fig_acc = plt.figure(figsize=(15, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training', fontsize=24)
    plt.ylabel('loss', fontdict={'fontsize': 18})
    plt.xlabel('epoch', fontdict={'fontsize': 18})
    plt.legend(['Training loss', 'Validation loss'], loc='upper left', fontsize=18)
    plt.show()
    print ("saving file:training loss figure")
    fig_acc.savefig(pic_dir + "/elm_enas_training_h1%s_h2%s_h3%s_h4%s_bs%s_sub%s_lr%s.png" %(int(h1), int(h2), int(h3), int(h4), int(bs), int(sub), str(lr)))
    return


def score_calculator(y_predicted, y_actual):
    # Score metric
    h_array = y_predicted - y_actual
    s_array = np.zeros(len(h_array))
    for j, h_j in enumerate(h_array):
        if h_j < 0:
            s_array[j] = math.exp(-(h_j / 13)) - 1

        else:
            s_array[j] = math.exp(h_j / 10) - 1
    score = np.sum(s_array)
    return score


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




def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='NAS CNN')
    parser.add_argument('-w', type=int, default=50, help='sequence length', required=True)
    parser.add_argument('-s', type=int, default=50, help='stride of filter')
    parser.add_argument('-bs', type=int, default=512, help='batch size')
    parser.add_argument('-ep', type=int, default=30, help='max epoch')
    parser.add_argument('-pt', type=int, default=30, help='patience')
    parser.add_argument('-vs', type=float, default=0.2, help='validation split')
    parser.add_argument('-lr', type=float, default=10**(-1*4), help='learning rate')
    parser.add_argument('-sub', type=int, default=1, help='subsampling stride')
    parser.add_argument('-t', type=int, required=True, help='trial')
    parser.add_argument('--pop', type=int, default=50, required=False, help='population size of EA')
    parser.add_argument('--gen', type=int, default=50, required=False, help='generations of evolution')
    parser.add_argument('--device', type=str, default="GPU", help='Use "basic" if GPU with cuda is not available')
    parser.add_argument('--obj', type=str, default="soo", help='Use "soo" for single objective and "moo" for multiobjective')

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

    ## Parameters for the GA
    pop_size = args.pop
    n_generations = args.gen
    cx_prob = 0.5  # 0.25
    mut_prob = 0.5  # 0.7
    cx_op = "one_point"
    mut_op = "uniform"

    if obj == "soo":
        sel_op = "best"
        other_args = {
            'mut_gene_probability': 0.3  # 0.1
        }

        mutate_log_path = 'EA_log/mute_log_ori_%s_%s_%s_%s_%s.csv' % (pop_size, n_generations, obj, trial, ep)
        mutate_log_col = ['idx', 'params_1', 'params_2', 'params_3', 'params_4', 'fitness_1',
                          'gen']
        mutate_log_df = pd.DataFrame(columns=mutate_log_col, index=None)
        mutate_log_df.to_csv(mutate_log_path, index=False)

        def log_function(population, gen, hv=None, mutate_log_path=mutate_log_path):
            for i in range(len(population)):
                indiv = population[i]
                if indiv == []:
                    "non_mutated empty"
                    pass
                else:
                    # print ("i: ", i)
                    indiv.append(indiv.fitness.values[0])
                    indiv.append(gen)

            temp_df = pd.DataFrame(np.array(population), index=None)
            temp_df.to_csv(mutate_log_path, mode='a', header=None)
            print("population saved")
            return


    # elif obj == "moo":
    else:
        sel_op = "nsga2"
        other_args = {
            'mut_gene_probability': 0.4  # 0.1
        }

        mutate_log_path = 'EA_log/mute_log_ori_%s_%s_%s_%s_%s.csv' % (pop_size, n_generations, obj, trial, ep)
        mutate_log_col = ['idx', 'params_1', 'params_2', 'params_3', 'params_4', 
                          'fitness_1', 'fitness_2', 'hypervolume', 'gen']
        mutate_log_df = pd.DataFrame(columns=mutate_log_col, index=None)
        mutate_log_df.to_csv(mutate_log_path, index=False)

        def log_function(population, gen, hv=None, mutate_log_path=mutate_log_path):
            for i in range(len(population)):
                indiv = population[i]
                if indiv == []:
                    "non_mutated empty"
                    pass
                else:
                    # print ("i: ", i)
                    indiv.append(indiv.fitness.values[0])
                    indiv.append(indiv.fitness.values[1])
                    # append val_rmse
                    indiv.append(hv)
                    indiv.append(gen)

            temp_df = pd.DataFrame(np.array(population), index=None)
            temp_df.to_csv(mutate_log_path, mode='a', header=None)
            print("population saved")
            return





    prft_path = os.path.join(directory_path, 'prft_out_ori_%s_%s_%s_%s.csv' % (pop_size, n_generations, trial,ep))





    start = time.time()

    # Assign & run EA
    task = SimpleNeuroEvolutionTask(
        train_sample_array = train_sample_array,
        train_label_array = train_label_array,
        val_sample_array = val_sample_array,
        val_label_array = val_label_array,
        batch=bs,
        epoch = ep,
        patience = pt,
        val_split = vs,
        model_path = model_temp_path,
        device = device,
        obj = obj
    )

    ga = GeneticAlgorithm(
        task=task,
        population_size=pop_size,
        n_generations=n_generations,
        cx_probability=cx_prob,
        mut_probability=mut_prob,
        crossover_operator=cx_op,
        mutation_operator=mut_op,
        selection_operator=sel_op,
        jobs=jobs,
        log_function=log_function,
        prft_path=prft_path,
        **other_args
    )

    pop, log, hof, prtf = ga.run()

    print("Best individual:")
    print ("hof", hof)
    print("hof[0]", hof[0])
    print(prtf)


    print("Best individual is saved")
    end = time.time()
    print("EA time: ", end - start)
    print ("####################  EA COMPLETE / HOF TEST   ##############################")






if __name__ == '__main__':
    main()