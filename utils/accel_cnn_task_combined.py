#!/bin/python3
"""
This file contains the implementation of a Task, used to load the data and compute the fitness of an individual

"""

import os
import math
import pandas as pd
import numpy as np
from abc import abstractmethod
import random
import tensorflow as tf
# from input_creator import input_gen
from utils.accel_cnn_network_combined import network_fit

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'


def release_list(lst):
   del lst[:]
   del lst

class Task:
    @abstractmethod
    def get_n_parameters(self):
        pass

    @abstractmethod
    def get_parameters_bounds(self):
        pass

    @abstractmethod
    def evaluate(self, genotype):
        pass

    # @abstractmethod
    # def evaluate_init(self, genotype):
    #     pass



class SimpleNeuroEvolutionTask(Task):
    '''
    Class for EA Task
    '''
    def __init__(self, train_sample_array, train_label_array, val_sample_array, val_label_array, batch,
                 epoch, ob_ep, st_ep, patience, val_split, threshold, model_path, device, obj):
        self.train_sample_array = train_sample_array
        self.train_label_array = train_label_array
        self.val_sample_array = val_sample_array
        self.val_label_array = val_label_array
        self.batch = batch
        self.epoch = epoch
        self.ob_ep = ob_ep
        self.st_ep = st_ep
        self.patience = patience
        self.val_split = val_split
        self.threshold = threshold
        self.model_path = model_path
        self.device = device
        self.obj = obj

    def get_n_parameters(self):
        return 4

    def get_parameters_bounds(self):
        bounds = [
            (3, 8), #n_layers
            (5, 25), #n_filters
            (5, 25), #kernel_size
            (5, 15), #n_mlp
        ]
        return bounds

    def evaluate(self, genotype):
        '''
        Create input & generate NNs & calculate fitness (to evaluate fitness of each individual)
        :param genotype:
        :return:
        '''
        print ("######################################################################################")
        n_layers = genotype[0]
        n_filters = genotype[1]
        kernel_size = genotype[2]
        n_mlp = 10 * genotype[3]
        lr = 10**(-1*4)
        # lr_lst = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        # lr= lr_lst[genotype[4]-1]

        print ("n_layers: ", n_layers)
        print("n_filters: ", n_filters)
        print("kernel_size: ", kernel_size)
        print("n_mlp: ", n_mlp)
        print("lr: ", lr)

        cnn_class = network_fit(self.train_sample_array, n_layers, n_filters, kernel_size, n_mlp,
                                self.batch, self.ob_ep, self.st_ep, self.epoch, self.patience, self.val_split, self.threshold,
                                self.model_path, self.device)


        cnn_net = cnn_class.trained_model()

        # Calculate architecture score
        
        geno_lst = [genotype[0], genotype[1], genotype[2], genotype[3]]
        validation, num_tran_params, train_time = cnn_class.train_net(cnn_net, lr, geno_lst, self.train_sample_array, self.train_label_array, self.val_sample_array,
                                    self.val_label_array)

        val_value = validation[0]

        if self.obj == "soo":
            fitness = (val_value,)
        elif self.obj == "moo":
            fitness = (val_value, num_tran_params)

        print("fitness: ", fitness)

        cnn_class = None
        cnn_net  = None
        del cnn_class, cnn_net

        return fitness


