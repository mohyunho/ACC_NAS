#!/bin/python3
"""
This file contains the implementation of a Task, used to load the data and compute the fitness of an individual

"""
import pandas as pd
from abc import abstractmethod

# from input_creator import input_gen
from utils.cnn_network import network_fit



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


class SimpleNeuroEvolutionTask(Task):
    '''
    Class for EA Task
    '''
    def __init__(self, train_sample_array, train_label_array, val_sample_array, val_label_array, batch,
                 epoch, patience, val_split, model_path, device, obj):
        self.train_sample_array = train_sample_array
        self.train_label_array = train_label_array
        self.val_sample_array = val_sample_array
        self.val_label_array = val_label_array
        self.batch = batch
        self.epoch = epoch
        self.patience = patience
        self.val_split = val_split
        self.model_path = model_path
        self.device = device
        self.obj = obj

    def get_n_parameters(self):
        return 4

    def get_parameters_bounds(self):
        bounds = [
            (1, 10), #n_layers
            (5, 30), #n_filters
            (5, 30), #kernel_size
            (1, 50), #n_mlp
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
                                self.batch, self.epoch, self.patience, self.val_split,
                                self.model_path, self.device)


        cnn_net = cnn_class.trained_model()

        validation, num_tran_params, train_time = cnn_class.train_net(cnn_net, lr, self.train_sample_array, self.train_label_array, self.val_sample_array,
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


