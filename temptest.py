
import argparse
import time
import json
import logging
import glob
import math
import matplotlib.pyplot as plt
import os, sys
from matplotlib.pyplot import cm
import matplotlib.figure
import numpy as np
import pandas as pd
import cvxpy as cp

from scipy.optimize import curve_fit
from utils.curve_functions import *

current_dir = os.path.dirname(os.path.abspath(__file__))

val_rmse_filepath = os.path.join(current_dir, 'val_rmse_hist.csv')
train_time_filepath = os.path.join(current_dir, 'train_time_hist.csv')


pic_dir = os.path.join(current_dir, 'Curves')
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)

all_models = {}
# all_models["vap"] = vap
all_models["pow3"] = pow3
# all_models["loglog_linear"] = loglog_linear
all_models["loglog_linear"] = loglog_linear
# all_models["dr_hill"] = dr_hill
# all_models["log_power"] = log_power
# all_models["pow4"] = pow4
all_models["mmf"] = mmf
# all_models["exp"] = exponential_fit
# all_models["exp3"] = exp3
# all_models["exp4"] = exp4
all_models["janoschek"] = janoschek
all_models["weibull"] = weibull
all_models["ilog2"] = ilog2
# all_models["dr_hill_zero_background"] = dr_hill_zero_background
# all_models["logx_linear"] = logx_linear
# all_models["exp3"] = exp3
# all_models["pow2"] = pow2
# all_models["sat_growth"] = sat_growth

#######################################################

def main():

    lst = [8.25815391540527, 6.46415662765503 , 6.8708176612854, 6.56571435928345 , 8.20171356201172, 6.60947322845459, 
    6.4962363243103, 6.38670587539673, 6.44164562225342]
    array = np.asarray(lst)
    print (array)

    rank = np.argsort(-1*np.asarray(array))[::-1]
    print (rank)



if __name__ == '__main__':
    main()
