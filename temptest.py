
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

from pygmo import *

def main():
    ref = [20.0, 200000.0]
    points = [[6.5, 19224],[7.2, 15000], [7.2, 10000],  [7.5, 30000], [7.1, 10000],[8.2, 15000],[6.2, 15000], [8.2, 5000], [11.2, 5000], [13.2, 3000], [7.2, 21000]]
    hv = hypervolume(points = points)
    result = hv.compute(ref_point = ref)
    print ("result", result)

if __name__ == '__main__':
    main()
