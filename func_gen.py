import os, sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.figure
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit


current_dir = os.path.dirname(os.path.abspath(__file__))
# data_filepath = os.path.join(current_dir, 'dati_bluetensor.xlsx')

pic_dir = os.path.join(current_dir, 'Curves')
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)


all_models = {}

def pow3(x, c, a, alpha):
    return  c + a * x**(-alpha)
all_models["pow3"] = pow3


def exponential_fit(x, a, b, c):
    return a*np.exp(-b*x) + c


def inv_exponential_fit(x, a, b, c):
    return a*np.exp(-b*x) + c


c = 0.84
b = 2.0
a = 0.52
alpha = 0.01


# x = np.linspace(0,29, num=30)
# print ("x", x)

# y = pow3(x, c, a, alpha)
# print ("y", y)

fig = matplotlib.figure.Figure(figsize=(6, 4))
# agg.FigureCanvasAgg(fig)
ax = fig.add_subplot(1, 1, 1)

x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([200, 40, 25, 22, 21, 24])
fitting_parameters, covariance = curve_fit(exponential_fit, x, y)
a, b, c = fitting_parameters

# next_x = 6
next_x = np.arange(6,30)
next_y = exponential_fit(next_x, a, b, c)


ax.plot(y)
ax.plot(np.append(y, next_y), 'ro')

fig.savefig(os.path.join(pic_dir, 'curve_temp.png'), bbox_inches='tight')

# plt.plot(y)
# plt.plot(np.append(y, next_y), 'ro')
# plt.show()








###################################################################à
# # n = len(df_curve)
# # color = iter(cm.rainbow(np.linspace(0, 1, n)))

# # Draw plot
# fig = matplotlib.figure.Figure(figsize=(6, 4))
# # agg.FigureCanvasAgg(fig)
# ax = fig.add_subplot(1, 1, 1)

# # y_min = 0
# # y_max = 600







# x_range = np.arange(4) 
# ax.plot(x, y, linewidth=0.5, label='pow3')

# # ax.hlines(np.arange(start_value - ref_avg*interval, start_value, interval), x_min, x_max, colors=(0.1, 0.1, 0.1, 0.1), zorder=2)
# # for index, row in df_curve.iterrows():
# #   c = next(color)
  
  
#   # ax.plot(x[x0:], row[x0:], linestyle='dashed', color=c, linewidth=0.5, alpha= 0.5)

# # ax.vlines(x0-1, y_min, y_max, linestyle='dashed', zorder=2)
# # ax.vlines(12, y_min, y_max, linestyle='dashed')
# ax.set_xticks(x_range)
# # ax.set_xticklabels(x_range+1, rotation=60, fontsize=8)
# # ax.set_yticks(np.arange(y_min, y_max, 2 * y_sp))
# # ax.set_xlim(x_min, x_max)
# # ax.set_ylim(y_min, y_max)
# # ax.set_title("Solutions and pareto front", fontsize=15)
# ax.set_xlabel('Epochs', fontsize=12)
# ax.set_ylabel('Validation loss', fontsize=12)
# ax.legend(fontsize=7)

# # Save figure
# # ax.set_rasterized(True)
# fig.savefig(os.path.join(pic_dir, 'curve_temp.png'), bbox_inches='tight')
