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


#######################################################

# 50-50-no samp, val rmse
# history.history['val_root_mean_squared_error'] [22.28994369506836, 9.049633026123047, 7.706231117248535, 7.238524436950684, 7.964917182922363, 7.521146774291992, 8.026594161987305, 7.991659641265869, 7.034694194793701, 7.5684380531311035, 7.264317512512207, 6.615848541259766, 6.616621494293213, 6.88593053817749, 9.603377342224121, 6.3677520751953125, 6.352144241333008, 6.367636203765869, 6.3384108543396, 6.352309226989746, 6.3333539962768555, 6.351520538330078, 6.331465721130371, 6.363064765930176, 6.40748929977417, 6.328959941864014, 6.3523478507995605, 6.338868618011475, 6.319766998291016, 6.3180437088012695]

# 50-50-no samp, val loss
# history.history['val_loss'] [496.841552734375, 81.8958511352539, 59.3859977722168, 52.39623260498047, 63.43990707397461, 56.567649841308594, 64.42621612548828, 63.86662673950195, 49.48692321777344, 57.281253814697266, 52.77030563354492, 43.76945114135742, 43.77968215942383, 47.416038513183594, 92.22486114501953, 40.54826736450195, 40.34973907470703, 40.546791076660156, 40.17544937133789, 40.35183334350586, 40.11137008666992, 40.34181213378906, 40.087459564208984, 40.488590240478516, 41.0559196472168, 40.05573272705078, 40.35232162475586, 40.18125534057617, 39.939456939697266, 39.91767883300781]



#######################################################
numb_obeservation = 5

all_models = {}

def pow3(x, c, a, alpha):
    return  c + a * x**(-alpha)
all_models["pow3"] = pow3


def exponential_fit(x, a, b, c):
    return a*np.exp(-b*x) + c


def inv_exponential_fit(x, a, b, c):
    return a*np.exp(-b*x) + c


# c = 0.84
# b = 2.0
# a = 0.52
# alpha = 0.01


# x = np.linspace(0,29, num=30)
# print ("x", x)

# y = pow3(x, c, a, alpha)
# print ("y", y)

fig = matplotlib.figure.Figure(figsize=(6, 4))
# agg.FigureCanvasAgg(fig)
ax = fig.add_subplot(1, 1, 1)

# First five points
y_all = [22.28994369506836, 9.049633026123047, 7.706231117248535, 7.238524436950684, 7.964917182922363, 7.521146774291992, 8.026594161987305, 7.991659641265869, 7.034694194793701, 7.5684380531311035, 7.264317512512207, 6.615848541259766, 6.616621494293213, 6.88593053817749, 9.603377342224121, 6.3677520751953125, 6.352144241333008, 6.367636203765869, 6.3384108543396, 6.352309226989746, 6.3333539962768555, 6.351520538330078, 6.331465721130371, 6.363064765930176, 6.40748929977417, 6.328959941864014, 6.3523478507995605, 6.338868618011475, 6.319766998291016, 6.3180437088012695]


x_max = np.arange(1e-4,30)
# x_observation = np.arange(0,numb_obeservation)
x_observation = x_max[:numb_obeservation]
y_obesrvation = y_all[:numb_obeservation]

# Curve fit
# With default method='lm', the algorithm uses the Levenberg-Marquardt algorithm through leastsq. 
# fitting_parameters, covariance = curve_fit(exponential_fit, x_observation, y_obesrvation, bounds=([-np.inf, 0.0001, -np.inf], [np.inf, 10, np.inf]))
fitting_parameters, covariance = curve_fit(pow3, x_observation, y_obesrvation)
# a, b, c = fitting_parameters
c, a, alpha = fitting_parameters


# linear combination for the following 5 points (np.linalg.solve)


# combined curve 


# Extrapolation of the combined curve.


# next_x = 6
next_x = np.arange(numb_obeservation,30)
# next_y = exponential_fit(next_x, a, b, c)
# next_y = exponential_fit(next_x, c, a, alpha )
next_y = pow3(next_x, c, a, alpha )

# Plot actual curve with solid line
ax.plot(x_max, y_all)

# Plot extrapolation with red circles
# ax.plot(np.append(y, next_y), 'ro')
ax.plot(next_x, next_y, 'ro')

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
