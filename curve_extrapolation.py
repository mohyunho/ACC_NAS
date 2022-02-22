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

# def vap(x, a, b, c):
#     """ Vapor pressure model """
#     return np.exp(a+b/x+c*np.log(x))
# all_models["vap"] = vap

def pow3(x, c, a, alpha):
    return  c - a * x**(-alpha)
all_models["pow3"] = pow3

# def loglog_linear(x, a, b):
#     x = np.log(x)
#     return np.log(a*x + b)
# all_models["loglog_linear"] = loglog_linear

def loglog_linear(x, a, b, c):
    # x = x+1
    x = np.log(x)
    return -1*np.log(np.abs(a*x - b))+c
all_models["loglog_linear"] = loglog_linear

# def dr_hill(x, alpha, theta, eta, kappa):
#     return alpha + (theta*(x**eta)) / (kappa**eta + x**eta)
# all_models["dr_hill"] = dr_hill

# def log_power(x, a, b, c):
#     #logistic power
#     return -1*a/(10.+ np.abs(x/np.exp(b))**c)
# all_models["log_power"] = log_power

# def pow4(x, c, a, b, alpha):
#     return c - (a*x+b)**-alpha
# all_models["pow4"] = pow4

def mmf(x, alpha, beta, kappa, delta):
    return alpha - (alpha - beta) / (1. + np.abs(kappa * x)**delta)
all_models["mmf"] = mmf

# def exponential_fit(x, a, b, c):
#     return a*np.exp(-b*x) + c
# all_models["exp"] = exponential_fit

# def exp3(x, c, a, b):
#     return -c + np.exp(-a*x+b)
# all_models["exp3"] = exp3

# def exp4(x, c, a, b, alpha):
#     return -c + np.exp(-a*(x**alpha)+b)
# all_models["exp4"] = exp4

def janoschek(x, a, beta, k, delta):
    return a - (a - beta) * np.exp(-k*x**delta)
all_models["janoschek"] = janoschek

def weibull(x, alpha, beta, kappa, delta):
    x = 1 + x
    return alpha - (alpha - beta) * np.exp(-(kappa * x)**delta)
all_models["weibull"] = weibull

def ilog2(x, c, a, b):
    x = 1 + x
    assert(np.all(x>1))
    return -c + a / np.log(b*x)
all_models["ilog2"] = ilog2

# def dr_hill_zero_background(x, theta, eta, kappa):
#     return (theta* x**eta) / (kappa**eta + x**eta)
# all_models["dr_hill_zero_background"] = dr_hill_zero_background

# def logx_linear(x, a, b):
#     x = np.log(x)
#     return a*x + b
# all_models["logx_linear"] = logx_linear

# def exp3(x, c, a, b):
#     return c - np.exp(-a*x+b)
# all_models["exp3"] = exp3

# def pow2(x, a, alpha):
#     return a * x**(-alpha)
# all_models["pow2"] = pow2

# def sat_growth(x, a, b):
#     return a * x / (b + x)
# all_models["sat_growth"] = sat_growth

#######################################################

# 50-50-no samp, val rmse
# history.history['val_root_mean_squared_error'] [22.28994369506836, 9.049633026123047, 7.706231117248535, 7.238524436950684, 7.964917182922363, 7.521146774291992, 8.026594161987305, 7.991659641265869, 7.034694194793701, 7.5684380531311035, 7.264317512512207, 6.615848541259766, 6.616621494293213, 6.88593053817749, 9.603377342224121, 6.3677520751953125, 6.352144241333008, 6.367636203765869, 6.3384108543396, 6.352309226989746, 6.3333539962768555, 6.351520538330078, 6.331465721130371, 6.363064765930176, 6.40748929977417, 6.328959941864014, 6.3523478507995605, 6.338868618011475, 6.319766998291016, 6.3180437088012695]

# 
# history.history['val_root_mean_squared_error'] [23.69171142578125, 22.754688262939453, 22.42828941345215, 21.883119583129883, 20.384809494018555, 10.34829330444336, 8.587149620056152, 7.931539535522461, 7.662485122680664, 7.543306350708008, 7.772759914398193, 7.262722015380859, 7.134206295013428, 7.4294562339782715, 7.071417808532715, 6.967785358428955, 6.975186347961426, 6.959003925323486, 6.95330286026001, 6.955667972564697, 6.963987827301025, 6.953071594238281, 6.937443733215332, 6.952975273132324, 6.925543785095215, 6.926819801330566, 6.92205810546875, 6.913516521453857, 6.901162147521973, 6.8993635177612305]

# history.history['val_root_mean_squared_error'] [22.801315307617188, 21.162397384643555, 9.488973617553711, 8.141096115112305, 7.90029764175415, 7.627315998077393, 8.38720417022705, 7.134346008300781, 6.866602420806885, 7.2095842361450195, 7.2944722175598145, 6.9655280113220215, 6.731527328491211, 7.41037654876709, 7.285154819488525, 6.547308921813965, 6.554522514343262, 6.546627521514893, 6.530821800231934, 6.5478997230529785, 6.536942005157471, 6.549753189086914, 6.523753643035889, 6.583749294281006, 6.538174152374268, 6.520839214324951, 6.532629489898682, 6.531630039215088, 6.516921520233154, 6.516541481018066]


# 50-50-no samp, val loss
# history.history['val_loss'] [496.841552734375, 81.8958511352539, 59.3859977722168, 52.39623260498047, 63.43990707397461, 56.567649841308594, 64.42621612548828, 63.86662673950195, 49.48692321777344, 57.281253814697266, 52.77030563354492, 43.76945114135742, 43.77968215942383, 47.416038513183594, 92.22486114501953, 40.54826736450195, 40.34973907470703, 40.546791076660156, 40.17544937133789, 40.35183334350586, 40.11137008666992, 40.34181213378906, 40.087459564208984, 40.488590240478516, 41.0559196472168, 40.05573272705078, 40.35232162475586, 40.18125534057617, 39.939456939697266, 39.91767883300781]

# history.history['val_loss'] [519.8999633789062, 447.8470458984375, 90.0406265258789, 66.27745056152344, 62.414703369140625, 58.17594909667969, 70.34519958496094, 50.89889144897461, 47.15022659301758, 51.97810745239258, 53.20932388305664, 48.518577575683594, 45.31345748901367, 54.91367721557617, 53.073482513427734, 42.86725616455078, 42.96176528930664, 42.858333587646484, 42.65163040161133, 42.87499237060547, 42.73161315917969, 42.899269104003906, 42.55936050415039, 43.34575653076172, 42.747718811035156, 42.52134323120117, 42.67524719238281, 42.662193298339844, 42.47026443481445, 42.46531295776367]

#######################################################
numb_obeservation = 10




# First five points
# y_all = [22.28994369506836, 9.049633026123047, 7.706231117248535, 7.238524436950684, 7.964917182922363, 7.521146774291992, 8.026594161987305, 7.991659641265869, 7.034694194793701, 7.5684380531311035, 7.264317512512207, 6.615848541259766, 6.616621494293213, 6.88593053817749, 9.603377342224121, 6.3677520751953125, 6.352144241333008, 6.367636203765869, 6.3384108543396, 6.352309226989746, 6.3333539962768555, 6.351520538330078, 6.331465721130371, 6.363064765930176, 6.40748929977417, 6.328959941864014, 6.3523478507995605, 6.338868618011475, 6.319766998291016, 6.3180437088012695]
# y_all = [23.69171142578125, 22.754688262939453, 22.42828941345215, 21.883119583129883, 20.384809494018555, 10.34829330444336, 8.587149620056152, 7.931539535522461, 7.662485122680664, 7.543306350708008, 7.772759914398193, 7.262722015380859, 7.134206295013428, 7.4294562339782715, 7.071417808532715, 6.967785358428955, 6.975186347961426, 6.959003925323486, 6.95330286026001, 6.955667972564697, 6.963987827301025, 6.953071594238281, 6.937443733215332, 6.952975273132324, 6.925543785095215, 6.926819801330566, 6.92205810546875, 6.913516521453857, 6.901162147521973, 6.8993635177612305]
y_all = [22.801315307617188, 21.162397384643555, 9.488973617553711, 8.141096115112305, 7.90029764175415, 7.627315998077393, 8.38720417022705, 7.134346008300781, 6.866602420806885, 7.2095842361450195, 7.2944722175598145, 6.9655280113220215, 6.731527328491211, 7.41037654876709, 7.285154819488525, 6.547308921813965, 6.554522514343262, 6.546627521514893, 6.530821800231934, 6.5478997230529785, 6.536942005157471, 6.549753189086914, 6.523753643035889, 6.583749294281006, 6.538174152374268, 6.520839214324951, 6.532629489898682, 6.531630039215088, 6.516921520233154, 6.516541481018066]

# x_max = np.arange(1e-4,30)
# print ("x_max", x_max)
x_max = np.arange(1,31)
print ("x_max", x_max)
x_observation = x_max[:numb_obeservation]
y_obesrvation = y_all[:numb_obeservation]
fig = matplotlib.figure.Figure(figsize=(8, 6))
# agg.FigureCanvasAgg(fig)
ax = fig.add_subplot(1, 1, 1)
# Plot actual curve with solid line
ax.plot(x_max, y_all, label="observation", color="black", linewidth=3, zorder=2)
color = iter(cm.rainbow(np.linspace(0, 1, len(all_models))))



for index, (key, value) in enumerate(all_models.items()):
    print ("value", value)
    next_x = np.arange(numb_obeservation+1,31)
    if key == "loglog_linear":
        fitting_parameters, covariance = curve_fit(value, x_observation, y_obesrvation, p0=[10,1,10], maxfev=50000)
    elif key == "pow3":
        fitting_parameters, covariance = curve_fit(value, x_observation, y_obesrvation, maxfev=50000)
    elif key == "ilog2":
        fitting_parameters, covariance = curve_fit(value, x_observation, y_obesrvation, maxfev=50000)
    else:
        fitting_parameters, covariance = curve_fit(value, x_observation, y_obesrvation, maxfev=50000)
    if len(fitting_parameters)==2:
        y_func = value(x_observation, fitting_parameters[0], fitting_parameters[1])
        next_y = value(next_x, fitting_parameters[0], fitting_parameters[1])
    elif len(fitting_parameters)==3:
        y_func = value(x_observation, fitting_parameters[0], fitting_parameters[1], fitting_parameters[2])
        next_y = value(next_x, fitting_parameters[0], fitting_parameters[1], fitting_parameters[2])
    elif len(fitting_parameters)==4:
        y_func = value(x_observation, fitting_parameters[0], fitting_parameters[1], fitting_parameters[2], fitting_parameters[3])
        next_y = value(next_x, fitting_parameters[0], fitting_parameters[1], fitting_parameters[2], fitting_parameters[3])
    # Plot extrapolation with red circles
    # ax.plot(np.append(y, next_y), 'ro')
    c = next(color)
    # ax.plot(next_x, next_y, color=c, marker='o', label=key)
    ax.plot(x_max, np.append(y_func, next_y), color=c, marker='o', linestyle='dashed', label=key, zorder=1)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_xlabel('Epoch', fontsize=15)
    ax.set_ylabel('Validation RMSE', fontsize=15)
    x_epoch = np.arange(1,31)
    ax.set_xticks(x_epoch)
    ax.set_xticklabels(x_epoch, rotation=60)
    ymax_plot = 25
    ax.set_ylim(0, ymax_plot)
    ax.vlines(numb_obeservation,  0, ymax_plot, colors=(0.1, 0.1, 0.1, 0.1), linestyle='-.',linewidth=1, zorder=3)
    fig.savefig(os.path.join(pic_dir, 'curve_archt2_epoch_%s.png' %numb_obeservation), bbox_inches='tight')


