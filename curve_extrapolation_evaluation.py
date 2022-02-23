
import argparse
import time
import json
import logging
import glob
import math
import os, sys
import matplotlib.figure
import numpy as np
import pandas as pd
import cvxpy as cp

import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt


import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.figure
import matplotlib.backends.backend_agg as agg
import matplotlib.backends.backend_svg as svg

from scipy import stats
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
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='sample creator')
    parser.add_argument('-obep', type=int, default=15, help='obsevation epochs')


    args = parser.parse_args()
    ob_ep = args.obep

    val_rmse_df = pd.read_csv(val_rmse_filepath)
    train_time_df = pd.read_csv(train_time_filepath)

    val_rmse_observation = []
    val_rmse_mid = []
    val_rmse_prediction = []

    extp_time_lst = []

    for arch_idx, val_rmse_hist in val_rmse_df.iterrows():

        start_itr = time.time()

        # print ("arch_idx", arch_idx)
        y_all = val_rmse_hist
        # print ("y_all", y_all)
        x_max = np.arange(1,31)
        # print ("x_max", x_max)
        x_observation = x_max[:ob_ep]
        y_obesrvation = y_all[:ob_ep]
        fig = matplotlib.figure.Figure(figsize=(8, 6))
        # agg.FigureCanvasAgg(fig)
        ax = fig.add_subplot(1, 1, 1)
        # Plot actual curve with solid line
        ax.plot(x_max, y_all, label="observation", color="black", linewidth=3, zorder=2)
        color = iter(cm.rainbow(np.linspace(0, 1, len(all_models)+1)))

        y_func_lst = []
        curve_y_lst = []

        for index, (key, value) in enumerate(all_models.items()):
            # print ("value", value)
            next_x = np.arange(ob_ep+1,31)
            if key == "loglog_linear":
                fitting_parameters, covariance = curve_fit(value, x_observation, y_obesrvation, p0=[10,1,10], maxfev=5000000)
            elif key == "pow3":
                fitting_parameters, covariance = curve_fit(value, x_observation, y_obesrvation, maxfev=5000000)
            elif key == "ilog2":
                fitting_parameters, covariance = curve_fit(value, x_observation, y_obesrvation, maxfev=5000000)
            else:
                fitting_parameters, covariance = curve_fit(value, x_observation, y_obesrvation, maxfev=5000000)
            
            if len(fitting_parameters)==2:
                y_func = value(x_observation, fitting_parameters[0], fitting_parameters[1])
                next_y = value(next_x, fitting_parameters[0], fitting_parameters[1])
            elif len(fitting_parameters)==3:
                y_func = value(x_observation, fitting_parameters[0], fitting_parameters[1], fitting_parameters[2])
                next_y = value(next_x, fitting_parameters[0], fitting_parameters[1], fitting_parameters[2])
            elif len(fitting_parameters)==4:
                y_func = value(x_observation, fitting_parameters[0], fitting_parameters[1], fitting_parameters[2], fitting_parameters[3])
                next_y = value(next_x, fitting_parameters[0], fitting_parameters[1], fitting_parameters[2], fitting_parameters[3])
            
            y_func_lst.append(y_func)
            curve_y_lst.append(np.append(y_func, next_y))
            # next_y_lst.append(next_y)

      
            # Plot extrapolation with red circles
            # ax.plot(np.append(y, next_y), 'ro')
            c = next(color)
            # ax.plot(next_x, next_y, color=c, marker='o', label=key)
            ax.plot(x_max, np.append(y_func, next_y), color=c, marker='o', linestyle='dashed', label=key, zorder=1)

        # ax.legend(loc='upper right', fontsize=12)
        # fig.savefig(os.path.join(pic_dir, 'curve_archt%s_epoch_%s.png' %(arch_idx, ob_ep)), bbox_inches='tight')

        # list of 1d arrays to 2d array
        # print ("y_func_lst", y_func_lst)
        input_arrays = np.transpose(np.stack(y_func_lst, axis=0))
        # print ("input_arrays.shape", input_arrays.shape)
        # Find coefficient of linear combination of vectors with least square (olve a least-squares problem with CVXPY) 
        # https://www.cvxpy.org/examples/basic/least_squares.html
        # m: length of vector, n: numb of curves
        # Shape of A:(m, n), length of b: (m)

        # np.random.seed(0)
        # import cvxpy as cp

        n = len(all_models)
        # print ("n", n)
        coeff = cp.Variable(n)
        # print ("coeff",coeff)
        cost = cp.sum_squares(input_arrays @ coeff - y_obesrvation)
        # print ("cost", cost)

        # constraints = [input_arrays @ coeff >= y_obesrvation, coeff >= 0]
        constraints = [coeff >= 0]

        prob = cp.Problem(cp.Minimize(cost), constraints)
        # prob.solve(verbose=True)
        prob.solve()

        # print("The optimal coeff is")
        # print(coeff.value)

        # Combine (linear combination) of curves
        curves_arrays = np.transpose(np.stack(curve_y_lst, axis=0))
        # print("curves_arrays.shape", curves_arrays.shape)
        combined_y = curves_arrays  @ coeff.value 
        # print ("combined_y", combined_y)

        end_itr = time.time()
        extp_time = end_itr - start_itr
        extp_time_lst.append(extp_time)


        c = next(color)
        ax.plot(x_max, combined_y, color=c, marker='D', linewidth=2, label="combined", zorder=2)

        ax.legend(loc='upper right', fontsize=12)
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Validation RMSE', fontsize=15)
        x_epoch = np.arange(1,31)
        ax.set_xticks(x_epoch)
        ax.set_xticklabels(x_epoch, rotation=60)
        ymax_plot = 25
        ax.set_ylim(0, ymax_plot)
        ax.vlines(ob_ep,  0, ymax_plot, colors=(0.7, 0.7, 0.7), linestyle='-.',linewidth=1, zorder=3)

        fig.savefig(os.path.join(pic_dir, 'curve_archt%s_epoch_%s.png' %(arch_idx, ob_ep)), bbox_inches='tight')

        val_rmse_observation.append(y_all[-1])
        val_rmse_mid.append(y_all[ob_ep-1])
        val_rmse_prediction.append(combined_y[-1])


    # save to csv file
    rank_df = pd.DataFrame()
    rank_df["val_rmse_observation"] = np.asarray(val_rmse_observation)
    rank_df["val_rmse_prediction"] = np.asarray(val_rmse_prediction)
    rank_df["val_rmse_mid"] = np.asarray(val_rmse_mid)
    print ("len(rank_df)", len(rank_df))
    rank_df.drop(rank_df.loc[rank_df['val_rmse_prediction']>=11.0].index, inplace=True)
    rank_df.drop(rank_df.loc[rank_df['val_rmse_prediction']<=1.0].index, inplace=True)
    print ("len(rank_df)", len(rank_df))

    temp = np.argsort(-1*np.asarray(rank_df["val_rmse_observation"]))[::-1]
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(rank_df["val_rmse_observation"]))
    rank_df["rank_observation"] = ranks

    
    temp = np.argsort(-1*np.asarray(rank_df["val_rmse_prediction"] ))[::-1]
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(rank_df["val_rmse_prediction"] ))
    rank_df["rank_prediction"] = ranks

    
    temp = np.argsort(-1*np.asarray(rank_df["val_rmse_mid"]))[::-1]
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(rank_df["val_rmse_mid"]))
    rank_df["rank_mid"] = ranks






    

    # Top 5 RMSE
    top_k = 50
    rank_df = rank_df.sort_values(by='rank_observation', ascending=True)
    topk_observation = rank_df["val_rmse_observation"][:top_k]
    topk_prediction = rank_df["val_rmse_prediction"][:top_k]
    rms = sqrt(mean_squared_error(topk_observation, topk_prediction))
    rms = round(rms, 4)
    print ("Topk_obs_pred_rmse", rms)

    topk_mid = rank_df["val_rmse_mid"][:top_k]
    rms = sqrt(mean_squared_error(topk_observation, topk_mid))
    rms = round(rms, 4)
    print ("Topk_obs_mid_rmse", rms)

    rank_df.to_csv(os.path.join(current_dir, 'rank_val_rmse_0_50_%s.csv' %ob_ep))


    # Spearman rank-order correlation coefficient 
    rho_pred, pval_pred = stats.spearmanr(rank_df["rank_observation"], rank_df["rank_prediction"])
    rho_obsn, pval_obsn = stats.spearmanr(rank_df["rank_observation"], rank_df["rank_mid"])

    print ("rho_pred", rho_pred)
    print ("pval_pred", pval_pred)
    print ("rho_obsn", rho_obsn)
    print ("pval_obsn", pval_obsn)


    # Spearman rank-order correlation coefficient 
    rho_pred, pval_pred = stats.spearmanr(rank_df["rank_observation"][:top_k], rank_df["rank_prediction"][:top_k])
    rho_obsn, pval_obsn = stats.spearmanr(rank_df["rank_observation"][:top_k], rank_df["rank_mid"][:top_k])

    print ("top k rho_pred", rho_pred)
    print ("top k pval_pred", pval_pred)
    print ("top k rho_obsn", rho_obsn)
    print ("top k pval_obsn", pval_obsn)

    # elapsed time for extrapolation per individual
    extp_time_avg = sum(extp_time_lst) / len(extp_time_lst)
    extp_time_avg = round(extp_time_avg, 2)
    print ("extp_time_avg", extp_time_avg)

    # box plot
    fig = matplotlib.figure.Figure(figsize=(8, 6))
    # agg.FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    box_plot_data=[val_rmse_observation,val_rmse_prediction, val_rmse_mid]
    ax.boxplot(box_plot_data)
    ax.set_ylabel('Validation RMSE', fontsize=15)
    ax.set_yticks(np.arange(6,11.5,0.5))
    ax.set_ylim(6, 10)
    ax.set_xticklabels(["observation", "prediction", "at epoch %s" %ob_ep], fontsize=15)
    fig.savefig(os.path.join(pic_dir, 'boxplot_0_50_epoch_%s.png' %ob_ep), bbox_inches='tight')


    ####################################
    # Draw scatter plot
    fig = matplotlib.figure.Figure(figsize=(3, 3))
    agg.FigureCanvasAgg(fig)
    # cmap = get_cmap(10)
    ax = fig.add_subplot(1, 1, 1)
    # Draw scatter plot

    # x_min = int(min(rank_df["val_rmse_observation"])) - 0.5
    # x_max = int(max(rank_df["val_rmse_observation"])) + 0.5

    ax.scatter(rank_df["val_rmse_observation"], rank_df["val_rmse_prediction"], facecolor=(1.0, 1.0, 0.4),
               edgecolors=(0.0, 0.0, 0.0), zorder=1, s=20 )

    rmse_range = np.arange(6,11.5,0.5)
    # y_range = np.arange(int(min(rank_df["val_rmse_prediction"])), int(max(rank_df["val_rmse_prediction"])) + 0.5 ,0.5)
    y_range = np.arange(6, 10.5 ,0.5)
    ax.set_xticks(rmse_range)
    ax.set_xticklabels(rmse_range, rotation=60)
    ax.set_yticks(y_range)
    # ax.set_yticklabels(rmse_range)
    # ax.set_xlim(x_min, x_max)
    ax.set_ylim(6, 10)
    # ax.set_title("Solutions and pareto front", fontsize=15)
    ax.set_xlabel('Val. RMSE Observation', fontsize=12)
    ax.set_ylabel('Val. RMSE Prediction', fontsize=12)
    # ax.legend(fontsize=9)

    # Save figure
    # ax.set_rasterized(True)
    fig.savefig(os.path.join(pic_dir, 'val_extpl_obs_pred_epoch_%s.png' %ob_ep), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.eps' % (pop, gen, trial)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.pdf' % (pop, gen, trial)), bbox_inches='tight')




##########################
        ####################################
    # Draw scatter plot
    fig = matplotlib.figure.Figure(figsize=(3, 3))
    agg.FigureCanvasAgg(fig)
    # cmap = get_cmap(10)
    ax = fig.add_subplot(1, 1, 1)
    # Draw scatter plot

    # x_min = int(min(rank_df["val_rmse_observation"])) - 0.5
    # x_max = int(max(rank_df["val_rmse_observation"])) + 0.5

    ax.scatter(rank_df["val_rmse_observation"], rank_df["val_rmse_mid"], facecolor=(1.0, 1.0, 0.4),
               edgecolors=(0.0, 0.0, 0.0), zorder=1, s=20 )

    rmse_range = np.arange(6,11.5,0.5)
    # y_range = np.arange(int(min(rank_df["val_rmse_mid"])), int(max(rank_df["val_rmse_mid"])) + 0.5 ,0.5)
    y_range = np.arange(6, 10.5 ,0.5)
    

    ax.set_xticks(rmse_range)
    ax.set_xticklabels(rmse_range, rotation=60)
    ax.set_yticks(y_range)
    ax.set_ylim(6, 10)
    # ax.set_yticklabels(rmse_range)
    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)
    # ax.set_title("Solutions and pareto front", fontsize=15)
    ax.set_xlabel('Val. RMSE Observation', fontsize=12)
    ax.set_ylabel('Val. RMSE Epoch %s' %ob_ep, fontsize=12)
    # ax.legend(fontsize=9)

    # Save figure
    # ax.set_rasterized(True)
    fig.savefig(os.path.join(pic_dir, 'val_extpl_obs_mid_epoch_%s.png' %ob_ep), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.eps' % (pop, gen, trial)), dpi=1500, bbox_inches='tight')
    # fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.pdf' % (pop, gen, trial)), bbox_inches='tight')


    # bar graphs for comparison



if __name__ == '__main__':
    main()
