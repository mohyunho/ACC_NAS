import os, sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.figure
import numpy as np
import pandas as pd


all_models = {}

def pow3(x, c, a, alpha):
    return c - a * x**(-alpha)
all_models["pow3"] = pow3


c = 0.84
a = 0.52
alpha = 0.01




n = len(df_curve)
color = iter(cm.rainbow(np.linspace(0, 1, n)))

# Draw plot
fig = matplotlib.figure.Figure(figsize=(6, 4))
# agg.FigureCanvasAgg(fig)
ax = fig.add_subplot(1, 1, 1)

y_min = 0
y_max = 600

x = np.linspace(0,29, num=30)
print (x)

x_range = np.arange(df_curve.shape[1]) 
x0 = 10
# ax.hlines(np.arange(start_value - ref_avg*interval, start_value, interval), x_min, x_max, colors=(0.1, 0.1, 0.1, 0.1), zorder=2)
for index, row in df_curve.iterrows():
  c = next(color)
  
  ax.plot(x, row, linewidth=0.5, color=c, label='Ind. %s' %(index+1))
  # ax.plot(x[x0:], row[x0:], linestyle='dashed', color=c, linewidth=0.5, alpha= 0.5)

# ax.vlines(x0-1, y_min, y_max, linestyle='dashed', zorder=2)
# ax.vlines(12, y_min, y_max, linestyle='dashed')
ax.set_xticks(x_range)
ax.set_xticklabels(x_range+1, rotation=60, fontsize=8)
# ax.set_yticks(np.arange(y_min, y_max, 2 * y_sp))
# ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
# ax.set_title("Solutions and pareto front", fontsize=15)
ax.set_xlabel('Epochs', fontsize=12)
ax.set_ylabel('Validation loss', fontsize=12)
ax.legend(fontsize=7)

# Save figure
# ax.set_rasterized(True)
fig.savefig(os.path.join(plot_dic, 'val_loss_full.png'), dpi=1500, bbox_inches='tight')
# fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.eps' % (pop, gen, trial)), dpi=1500, bbox_inches='tight')
# fig.savefig(os.path.join(pic_dir, 'val_score_%s_%s_%s.pdf' % (pop, gen, trial)), bbox_inches='tight')