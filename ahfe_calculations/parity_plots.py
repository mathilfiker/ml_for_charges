import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import json

mobleys = ['36119','186894','242480','296847','337666','468867','1017962','1034539','1662128','1770205','2023925','2197088','2341732','2518989','3047364','3169935','3201701','4013838','7455579','7754849','8492526','9897248']
# Add labels to cross reference plots
labels = [3,13,11,10,18,21,4,20,1,15,22,2,6,17,14,19,12,9,7,16,5,8]

# Experimental values
experimental = [-4.09,-1.10,-11.85,-4.69,-16.92,-2.55,-2.49,-3.04,1.38,-10.03,0.10,3.16,-2.56,-5.74,-7.65,-4.91,-9.41,-2.74,-2.22,-6.10,-3.03,-4.78] 
exp_err = [0.60,0.20,0.35,0.60,0.88,0.10,0.60,0.10,0.60,1.37,0.60,0.60,0.60,1.93,0.45,0.60,1.93,0.60,0.60,1.37,0.60,0.60]
x= np.array(experimental)
x_err = np.array(exp_err)

# Values from Boltzmann Percentile charges
bp = [-3.23,-0.59,-11.04,-2.72,-12.40,-2.83,-2.25,-2.48,1.47,-8.05,-1.04,0.73,-2.41,-10.38,-7.67,-3.90,-7.91,-1.51,-2.19,-5.80,-2.59,-2.74]
er_bp = [0.02,0.03,0.07,0.03,0.02,0.03,0.02,0.07,0.03,0.07,0.05,0.03,0.06,0.03,0.1,0.04,0.06,0.05,0.03,0.07,0.04,0.04]
y_bp = np.array(bp)
y_err_bp = np.array(er_bp)

# Values from ESP charges
y_esp = np.load('ahfe_esp.npy')
y_err_esp = np.load('er_ahfe_esp.npy')

# Values from RESP charges
y_resp = np.load('ahfe_resp.npy')
y_err_resp = np.load('er_ahfe_resp.npy')

# Values from AM1-BCC charges
y_am1 = np.load('ahfe_am1bcc.npy')
y_err_am1 = np.load('er_ahfe_am1bcc.npy')

# Values from XGB 1-shot charges
y_oneshot = np.load('ahfe_1shot.npy')
y_err_oneshot = np.load('er_ahfe_1shot.npy')

# Common min and max to use the same scale on plots
y_min = min(min(x),min(y_esp),min(y_am1),min(y_oneshot),min(y_resp),min(y_bp))
y_max = max(max(x),max(y_esp),max(y_am1),max(y_oneshot),max(y_resp),max(y_bp))

# Plots
order_methods = [y_am1, y_esp, y_oneshot, y_resp, y_bp]
order_errors = [y_err_am1, y_err_esp, y_err_oneshot, y_err_resp, y_err_bp]
titles = ['AM1-BCC Charges', 'ESP Charges', 'XGB Charges: 1-shot', 'RESP Charges', 'XGB Charges: BP']
to_save_as = ['am1bcc','esp','1_shot','resp','bp']
for m in range(len(order_methods)):
    y = order_methods[m]
    y_err = order_errors[m]
    #metrics
    abs_err = mae(x,y)
    err = rmse(x,y)
    tau = kendalltau(x, y).statistic
    pearson = pearsonr(x, y).statistic
    r2 = r2_score(x, y)
    spear = spearmanr(x,y).statistic
    # Initialize figure
    fig, ax = plt.subplots(figsize=(16, 16))
    # Scatter plot with error bars
    ax.errorbar(x, y, xerr=x_err,yerr=y_err,fmt='o',color='tab:blue', alpha=0.8,markersize=30)
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.text(xi + 0.1, yi + 0.1, labels[i], fontsize=25, color='black')  # small offset to avoid overlap
    x_range = np.linspace(y_min, y_max, 400)
    # Plot the diagonal line (y = x)
    ax.plot(x_range, x_range, 'k-')
    # Shaded confidence regions (example)
    ax.fill_between(x_range, x_range-2, x_range+2, color="gray", alpha=0.3)
    ax.fill_between(x_range, x_range-1, x_range+1, color="gray", alpha=0.5)

    stats_text = (f"RMSE = {err:.2f} kcal/mol\n"
                  f"τ  = {tau:.2f}\n"
                  f"r  = {pearson:.2f}\n"
                  f"R² = {r2:.2f}\n"
                  r"$\rho$" +f"  = {spear:.2f}")
    ax.legend(title=stats_text, loc="upper left",title_fontsize=45)
    ax.set_ylim(y_min-1,y_max+1)
    ax.set_xlim(y_min-1,y_max+1)
    # Labels and title
    ax.set_xlabel("Experimental ΔG (kcal/mol)",fontsize=45)
    ax.set_ylabel("Computed ΔG (kcal/mol)",fontsize=45)
    ax.tick_params(axis='x',labelsize=40)
    ax.tick_params(axis='y',labelsize=40)
    ax.set_title(f"{titles[m]}",fontsize=45)
    plt.savefig(f'plots/{to_save_as[m]}.png', bbox_inches='tight',pad_inches = 0)
    plt.show()


#Extra 8 molecules
mobleys_extra = ['1944394','3105103','4188615','8966374','9073553','9201263','9460824','9979854']
labels_extra = [29,30,23,24,25,26,27,28]

experimental_extra = [-9.34,-11.14,-9.29,-5.51,-1.61,-4.87,-4.37,-4.20]
er_exp_extra = [0.62,1.93,0.60,0.60,0.60,0.60,0.10,0.20]

semi_extra = [-8.31,-13.76,-8.28,-5.67,-0.007,-8.59,-10.62,-3.48]
er_semi_extra = [0.08,0.07,0.03,0.03,0.03,0.05,0.05,0.07]

per_extra = [-6.70,-10.49,-8.60,-4.25,0.32,-6.45,-4.59,-2.90]
er_per_extra = [0.02,0.04,0.04,0.04,0.01,0.02,0.04,0.04]

one_shot_extra = [-3.59,-9.79,-5.78,-4.48,0.11,-7.53,-4.43,-3.56]
er_one_shot_extra = [0.04,0.03,0.06,0.05,0.01,0.03,0.07,0.02]

# Merge with previous data
labels_final = labels + labels_extra
x = np.append(x, experimental_extra)
x_err = np.append(x_err, er_exp_extra)

y_semi = np.append(y_am1,semi_extra)
y_err_semi = np.append(y_err_am1, er_semi_extra)

y_per = np.append(y_bp, per_extra)
y_err_per = np.append(y_err_bp, er_per_extra)

y_oneshot = np.append(y_oneshot, one_shot_extra)
y_err_oneshot = np.append(y_err_oneshot, er_one_shot_extra)

y_min = min(min(x),min(y_semi),min(y_per),min(y_oneshot))
y_max = max(max(x),max(y_semi),max(y_per),max(y_oneshot))

order_methods = [y_semi, y_oneshot, y_per]
order_errors = [y_err_semi, y_err_oneshot, y_err_per]
titles = ['AM1-BCC Charges','XGB Charges: 1-shot', 'XGB Charges: BP']
to_save_as = ['am1bcc','1_shot','bp']

for m in range(len(order_methods)):
    y = order_methods[m]
    y_err = order_errors[m]
    #metrics
    abs_err = mae(x,y)
    err = rmse(x,y)
    tau = kendalltau(x, y).statistic
    pearson = pearsonr(x, y).statistic
    r2 = r2_score(x, y)
    spear = spearmanr(x,y).statistic
    # Initialize figure
    fig, ax = plt.subplots(figsize=(16, 16))
    
    # Scatter plot with error bars
    ax.errorbar(x, y, xerr=x_err,yerr=y_err,fmt='o',color='tab:blue', alpha=0.8,markersize=30)
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.text(xi + 0.2, yi + 0.2, labels_final[i], fontsize=30, color='black')  # small offset to avoid overlap
    x_range = np.linspace(y_min, y_max, 400)
    
    # Plot the diagonal line (y = x)
    ax.plot(x_range, x_range, 'k-')
    # Shaded confidence regions (example)
    ax.fill_between(x_range, x_range-2, x_range+2, color="gray", alpha=0.3)
    ax.fill_between(x_range, x_range-1, x_range+1, color="gray", alpha=0.5)
    
    stats_text = (#f"MAE = {abs_err:.2f} kcal/mol\n"
                    f"RMSE = {err:.2f} kcal/mol\n"
                    f"τ = {tau:.2f}\n"
                    f"r = {pearson:.2f}\n"
                    f"R² = {r2:.2f}\n"
                    r"$\rho$" +f"  = {spear:.2f}")
    ax.legend(title=stats_text, loc="upper left",title_fontsize=45)
    ax.set_ylim(y_min-1,y_max+1)
    ax.set_xlim(y_min-1,y_max+1)
    # Labels and title
    ax.set_xlabel("Experimental ΔG (kcal/mol)",fontsize=45)
    ax.set_ylabel("Computed ΔG (kcal/mol)",fontsize=45)
    ax.tick_params(axis='x',labelsize=40)
    ax.tick_params(axis='y',labelsize=40)
    ax.set_title(f"{titles[m]}",fontsize=45)
    # Show plot
    plt.savefig(f'plots/extra_{to_save_as[m]}.png', bbox_inches='tight',pad_inches = 0)
    plt.show()