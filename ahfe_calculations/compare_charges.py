import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_error as mae
import json
from matplotlib.transforms import Bbox

def av_rmse(x,y):
    errors = []
    for i in range(len(x)):
        errors.append(rmse(x[i],y[i]))
    return np.array(errors)

mobleys = ['36119','186894','242480','296847','337666','468867','1017962','1034539','1662128','1770205','2023925','2197088','2341732','2518989','3047364','3169935','3201701','4013838','7455579','7754849','8492526','9897248']
semi = []
dft_esp = []
hf_resp = []
one_shot = []
percentile = []
for idx in mobleys:
    semi.append(np.load(f'am1bcc_charges/mobley_{idx}.npy'))
    dft_esp.append(np.load(f'esp_charges/mobley_{idx}.npy'))
    hf_resp.append(np.load(f'resp_charges/mobley_{idx}.npy'))
    one_shot.append(np.load(f'predicted_charges/mobley_{idx}.npy'))
    percentile.append(np.load(f'bp_charges/mobley_{idx}.npy'))

arrays = [semi, dft_esp, one_shot, hf_resp, percentile]
names = ['AM1-BCC','ESP','One-shot','RESP','BP']
n = len(arrays)
rmse_matrix = np.zeros((n,n))
for i in range(n):
    for j in range(i+1,n):
        error = av_rmse(arrays[i],arrays[j]).mean()
        rmse_matrix[i,j] = error
mask = np.tril(np.ones_like(rmse_matrix, dtype = bool),k=0)

# Create the heatmap figure
fig, ax = plt.subplots(figsize=(8, 6))

hm = sns.heatmap(
    rmse_matrix,
    mask=mask,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    xticklabels=names,
    yticklabels=names,
    cbar_kws={'label': r'RMSD [e]', 'shrink': 0.7}
)

# --- FONT SIZE ADJUSTMENTS ---
ax.tick_params(labelsize=16)  
hm.set_xlabel('', fontsize=18)
hm.set_ylabel('', fontsize=18)
for text in hm.texts:
    text.set_fontsize(18) 

# Adjust colorbar fonts
cbar = hm.collections[0].colorbar
cbar.ax.yaxis.set_tick_params(labelsize=14)
cbar.set_label(r'RMSD [e]', fontsize=16)

# Adjust layout so nothing is cut off
fig.tight_layout()

# --- Save the entire figure (heatmap + colorbar) ---
fig.savefig("heatmap.png", dpi=300, bbox_inches='tight')
