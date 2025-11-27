import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error as rmse
import sys
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

data = pd.read_csv("../data/subset_test.csv")
X= data.iloc[:,3:451]
Y = data.iloc[:, -1] 
atomic_numbers = data.iloc[:,2]
element_map = {
    1: 'H',
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F',
    15: 'P',
    16: 'S',
    17: 'Cl'
}
element_colors_map = {
    'H': 'white',
    'C': 'grey',
    'N': 'blue',
    'O': 'red',
    'F': 'green',
    'Cl': 'lime',  
    'P': 'orange',
    'S': 'yellow'
}
# Convert elements to array for indexing (not necessary for this toy dataset)
elements = np.array([element_map[z] for z in atomic_numbers])

# Maximum number of points per overrepresented element (C and H=)
max_points = 50000

# List to hold indices to keep
keep_indices = []

# Loop over unique elements
for elem in np.unique(elements):
    idx = np.where(elements == elem)[0]
    if elem in ['H', 'C']:
        # Randomly sample max_points indices if there are more
        if len(idx) > max_points:
            sampled_idx = np.random.choice(idx, max_points, replace=False)
        else:
            sampled_idx = idx
        keep_indices.extend(sampled_idx)
    else:
        # Keep all other elements
        keep_indices.extend(idx)

# Run PCA    
pca = PCA(n_components=2)
X_sub = X.iloc[keep_indices]
Y_sub = Y.iloc[keep_indices]
X_pca = pca.fit_transform(X_sub)

# Get explained variance ratio
explained_var = pca.explained_variance_ratio_
print(f"Explained variance by first 2 PCs: {explained_var.sum():.2%}")
print(f"  PC1: {explained_var[0]:.2%}, PC2: {explained_var[1]:.2%}")

# PCA plot colored by Element
elements_sub = elements[keep_indices]
unique_elements = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl']
colors = plt.cm.tab10.colors[:len(unique_elements)]
color_dict = dict(zip(unique_elements, colors))

plt.figure(figsize=(20,15))
for elem in unique_elements:
    mask_elem = elements_sub == elem
    plt.scatter(X_pca[mask_elem, 0],
                X_pca[mask_elem, 1],
                label=elem, color=element_colors_map[elem],
                alpha=0.7, edgecolor='k', s=70)

plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
plt.xlabel('PC1', fontsize = 46)
plt.ylabel('PC2', fontsize = 46)
plt.title('PCA Colored by Element',fontsize = 46)
plt.xticks(fontsize=46)
plt.yticks(fontsize=46)
plt.legend(loc='best',fontsize = 38,markerscale=3)
plt.tight_layout()
plt.savefig('pca_by_element.png',dpi=300)
plt.clf()

# Get predictions
model = xgb.XGBRegressor()
model.load_model('../model/xgb_chgs.json')
pred_chgs = model.predict(X_sub.values)
true_chgs = Y_sub.values
print("RMSE on the subset: ",rmse(pred_chgs,true_chgs))
errors = pred_chgs - true_chgs

#PCA plot colored by prediction error
errors_abs = abs(errors)
plt.figure(figsize=(20,15))

# Scatter plot colored by error
sc = plt.scatter(X_pca[:,0], X_pca[:,1],
                 c=errors_abs,          # color = prediction error
                 cmap='viridis',       # choose a colormap
                 norm=mcolors.Normalize(vmin=np.percentile(errors_abs, 5), vmax=np.percentile(errors_abs, 95)),
                 s=70,
                 edgecolor='k',
                 alpha=0.7)

plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
plt.xlabel('PC1', fontsize = 46)
plt.ylabel('PC2', fontsize = 46)
plt.title('PCA Colored by Error',fontsize = 46)
plt.xticks(fontsize=46)
plt.yticks(fontsize=46)
cbar = plt.colorbar(sc)  # create the colorbar
cbar.set_label('Error [e]', fontsize=46)  # label font size
cbar.ax.tick_params(labelsize=38)         # tick label font size
plt.tight_layout()
plt.savefig('pca_by_error.png',dpi=300)
plt.show()