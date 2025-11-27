from ase import Atoms
import numpy as np
from ase.io import read, write
import pandas as pd
from mace.calculators import MACECalculator
import h5py
import joblib
import sys
import xgboost as xgb

# Function to redistribute charge
def charge_norm(qs,delta_q):
    new_q = []
    add = delta_q/len(qs)
    for i in qs:
        new_q.append(i-add)
    return new_q

# Defining theCalculator for the descriptors and loading the model
calculator = MACECalculator(model_paths='your_path_to_mace/MACE-OFF23_large.model',device='cpu') # replace with your installation of the model
model = xgb.XGBRegressor()
model.load_model('model/xgb_chgs.json')

# Reading the molecule

molecule = read('mobley_36119.xyz') #xyz files are optimized with UFF and this slightly improve calculations 
#molecule =read('mobley_36119.sdf') #.sdf format is supported as well

# Computing MACE descriptors
descriptor_mace = calculator.get_descriptors(molecule)

#Feeding the descriptors to the model
predictions = model.predict(descriptor_mace)

#Distributing the extra charge 
delta = np.sum(predictions)
new_q = charge_norm(predictions,delta) 
np.save('chgs_mobley_36119',new_q)
