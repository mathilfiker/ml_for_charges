import MDAnalysis as mda
from rdkit import Chem
from rdkit.Chem import AllChem
from ase.io import read,write
import xgboost as xgb
from mace.calculators import MACECalculator
import numpy as np
from openmm import app, openmm, unit
from openmmforcefields.generators import GAFFTemplateGenerator
from openff.toolkit import Molecule
from openmm.app import ForceField
import mdtraj as md
import os
import glob
from matplotlib import pyplot as plt
from ase import Atoms
from sklearn.metrics import root_mean_squared_error as rmse
import sys
from openmm.app import *
from openmm.unit import *
from openmm import app, openmm, unit
from openmm import Platform

sys.stdout.flush()
def charge_norm(qs,delta_q):
    new_q = []
    add = delta_q/len(qs)
    for i in qs:
        new_q.append(i-add)
    return new_q
 
calculator = MACECalculator(model_paths='your_path_to_mace/MACE-OFF23_large.model',device='cpu') # Replace with the path from your installation
model = xgb.XGBRegressor()
model.load_model('../charge_prediction/model/xgb_aqm_gas.json')
kB = 0.001987  # kcal/(mol*K)
T = 300  # Kelvin
at = read(f'mobley_36119.pdb')
pdb = app.PDBFile(f"mobley_36119.pdb")
atoms = at.get_atomic_numbers()
u = mda.Universe(f"mobley_36119.pdb",f"trajectory_36119.dcd")
molecules = Molecule.from_file(f'../ahfe_calculations/molecules/sdf/mobley_36119.sdf')
gaff = GAFFTemplateGenerator(molecules=molecules, forcefield="gaff-2.11")
# Select a force field
forcefield = ForceField("your_path_to_openfe_environment/lib/python3.10/site-packages/openmmforcefields/ffxml/amber/gaff/ffxml/gaff-2.11.xml") # Replace with the path from your installation
forcefield.registerTemplateGenerator(gaff.generator)
# Define a system (obejct implementing the force field for a topology)
system = forcefield.createSystem(pdb.topology, nonbondedCutoff=1 * nanometer)
# Define an integrator
integrator = openmm.LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2.0 * unit.femtosecond)
# Use GPU (recommended)
platform = Platform.getPlatformByName('CUDA')
properties = {'DeviceIndex': '0', 'Precision': 'double'}
simulation = Simulation(topology, system, integrator,platform, properties)
chgs_min =[]
energy_results = []
for ts in u.trajectory:
    comodo = u.atoms
    positions = comodo.positions
    simulation.context.setPositions(positions * angstroms)
    state0 = simulation.context.getState(getEnergy=True)
    pos0 = simulation.context.getState(getPositions=True)
    # Minimize the energy.
    simulation.minimizeEnergy()
    # Get charges on minimized energy.
    state1 = simulation.context.getState(getEnergy=True)
    energy_results.append(state1.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))
    pos1 = simulation.context.getState(getPositions=True)
    new_pos = pos1.getPositions().value_in_unit(unit.angstroms)
    pos_to_pred = []
    for j in range(len(new_pos)):
        pos_to_pred.append([new_pos[j][0],new_pos[j][1],new_pos[j][2]])
    to_pred = Atoms(symbols=atoms,positions=pos_to_pred)
    to_pred.set_calculator(calculator)
    descriptor_mace = calculator.get_descriptors(to_pred)
    predictions = model.predict(descriptor_mace)
    delta = np.sum(predictions)
    chgs_conf = charge_norm(predictions,delta)
    chgs_min.append(chgs_conf)
energy_results = np.array(energy_results)
chgs_min = np.array(chgs_min)
weights = np.exp(-(energy_results - np.min(energy_results)) / (kB*T)) #unnormalized
np.save(f'.all_charges_mobley_36119',chgs_min)
np.save(f'.all_weights_mobley_36119',weights)
