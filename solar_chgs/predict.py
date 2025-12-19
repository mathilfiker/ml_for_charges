import numpy as np
import pandas as pd
import h5py
import sys
import jax
import jax.numpy as jnp
import mlff
from mlff.config import from_config
from ml_collections import config_dict
from orbax import checkpoint
import ase
import pathlib
import json
from ase.io import read
from ase import Atoms
from rdkit import Chem
import xgboost as xgb

def charge_norm(qs,delta_q):
    new_q = []
    add = delta_q/len(qs)
    for i in qs:
        new_q.append(i-add)
    return new_q

def load_hyperparameters(workdir: str):
    with open(pathlib.Path(workdir) / "hyperparameters.json", "r") as fp:
        cfg = json.load(fp)

    cfg = config_dict.ConfigDict(cfg)
    return cfg

def load_representation_model_from_workdir(
        workdir: str,
        model='so3krates',
        from_file: bool = False
):
    cfg = load_hyperparameters(workdir)

    if from_file is True:
        import pickle

        with open(pathlib.Path(workdir) / 'params.pkl', 'rb') as f:
            params = pickle.load(f)
    else:
        loaded_mngr = checkpoint.CheckpointManager(
            pathlib.Path(workdir) / "checkpoints",
            item_names=('params',),
            item_handlers={'params': checkpoint.StandardCheckpointHandler()},
            options=checkpoint.CheckpointManagerOptions(step_prefix="ckpt"),
        )

        mngr_state = loaded_mngr.restore(
            loaded_mngr.latest_step()
        )

        params = mngr_state.get('params')

    if model == 'so3krates':
        net = from_config.make_so3krates_sparse_from_config(
            cfg,
            return_representations_bool=True
        )
    # elif model == 'itp_net':
    #     net = from_config.make_itp_net_from_config(cfg)
    else:
        raise ValueError(
            f'{model=} is not a valid model.'
        )

    return net, params


def make_so3lr_representation_fn(
    workdir: str,
    params_are_input: bool = False
):
    so3lr, _params = load_representation_model_from_workdir(
        workdir=workdir,
        from_file=True
    )
    
    def representation_fn(params, inputs):
        
        return so3lr.apply(
            params,
            inputs
        )
    
    # params are an explicit input, e.g. interesting for finetuning follow up layers.
    if params_are_input:
        return representation_fn
    else:
        # params are not an explicit input. The function is just graph -> representations.
        return lambda graph: representation_fn(_params, graph)


def atoms_to_inputs(atoms):
    positions = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    total_charge = np.array([0])
    num_unpaired_electrons = np.array([0])
    
    if (atoms.pbc == False).all():
        idx_i, idx_j = ase.neighborlist.neighbor_list(
            'ij', 
            atoms,
            cutoff=4.5
        )
        cell_offset = None
        cell = None
    else:
        idx_i, idx_j, cell_offset = ase.neighborlist.neighbor_list(
            'ijS', 
            atoms,
            cutoff=4.5
        )
        cell = np.array(atoms.cell)
        # For the internal calculations in SO3krates 
        # the cell must be repeated for each edge in the system.
        cell = np.repeat(cell[None], axis=0, repeats=len(idx_i))

    inputs = dict(
        atomic_numbers=atomic_numbers,
        positions=positions,
        idx_i=idx_i,
        idx_j=idx_j,
        cell=cell,
        cell_offset=cell_offset,
        total_charge=total_charge,
        num_unpaired_electrons=num_unpaired_electrons,
        graph_mask=np.array([True]),
        batch_segments=np.zeros((len(atoms), ), dtype=int)
    )
    
    return inputs

path_to_so3lr = pathlib.Path("your_path_to_so3lr_environment") # replace with your installation of the sol3r environment
model = xgb.XGBRegressor()
model.load_model('model/xgb_chgs_so3krates.json')

# Reading the molecule

molecule = read('mobley_36119.xyz') #xyz files are optimized with UFF and this slightly improve calculations 
#molecule =read('mobley_36119.sdf') #.sdf format is supported as well

# Computing SO3KRATES (SOL3R) descriptors
so3lr_representation_fn = jax.jit(
    make_so3lr_representation_fn(
        workdir=path_to_so3lr / 'so3lr/so3lr/params/'
    )
)
# transform atoms to inputs.
inputs = atoms_to_inputs(molecule)
# execute the representations_fn.
dna_representations = so3lr_representation_fn(
    jax.tree.map(
        jnp.array, inputs
    )  # make jnp.arrays from the np.arrays
)            
descr = []
descriptor_solar = dna_representations['atomic_representations']
for j in range(len(descriptor_solar)):
    descr.append(descriptor_solar[j])

#Feeding the descriptors to the model
predictions = model.predict(descr)

#Distributing the extra charge 
delta = np.sum(predictions)
new_q = charge_norm(predictions,delta) 
np.save('chgs_mobley_36119',new_q)
