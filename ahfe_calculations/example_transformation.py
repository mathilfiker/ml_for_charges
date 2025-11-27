from rdkit import Chem
from openff.toolkit import Molecule
import openfe
from openfe.protocols.openmm_afe import AbsoluteSolvationProtocol
import pathlib
import glob
import os
from pint import UnitRegistry
import numpy as np
from openff.units import unit, Quantity

sdf_file = 'molecules/sdf/mobley_36119.sdf'
suppl = Chem.SDMolSupplier(sdf_file,removeHs=False)
off_molecule = Molecule.from_rdkit(suppl[0])
# Load predicted charges and assign them to the Molecule
magnitudes = np.load('esp_charges/mobley_36119.npy')
ureg=UnitRegistry()
charges = magnitudes*ureg.elementary_charge
off_molecule.partial_charges =Quantity(magnitudes,unit.elementary_charge)
ligand = openfe.SmallMoleculeComponent.from_openff(off_molecule)
solvent = openfe.SolventComponent()
systemA = openfe.ChemicalSystem({
    'ligand': ligand,
    'solvent': solvent,
}, name=ligand.name)

systemB = openfe.ChemicalSystem({'solvent': solvent})
settings = AbsoluteSolvationProtocol.default_settings()
settings.solvent_engine_settings.compute_platform='CUDA'
settings.protocol_repeats = 4
settings.lambda_settings.lambda_elec = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,1.0,1.0]
settings.lambda_settings.lambda_vdw = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.10,0.20,0.30,0.40,0.50,0.60,0.65,0.70,0.75,0.80,0.90,1.0]
settings.lambda_settings.lambda_restraints = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
settings.solvent_simulation_settings.n_replicas=22
settings.vacuum_simulation_settings.n_replicas=22
settings.solvent_forcefield_settings.small_molecule_forcefield='gaff-2.11'
settings.vacuum_forcefield_settings.small_molecule_forcefield='gaff-2.11'
settings.solvation_settings.solvent_padding=1.5 * unit.nanometer
protocol = AbsoluteSolvationProtocol(settings=settings)
transformation = openfe.Transformation(
        stateA=systemA,
        stateB=systemB,
        mapping=None,
        protocol=protocol,  # use protocol created above
        name=f"{systemA.name}"
    )
transformation.dump(f"{transformation.name}.json")