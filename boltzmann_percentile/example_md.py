from rdkit import Chem
from openmm import app, openmm, unit
from openmmforcefields.generators import GAFFTemplateGenerator
from openff.toolkit import Molecule
from openmm.app import ForceField
from openmm import Platform
import mdtraj as md
import os

sdf_file = "../ahfe_calculations/molecules/sdf/mobley_36119.sdf"
mol_supplier = Chem.SDMolSupplier(sdf_file,removeHs=False)
mol = mol_supplier[0]
Chem.MolToPDBFile(mol,'mobley_36119.pdb')
molecules = Molecule.from_file('../ahfe_calculations/molecules/sdf/mobley_36119.sdf')
gaff = GAFFTemplateGenerator(molecules=molecules, forcefield="gaff-2.11")
# forcefield location (search exact path in your environment)
forcefield = ForceField("your_path_to_openfe_environment/lib/python3.10/site-packages/openmmforcefields/ffxml/amber/gaff/ffxml/gaff-2.11.xml")
forcefield.registerTemplateGenerator(gaff.generator)
pdb = app.PDBFile("mobley_36119.pdb")
system = forcefield.createSystem(pdb.topology)
# Define integrator (Langevin dynamics at 300K)
integrator = openmm.LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2.0 * unit.femtosecond)

# Set up simulation with GPU (recommended)
platform = Platform.getPlatformByName('CUDA') # comment this line for running with CPU
properties = {'DeviceIndex': '0', 'Precision': 'double'} # comment this line for running with CPU
simulation = app.Simulation(pdb.topology, system, integrator,platform, properties) # comment this line for running with CPU
#simulation = app.Simulation(pdb.topology, system, integrator) uncommment this line for running with CPU
simulation.context.setPositions(pdb.positions)

# Minimize energy
simulation.minimizeEnergy()
# Run MD simulation (1 ns) 
simulation.reporters.append(app.DCDReporter("trajectory_36119.dcd", 1000))
simulation.step(500000)  # 1 ns @ 2 fs timestep

print("Gas-phase MD simulation completed.")