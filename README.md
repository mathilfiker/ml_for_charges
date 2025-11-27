Repository containing the charge prediction model and the scripts to reproduce the analyses shown in the paper "Machine-Learned Electrostatic Potentials for Accurate Hydration Free Energy Calculations".

To use the model it is necessary to have a copy of the MACEOFF23-large model available here: https://github.com/ACEsuit/mace-off .

To reproducee the analyses, the following is necessary:
- FreeSolv dataset: https://github.com/MobleyLab/FreeSolv
- Aquamarine dataset: https://zenodo.org/records/10208010
- MACE descriptors + ESP charges for Aquamarine molecules: *add link*

The exact environment as used for calculations and analysis can be retrieved from the file **environment.yml**

The repository is structured as follow:
-  **ahfe_calculations** contains all the necessary to reproduce the parity plots shown in the paper: AM1-BCC, ESP, RESP, 1-shot, and Boltzmann percentile charges, as well as energies and errors computed with each method. The subfolder "molecules" contains the .sdf and .xyz files for the selected molecules.
- **boltzmann_percentile** contains an example of MD simulations and BP charge assignment for one molecule, and the script to replicate the analysis of charge variability across conformations.
- **charge_prediction** contains the model, an example of how to use it to predict charges, and scripts to train and validate the model on a subset of the real training data.
- **functional_groups_analysis** contains the script to replicate the analaysis on the functional groups.
