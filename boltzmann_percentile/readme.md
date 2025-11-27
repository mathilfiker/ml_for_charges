Contents of the folder:
- **example_md.py** to run MD on the example molecule mobley_36119. Running this script will generate as output *trajectory_36119.dcd* and *mobley_36119.pdb* .
- **get_chgs_and_weights.py** to analyze the trajectory file. This script computes Boltzmann weight and predicts charges with the proposed model, and returns as output the files *all_charges_mobley_36119.npy* and *all_weights_mobley_36119.npy* .
- **percentile.py** to read charges and weights arrays and apply the BP assignment as explained in the main text. The output is the array *percentile_mobley_36119.npy* .
- **conformational flexibility**: folder containing the script *plot_charge_variability.py* to reproduce the plots in Fig.4 of the main text.
