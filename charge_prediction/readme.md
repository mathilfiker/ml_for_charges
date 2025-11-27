Content of the folder:
- **data**: folder containing a suubset of the training, validation, and test sets to rapidly run the analyses
- **model**: folder containing scripts to run Optuna optimization (*optimization_xgb.py*), retrieve the best hyperparameters (*retrieve.py*), train with those parameters (*train.py*), test the trained model (*test.py*), and the trained model itself (*xgb_chgs.json*).
- **validation**: folder containing scripts to reproduce the PCAs in Fig.3 (*make_pca.py*) and Table 1 (*err_er_species.py*) in the main text.
- **predict.py** to show how to use the trained model to predict charges on the example molecule *mobley_36119.xyz*. This script outputs charges in a numpy array in the same order as the atoms of the query molecule.
