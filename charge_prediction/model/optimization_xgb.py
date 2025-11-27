import optuna 
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np
import os
import pandas as pd

# Set up a persistent SQLite database for Optuna to track progress
#DB_FILE = 'sqlite:///optuna_hyperparameter_optimization.db' # Name of the already optimised file
DB_FILE = 'sqlite:///example_hyperparameter_optimization.db' # for running optimization on the subset

data = pd.read_csv('../data/subset_training.csv',delimiter=",",header=None)
X_train = data.iloc[:, :-1]
y_train = data.iloc[:, -1]
test = pd.read_csv('../data/subset_validation.csv',delimiter=",",header=None)
X_test = test.iloc[:,:-1]
y_test = test.iloc[:,-1]

def objective(trial):
    # Define the search space for hyperparameters
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_uniform('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-2, 10),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-2, 10)
    }

    # Create an XGBRegressor model
    model = XGBRegressor(**param, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)
   
    # Compute RMSE
    rmse = root_mean_squared_error(y_test,y_pred)
    return rmse

# Connect Optuna to the SQLite database
study = optuna.create_study(direction='minimize', study_name='xgb_optimization', storage=DB_FILE, load_if_exists=True)

# Set the number of trials to run
study.optimize(objective, n_trials=50, n_jobs=1)  # 'n_jobs=1' since this script is running in parallel
