import pandas as pd
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
import optuna
import sys

# Load data
data = pd.read_csv('../data/subset_training.csv',delimiter=",",header=None)
X_train = data.iloc[:, :-1]
y_train = data.iloc[:, -1]
test = pd.read_csv('../data/subset_validation.csv',delimiter=",",header=None)
X_test = test.iloc[:,:-1]
y_test = test.iloc[:,-1]
# Load Optuna study
#DB_FILE = 'sqlite:///optuna_hyperparameter_optimization.db' # Name of the already optimised file
DB_FILE = 'sqlite:///example_hyperparameter_optimization.db' # for optimization on the subset
study = optuna.load_study(study_name='xgb_optimization', storage = DB_FILE)
best_trial = study.best_trial
# Define best model
best_params = best_trial.params
model = xgb.XGBRegressor(**best_params)
model.fit(X_train,y_train)
#model.save_model('xgb_chgs_on_subset.json') # uncomment to save the model trained on the subset
y_pred = model.predict(X_test)
print("RMSE: ",root_mean_squared_error(y_test,y_pred))
sys.stdout.flush()
