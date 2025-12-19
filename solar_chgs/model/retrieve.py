import optuna
import sys
DB_FILE = 'sqlite:///optuna_hyperparameter_optimization.db'
study = optuna.load_study(study_name='xgb_optimization', storage = DB_FILE)
best_trial = study.best_trial
print("Best hyperparameters:")
for key, value in best_trial.params.items():
    print(f"{key}: {value}")

print(f"Best RMSE: {best_trial.value}")
sys.stdout.flush()
