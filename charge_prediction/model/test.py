import pandas as pd
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error as rmse
import sys
test = pd.read_csv('../data/subset_test.csv',delimiter=",")
X_test = test.iloc[:,3:451].values
y_test = test.iloc[:,-1]
model = xgb.XGBRegressor()
model.load_model('xgb_chgs.json')
y_pred = model.predict(X_test)
print("Test error (RMSE): ", rmse(y_test,y_pred))
sys.stdout.flush()