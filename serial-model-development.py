import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV # type: ignore

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import cross_val_score, cross_val_predict
import pickle

train = pd.read_csv("train_FE.csv")
train = train.drop("Unnamed: 0", axis=1)

X = train[train.columns[3:]]
X = X[X.columns.difference(["pickup_day_of_week"])]
X = X[X.columns.difference(["pickup_date"])]
X = X[X.columns.difference(["euc_distance"])]

y = train[train.columns[1:2]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

###################
# Linear regression
###################
lr = LinearRegression()
lr.fit(X_train, y_train)

scores = cross_val_score(lr, X_train, y_train, cv=6)

lr_y_pred = cross_val_predict(lr, X_test, y_test, cv=6)

print("Coefficients: \n", lr.coef_)

rmse = np.sqrt(mean_squared_error(y_test, lr_y_pred))
print("RMSE: %f" % (rmse))
print("Mean squared error: %.2f" % mean_squared_error(y_test, lr_y_pred))
print("Variance score: %.2f" % r2_score(y_test, lr_y_pred))

###################
# Decision tree
###################
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
scores = cross_val_score(dt, X_train, y_train, cv=6)
dt_y_pred = cross_val_predict(dt, X_test, y_test, cv=6)

rmse_dt = np.sqrt(mean_squared_error(y_test, dt_y_pred))
print("RMSE: %f" % (rmse_dt))
print("Mean squared error: %.2f" % mean_squared_error(y_test, dt_y_pred))
print("Variance score: %.2f" % r2_score(y_test, dt_y_pred))

###################
# Random forest
###################
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

rf_y_pred = rf.predict(X_test)

# Cross validation
rmse_rf = np.sqrt(mean_squared_error(y_test, rf_y_pred))
print("RMSE: %f" % (rmse_rf))
print("Mean squared error: %.2f" % mean_squared_error(y_test, rf_y_pred))
print("Variance score: %.2f" % r2_score(y_test, rf_y_pred))

###################
# XGBoost
###################
xgbr = xgb.XGBRegressor()
xgbr.fit(X_train, y_train)
scores = cross_val_score(xgbr, X_train, y_train, cv=6)
xgbr_y_pred = cross_val_predict(xgbr, X_test, y_test, cv=6)
rmse_xgbr = np.sqrt(mean_squared_error(y_test, xgbr_y_pred))
print("RMSE: %f" % (rmse_xgbr))
print("Mean squared error: %.2f" % mean_squared_error(y_test, xgbr_y_pred))
print("Variance score: %.2f" % r2_score(y_test, xgbr_y_pred))

params = {
    "max_depth": [7, 8, 9, 10],
    "eta": [1],
    "silent": [1],
    "objective": ["reg:linear"],
    "eval_metric": ["rmse"],
    "learning_rate": [0.1, 0.15, 0.2],
}

grid_search = GridSearchCV(estimator=xgbr, param_grid=params, cv=5, n_jobs=-1)

grid_search.fit(X_train, y_train)

xgbr2_y_pred = grid_search.predict(X_test)


rmse_xgbr2 = np.sqrt(mean_squared_error(y_test, xgbr2_y_pred))
print("RMSE: %f" % (rmse_xgbr2))
print("Mean squared error: %.2f" % mean_squared_error(y_test, xgbr2_y_pred))
print("Variance score: %.2f" % r2_score(y_test, xgbr2_y_pred))

###################
# Light GBM
###################
lgbm = LGBMRegressor(
    boosting_type="gbdt",
    num_leaves=31,
    learning_rate=0.1,
    max_depth=5,
    n_estimators=100,
    silent=True,
)
lgbm.fit(X_train, y_train)
lgbm_y_predict = lgbm.predict(X_test)

rmse_lbgm = np.sqrt(mean_squared_error(y_test, lgbm_y_predict))
print("RMSE: %f" % (rmse_lbgm))
print("Mean squared error: %.2f" % mean_squared_error(y_test, lgbm_y_predict))
print("Variance score: %.2f" % r2_score(y_test, lgbm_y_predict))

result_df = pd.DataFrame(
    data={
        "Model": [
            "Linear regression",
            "Decision Tree",
            "Random forest",
            "XGBoost",
            "Light GBM",
        ],
        "RMSE": [rmse, rmse_dt, rmse_rf, rmse_xgbr2, rmse_lbgm],
        "MSE": [
            mean_squared_error(y_test, lr_y_pred),
            mean_squared_error(y_test, dt_y_pred),
            mean_squared_error(y_test, rf_y_pred),
            mean_squared_error(y_test, xgbr2_y_pred),
            mean_squared_error(y_test, lgbm_y_predict),
        ],
        "R2 score": [
            r2_score(y_test, lr_y_pred),
            r2_score(y_test, dt_y_pred),
            r2_score(y_test, rf_y_pred),
            r2_score(y_test, xgbr2_y_pred),
            r2_score(y_test, lgbm_y_predict),
        ],
    }
)
result_df
