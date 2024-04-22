from dask_ml.metrics import mean_squared_error, r2_score
import dask.array as da
from distributed import default_client
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LinearRegression
from dask_ml.xgboost import XGBRegressor

# from dask_ml.ensemble import RandomForestRegressor
from dask_ml.model_selection import GridSearchCV
from dask.distributed import Client
import joblib
from sklearn.model_selection import cross_val_score
import dask.array as da
from dask.distributed import Client
import dask_xgboost as dxgb
import xgboost as xgb
from dask_ml.model_selection import GridSearchCV, train_test_split
from dask_ml.metrics import mean_squared_error
from dask_ml.wrappers import ParallelPostFit
import numpy as np
import numpy as np
import dask.array as da
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from dask.distributed import Client
import joblib


# Configure the new number of workers and threads per worker
n_workers = 4  # For example, changing to 8 workers
threads_per_worker = 2  # And adjusting threads per worker

# Start a new client with the updated configuration
client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker)
print("Dashboard link:", client.dashboard_link)
train = dd.read_csv("train_FE.csv", blocksize="64MB")
train = train.drop("Unnamed: 0", axis=1)

X = train[train.columns[3:]]
X = X.drop(["pickup_day_of_week", "euc_distance"], axis=1)
y = train["fare_amount"].to_frame()  # Assuming 'fare_amount' is your target column

# Convert to dask arrays if necessary for some models
X, y = X.to_dask_array(lengths=True), y.to_dask_array(lengths=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


######################
# Linear regression
######################
lr = LinearRegression()
lr.fit(X_train, y_train)

lr_y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, lr_y_pred)
rmse = da.sqrt(mse).compute()  # Compute RMSE from MSE
r2 = r2_score(y_test, lr_y_pred).compute()  # Compute R²

print(f"RMSE: {rmse}")
print(f"R² score: {r2}")


with joblib.parallel_backend("dask"):
    scores = cross_val_score(lr, X_train, y_train, cv=5)

print(f"Cross-validation scores: {scores}")
print(f"Average score: {scores.mean()}")

######################
# Decision tree
######################

dt = DecisionTreeRegressor()

# Since DecisionTreeRegressor does not natively support Dask,
# the fitting process here won't be parallelized beyond what sklearn internally supports (e.g., through joblib for some operations)
dt.fit(X_train.compute(), y_train.compute())

# Parallelize cross-validation using Dask with Joblib
with joblib.parallel_backend("dask"):
    scores = cross_val_score(dt, X_train.compute(), y_train.compute(), cv=6)
    # cross_val_predict is not demonstrated to be parallelized directly via Dask as its primary goal isn't scoring or hyperparameter tuning
    dt_y_pred = cross_val_predict(dt, X_test.compute(), y_test.compute(), cv=6)

# Compute metrics
rmse_dt = np.sqrt(mean_squared_error(y_test.compute(), dt_y_pred))
r2_dt = r2_score(y_test.compute(), dt_y_pred)

print(f"RMSE: {rmse_dt}")
print(f"Mean squared error: {mean_squared_error(y_test.compute(), dt_y_pred)}")
print(f"Variance score: {r2_dt}")


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

###################
# Random forest
###################

rf = RandomForestRegressor()

with joblib.parallel_backend("dask"):
    rf.fit(
        X_train.compute(), y_train.compute()
    )  # `.compute()` ensures data is in memory. Be mindful of large datasets.

# Predict in a parallel computation context
with joblib.parallel_backend("dask"):
    rf_y_pred = rf.predict(
        X_test.compute()
    )  # Ensure X_test is computed if it's a Dask object.

# Calculate metrics
rmse_rf = np.sqrt(
    mean_squared_error(y_test.compute(), rf_y_pred)
)  # Ensure y_test is computed if it's a Dask object.
r2_rf = r2_score(y_test.compute(), rf_y_pred)

print(f"RMSE: {rmse_rf}")
print(f"Mean squared error: {mean_squared_error(y_test.compute(), rf_y_pred)}")
print(f"Variance score: {r2_rf}")

###################
# XGBoost
###################

xgbr = dxgb.XGBRegressor()

# Convert to Dask Array if your data is not already in this format
# X_train, y_train = da.rechunk(X_train, chunks=(1000, X_train.shape[1])), da.from_array(y_train, chunks=1000)

# Train model
xgbr.fit(X_train, y_train)

# Predictions
xgbr_y_pred = xgbr.predict(X_test)


params = {
    "max_depth": [7, 8],
    # "eta": [0.2],
    "objective": ["reg:squarederror"],
    "eval_metric": ["rmse"],
    "learning_rate": [0.2],
}

# Note: dask_ml's GridSearchCV does not require the `n_jobs` parameter as parallelism is managed by Dask
grid_search = GridSearchCV(estimator=xgbr, param_grid=params, cv=5)

grid_search.fit(X_train, y_train_dask)

xgbr2_y_pred = grid_search.predict(X_test.compute())

# Ensure y_test is a NumPy array for comparison
y_test_computed = y_test.compute()

rmse_xgbr2 = np.sqrt(mean_squared_error(y_test_computed, xgbr2_y_pred))
print(f"RMSE: {rmse_xgbr2}")
print(f"Mean squared error: {mean_squared_error(y_test_computed, xgbr2_y_pred)}")
print(f"Variance score: {r2_score(y_test_computed, xgbr2_y_pred)}")


###################
# XGBoost - non deprecated
###################
import dask.array as da
from dask.distributed import Client
import dask_xgboost as dxgb
import xgboost as xgb
from dask_ml.model_selection import GridSearchCV, train_test_split
from dask_ml.metrics import mean_squared_error
from dask_ml.wrappers import ParallelPostFit
import numpy as np
from dask_ml.xgboost import XGBRegressor

xgbr = XGBRegressor()

# Convert to Dask Array if your data is not already in this format
# X_train, y_train = da.rechunk(X_train, chunks=(1000, X_train.shape[1])), da.from_array(y_train, chunks=1000)

# Train model
xgbr.fit(X_train, y_train)

# Predictions
xgbr_y_pred = xgbr.predict(X_test)


params = {
    "max_depth": [7],
    "eta": [0.3],
    "objective": ["reg:squarederror"],
    "eval_metric": ["rmse"],
    # "learning_rate": [0.1],
}

# Note: dask_ml's GridSearchCV does not require the `n_jobs` parameter as parallelism is managed by Dask
grid_search = GridSearchCV(estimator=xgbr, param_grid=params, cv=5)

# rechunking the data
chunks = 1000
X_train = da.rechunk(X_train, chunks=(chunks, X_train.shape[1]))
y_train = da.rechunk(y_train, chunks=chunks)

grid_search.fit(X_train, y_train)

xgbr2_y_pred = grid_search.predict(X_test.compute())

# Ensure y_test is a NumPy array for comparison
y_test_computed = y_test.compute()

rmse_xgbr2 = np.sqrt(mean_squared_error(y_test_computed, xgbr2_y_pred))
print(f"RMSE: {rmse_xgbr2}")
print(f"Mean squared error: {mean_squared_error(y_test_computed, xgbr2_y_pred)}")
print(f"Variance score: {r2_score(y_test_computed, xgbr2_y_pred)}")

###
# XGBoost without grid search
###
