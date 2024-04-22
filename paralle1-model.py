#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dask_ml.metrics import mean_squared_error, r2_score
import dask.array as da
from distributed import default_client
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LinearRegression
from dask_ml.xgboost import XGBRegressor

# from dask_ml.ensemble import RandomForestRegressor
from dask_ml.model_selection import GridSearchCV
from dask.distributed import get_client, Client
import joblib
from sklearn.model_selection import cross_val_score
import dask.array as da
from dask.distributed import Client
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
import dask.array as da
import xgboost as xgb
from dask_ml.model_selection import GridSearchCV, train_test_split
from dask_ml.metrics import mean_squared_error
from dask_ml.wrappers import ParallelPostFit
import numpy as np
from dask_ml.xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[2]:


import dask.dataframe as dd


def print_dask_df_info(dask_df):
    """
    Prints comprehensive information about a Dask DataFrame including:
    - Number of partitions
    - Memory usage of each partition
    - Division information
    - Column data types
    """
    # Ensure the input is a Dask DataFrame
    if not isinstance(dask_df, dd.DataFrame):
        print("The input is not a Dask DataFrame.")
        return

    # Number of partitions
    num_partitions = dask_df.npartitions
    print(f"Number of partitions: {num_partitions}")

    # Memory usage of each partition
    try:
        partition_memory_usage = dask_df.memory_usage(deep=True).compute()
        print("Partition memory usage (in MB):\n", partition_memory_usage / 1024 / 1024)
    except Exception as e:
        print(f"Could not compute memory usage: {e}")

    # # Division information
    # division_info = dask_df.divisions
    # print('Division information:', division_info)

    # Column data types
    dtypes = dask_df.dtypes
    print("Column data types:\n", dtypes)


# In[3]:


from dask.distributed import Client, get_client
from functools import wraps
import time


def with_execution_info(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Ensure a Dask Client is available or create a new one
        try:
            client = get_client()
        except ValueError:
            client = Client()
            print("Initialized a new Dask Client.")

        # Get Dask Worker details
        workers = client.scheduler_info()["workers"]
        cpus = sum(worker["nthreads"] for worker in workers.values())
        threads = len(workers)

        # Measure function execution time
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time

        # Information dictionary
        info = {
            "result": result,
            "execution_time": execution_time,
            "total_cpus": cpus,
            "total_threads": threads,
        }

        return info

    return wrapper


# In[4]:


def get_partition_info(dask_df):
    """
    Retrieves information about the partitions of a Dask DataFrame including:
    - Total number of partitions
    - Number of rows in each partition
    - Estimated memory usage of each partition in bytes

    Parameters:
    - dask_df: A Dask DataFrame

    Returns:
    A summary dictionary with partition information.
    """
    if not isinstance(dask_df, dd.DataFrame):
        raise ValueError("The input must be a Dask DataFrame.")

    # Function to compute rows in each partition
    def count_rows(partition):
        return len(partition)

    # Function to compute memory usage in each partition
    def get_memory_usage(partition):
        return partition.memory_usage(deep=True).sum()

    # Calculating partition info
    num_partitions = dask_df.npartitions
    rows_per_partition = dask_df.map_partitions(count_rows).compute().tolist()
    memory_usage_per_partition = (
        dask_df.map_partitions(get_memory_usage).compute().tolist()
    )

    # Constructing the summary dictionary
    partition_info = {
        "total_partitions": num_partitions,
        "rows_per_partition": rows_per_partition,
        "memory_usage_per_partition_bytes": memory_usage_per_partition,
    }

    return partition_info


# In[5]:


def with_repartition(mem="32MB", *args):
    """
    This function repartitions the input Dask DataFrames/Series to the specified memory size.
    """
    if len(args) == 0:
        raise ValueError("No input Dask DataFrames/Series provided.")

    if not isinstance(mem, str):
        raise ValueError("The memory size must be a string.")
    for i in args:
        if not isinstance(i, (dd.DataFrame, dd.Series)):
            raise ValueError("The input must be a Dask DataFrame or Series.")
        i = i.repartition(partition_size=mem)
    return args


# In[6]:


def create_dask_client(cpus=4, threads=1, memory="2GB"):
    try:
        # Attempt to get the current client
        client = get_client()
        # If successful, restart the client
        print("Restarting existing Dask Client...")
        client.close()
        print("Dask Client restarted.")
    except ValueError:
        pass
        # client = Client(n_workers=cpus, threads_per_worker=2, processes=True, memory_limit='2GB')
    client = Client(
        n_workers=cpus, threads_per_worker=threads, processes=True, memory_limit=memory
    )
    print("Dashboard link:", client.dashboard_link)
    return client


# In[7]:


def load_data(dir="data/*.part", blocksize="32MB", partition_size="32MB"):
    df = dd.read_csv(dir, blocksize=blocksize)
    df = df.repartition(partition_size=partition_size)
    return df


# In[8]:


def prepare_train_test_split(df, test_size=0.3):
    df = df.drop("Unnamed: 0", axis=1)
    X = df[df.columns[3:]]
    X = X.drop(["pickup_day_of_week", "eucledian_distance"], axis=1)
    y = df["fare_amount"].to_frame()
    # todo - check if we need to convert to dask array
    X, y = X.to_dask_array(lengths=True), y.to_dask_array(lengths=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test


# In[9]:


# client = create_dask_client(cpus=4, threads=1, memory='2GB')
# train = load_data('final_data.csv/*.part')
# X_train, X_test, y_train, y_test = prepare_train_test_split(train)

# print_dask_df_info(X_train)
# partition_info = get_partition_info(X_train)


# In[10]:


def linearRegressionWith(
    path="final_data/*.part",
    cpus=1,
    threads=1,
    mem_per_partition="32MB",
    load_block_size="32MB",
    partition_size="32MB",
):

    create_dask_client(cpus=cpus, threads=threads, memory=mem_per_partition)
    train = load_data(path, blocksize=load_block_size, partition_size=partition_size)
    X_train, X_test, y_train, y_test = prepare_train_test_split(train)
    lr = LinearRegression(solver_kwargs={"normalize": False})
    lr_fit = lambda lr, X_train, y_train: lr.fit(X_train, y_train)

    fit = with_execution_info(lr_fit)

    info = fit(lr, X_train, y_train)

    lr_y_pred = lr.predict(X_test)
    mse = mean_squared_error(y_test, lr_y_pred)
    rmse = da.sqrt(mse)  # Compute RMSE from MSE
    r2 = r2_score(y_test, lr_y_pred)  # Compute R²

    print(f"RMSE: {rmse}")
    print(f"R² score: {r2}")

    with joblib.parallel_backend("dask"):
        scores = cross_val_score(lr, X_train, y_train, cv=5)

    print(f"Cross-validation scores: {scores}")
    print(f"Average score: {scores.mean()}")

    info["rmse"] = rmse
    info["r2"] = r2
    info["cross_val_scores"] = scores
    info["average_score"] = scores.mean()
    return info


# In[11]:


# linearRegressionWith('final_data/0000part', cpus=1, threads=1, mem_per_partition='2GB', load_block_size='32MB')


# In[17]:


from sklearn.model_selection import cross_val_score


def xgbWith(
    path="train_data_head.csv",
    cpus=1,
    threads=1,
    mem_per_partition="32MB",
    load_block_size="32MB",
    partition_size="32MB",
):

    create_dask_client(cpus=cpus, threads=threads, memory=mem_per_partition)
    # train = load_data('final_data.csv/0*.part', blocksize=load_block_size, partition_size=partition_size)
    train = load_data(path, blocksize=load_block_size, partition_size=partition_size)
    X_train, X_test, y_train, y_test = prepare_train_test_split(train)
    xgb = XGBRegressor()
    xgb_fit = lambda xgb, X_train, y_train: xgb.fit(X_train, y_train)

    fit = with_execution_info(xgb_fit)

    info = fit(xgb, X_train, y_train)

    xgb_y_pred = xgb.predict(X_test)
    mse = mean_squared_error(y_test, xgb_y_pred)
    rmse = da.sqrt(mse)  # Compute RMSE from MSE
    r2 = r2_score(y_test, xgb_y_pred)  # Compute R²

    print(f"RMSE: {rmse}")
    print(f"R² score: {r2}")

    scores = cross_val_score(xgb, X_train, y_train, cv=5)

    print(f"Cross-validation scores: {scores}")
    print(f"Average score: {scores.mean()}")

    info["rmse"] = rmse
    info["r2"] = r2
    info["cross_val_scores"] = scores
    info["average_score"] = scores.mean()
    return info


# In[19]:


info_1_1_32 = xgbWith(
    "train_data_head.csv",
    cpus=1,
    threads=1,
    mem_per_partition="2GB",
    load_block_size="32MB",
    partition_size="32MB",
)


# In[15]:


info_8_1_64 = xgbWith(
    "train_data_head.csv",
    cpus=4,
    threads=1,
    mem_per_partition="2GB",
    load_block_size="64MB",
    partition_size="64MB",
)


# In[16]:


info_8_1_64


# In[ ]:
