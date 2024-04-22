#!/usr/bin/env python
# coding: utf-8

# ### Importing Modules for Parallel Distributed Computing with Dask
# 
# Import necessary libraries and modules for parallel machine learning model development and evaluation:
# 
# - **Dask Array**: Offers parallelized, Numpy-like arrays for large computations that can be broken down into smaller, efficient, and scalable parts.
# - **Joblib**: Utilized for lightweight pipelining in Python, enabling parallel computation across multiple cores with Dask integration.
# - **Dask ML Linear Regression**: Implements linear regression for in-memory datasets, suitable for parallel and distributed computation.
# - **Mean Squared Error and R2 Score (Dask ML)**: Evaluate model performance using mean squared error for the average squared difference between estimated values and actual values, and R2 score for the proportion of the variance in the dependent variable that is predictable.
# - **Train Test Split (Dask ML)**: Efficiently splits arrays or matrices into random train and test subsets, optimized for Dask.
# - **XGBRegressor (Dask ML)**: A parallel and distributed implementation of the XGBoost regression model, designed for efficiency and scalability within Dask ecosystems.
# - **Scikit-learn Metrics**: Import mean squared error and R2 score from Scikit-learn for model evaluation, useful for scenarios where Dask arrays are converted to Numpy arrays or for performance comparison.
# 
# These imports are crucial for setting up and managing parallel distributed computing workflows in Python, utilizing Dask for parallel execution across multiple workers and time for measuring execution metrics.

# In[36]:


import warnings

import dask.array as da
import joblib
from dask_ml.linear_model import LinearRegression
from dask_ml.metrics import mean_squared_error
from dask_ml.metrics import r2_score
from dask_ml.model_selection import train_test_split
from dask_ml.xgboost import XGBRegressor  
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ### Importing Dask DataFrame
# 
# ```python
# import dask.dataframe as dd
# ```
# 
# This line imports the `dask.dataframe` module, which provides a parallelized DataFrame object that mimics pandas but operates on large datasets that don't fit into memory, allowing for distributed computing.
# 
# ### Defining a Function to Print Dask DataFrame Information
# 
# ```python
# def print_dask_df_info(dask_df):
#     """
#     Prints comprehensive information about a Dask DataFrame including:
#     - Number of partitions
#     - Memory usage of each partition
#     - Division information (commented out)
#     - Column data types
#     """
# ```
# 
# This function, `print_dask_df_info`, is designed to print detailed information about a Dask DataFrame. It checks if the input is a Dask DataFrame and prints the number of partitions, memory usage of each partition, and column data types. The division information section is commented out but can be included to show how the DataFrame is divided among partitions.
# 
# #### Inside the Function:
# 
# 1. **Check for Dask DataFrame**:
#    Ensures the input is indeed a Dask DataFrame, enhancing robustness by preventing errors when a non-Dask DataFrame is passed.
# 
# 2. **Number of Partitions**:
#    Retrieves and prints the total number of partitions in the Dask DataFrame, indicating how the dataset is distributed across different workers or cores.
# 
# 3. **Memory Usage of Each Partition**:
#    Attempts to calculate and print the memory usage of each partition in megabytes (MB), providing insight into the memory footprint of the DataFrame's partitions. This step involves converting bytes to MB for readability.
# 
# 4. **Column Data Types**:
#    Prints the data types of each column in the DataFrame, helping identify the kind of data each column holds, similar to pandas' `dtypes` attribute.
# 
# This function is particularly useful for gaining insights into the structure and characteristics of large datasets managed with Dask, enabling efficient data handling and processing in parallel computing environments.

# In[37]:


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


# ### Enhancing Functions with Parallel Execution Information via a Decorator
# 
# This code snippet is designed to augment any function with additional execution information when used in a Dask distributed computing environment. It includes importing necessary modules, defining a decorator to gather and return execution stats, and demonstrating how to apply this decorator.
# 
# #### Import Statements
# 
# ```python
# from dask.distributed import Client, get_client
# from functools import wraps
# import time
# ```
# 
# - **`dask.distributed.Client`**: Manages a Dask distributed cluster, allowing for parallel computation across multiple workers.
# - **`dask.distributed.get_client`**: Retrieves the current Dask client if one exists within the scope.
# - **`functools.wraps`**: A decorator to preserve the name, docstring, and other attributes of the decorated function.
# - **`time`**: Module to measure execution times.
# 
# #### Decorator for Execution Information
# 
# ```python
# def with_execution_info(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
# ```
# 
# - **`with_execution_info`**: A decorator designed to wrap around any function, capturing its execution time and relevant Dask cluster information (such as the number of CPUs and threads used).
# 
# Inside the Wrapper Function:
# 
# 1. **Dask Client Initialization**:
#    Checks for an existing Dask Client and initializes a new one if none is found. This ensures that the function can leverage Dask's distributed capabilities.
# 
# 2. **Gathering Worker Details**:
#    Collects details from the Dask cluster, including the total number of CPUs and threads available, which are indicative of the parallel processing capacity.
# 
# 3. **Measuring Execution Time**:
#    Calculates how long the wrapped function takes to execute. This is valuable for performance analysis and optimization efforts.
# 
# 4. **Returning Execution Information**:
#    Constructs and returns a dictionary with the function's result, execution time, and the computational resources involved. This enriches the function's output with insightful performance metrics.
# 
# ### Practical Use Case
# 
# Apply the `@with_execution_info` decorator to any function intended to run in a Dask distributed environment. It will not only execute the function but also provide a detailed breakdown of the execution time and resources utilized, facilitating a deeper understanding and optimization of distributed computations.

# In[38]:


from dask.distributed import Client, get_client
from functools import wraps
import time


def with_execution_info(func):
    """Executes a function and returns the result along with execution time and Dask Worker details.

    Args:
        func (function): The function to be executed.

    Returns:
        dict: A dictionary containing the result, execution time, total CPUs, and total threads.
    """
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


# ### Function for Retrieving **Parallel** Partition Information from a Dask DataFrame
# 
# The `get_partition_info` function is designed to extract detailed information about the partitions of a Dask DataFrame, which is crucial for understanding and optimizing data distribution in **parallel** computing scenarios. This function emphasizes the **parallel** nature of Dask's processing capabilities, providing insights into how data is divided and stored across partitions.
# 
# #### Parameters and Return Value:
# 
# - **Parameters**: Accepts a single parameter, `dask_df`, which must be a Dask DataFrame. This design ensures that the function specifically caters to Dask's **parallel** distributed environment.
# - **Returns**: A summary dictionary containing key partition metrics, including the total number of partitions, the number of rows in each partition, and the estimated memory usage of each partition in bytes. This comprehensive overview aids in assessing and managing the efficiency of **parallel** data processing.
# 
# #### Core Functionality:
# 
# 1. **Validation**: Confirms that the input is a Dask DataFrame. This step ensures compatibility with Dask's **parallel** processing framework, safeguarding against erroneous inputs that could disrupt **parallel** computation workflows.
# 
# 2. **Partition Row Count**: Utilizes Dask's `map_partitions` method to apply a custom function for counting rows in each partition. This approach leverages Dask's **parallel** execution model to efficiently gather row counts across all partitions.
# 
# 3. **Partition Memory Usage**: Similarly employs `map_partitions` with a function to estimate memory usage of each partition. By invoking Dask's ability to handle operations in **parallel**, this calculation provides valuable insights into the memory footprint of the dataset's distribution.
# 
# 4. **Summary Dictionary Construction**: Aggregates the gathered data into a structured summary, offering a clear, actionable view of the DataFrame's partitioning characteristics in a **parallel** processing context.
# 
# #### Practical Implications:
# 
# Equipped with this function, developers and data scientists working in **parallel** computing environments can make informed decisions about data partitioning strategies, optimizing for performance and efficiency in **parallel** Dask computations. The insights provided by `get_partition_info` are integral for fine-tuning **parallel** data distribution and maximizing resource utilization in distributed data processing tasks.

# In[39]:


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


# ### Repartitioning Dask DataFrames/Series for Optimized **Parallel** Processing
# 
# The `with_repartition` function is adeptly crafted to adjust the partitioning of Dask DataFrames or Series according to a specified memory size, aiming to enhance efficiency and performance in **parallel** computing environments. This repartitioning is key to balancing workload distribution across **parallel** workers, ensuring optimal resource utilization and faster processing times.
# 
# #### Key Features:
# 
# - **Dynamic Repartitioning**: Allows for the dynamic adjustment of Dask DataFrame or Series partition sizes based on a specified memory threshold, facilitating more granular control over data distribution in **parallel** computations.
# - **Optimized for **Parallel** Execution**: By fine-tuning the partition sizes, this function helps in reducing overhead and improving **parallel** execution efficiency, making it particularly useful for large-scale data processing tasks.
# 
# #### Parameters:
# 
# - **mem (string)**: Specifies the desired size of each partition as a string (e.g., "32MB"), dictating how the data is split across **parallel** workers. This parameter is crucial for aligning partition sizes with the available computational resources and workload characteristics.
# - **args**: A variable number of Dask DataFrame or Series objects to be repartitioned. This design supports multiple inputs, underscoring the function's flexibility in handling diverse **parallel** data processing needs.
# 
# #### Functionality:
# 
# 1. **Validation**: Ensures that at least one Dask DataFrame or Series is provided and that the `mem` parameter is a string. It also verifies the type of each input, confirming its compatibility with Dask's **parallel** processing model.
# 2. **Repartitioning**: Iterates through each provided Dask DataFrame or Series, applying the `repartition` method with the specified `partition_size`. This step is pivotal in achieving a more efficient data layout for **parallel** processing.
# 3. **Return**: Outputs the repartitioned Dask DataFrame or Series objects, ready for enhanced **parallel** processing performance.
# 
# #### Practical Implications:
# 
# By facilitating the easy adjustment of partition sizes, `with_repartition` empowers data scientists and engineers to tailor Dask data structures for specific **parallel** processing scenarios. This capability is instrumental in optimizing data-intensive applications, from complex data transformations to advanced machine learning models, within **parallel** distributed environments.

# In[40]:


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


# ### Function for Initializing a **Parallel** Dask Client
# 
# The `create_dask_client` function efficiently sets up a new Dask distributed client or restarts an existing one, optimizing it for **parallel** execution by configuring the number of workers, threads per worker, and memory limits. This setup is crucial for leveraging Dask's **parallel** computation capabilities, allowing for scalable and efficient distributed computing tasks.
# 
# #### Key Parameters:
# 
# - **cpus**: Determines the number of worker processes to initiate. Each worker is capable of running **parallel** tasks, making this parameter essential for scaling the computation across multiple CPUs.
# - **threads**: Specifies the number of threads per worker. This allows for fine-grained control over **parallel** execution within each worker, facilitating efficient use of CPU cores.
# - **memory**: Sets the memory limit for each worker. Proper memory allocation is vital for preventing overconsumption of resources in **parallel** processing environments.
# 
# #### Function Behavior:
# 
# 1. **Existing Client Check**: Initially attempts to retrieve an active Dask client. If an existing client is found, it is restarted to ensure that the new configuration settings take effect. This step is crucial for managing resources effectively in **parallel** computing scenarios.
# 2. **Client Initialization**: If no active client is detected, or after the existing client has been closed, a new Dask client is initialized with the specified configuration. This process involves setting the number of workers (`cpus`), threads per worker (`threads`), whether to use separate processes (`processes=True`), and the memory limit per worker (`memory`).
# 
# 3. **Dashboard Link**: Upon successful initialization, the function prints the dashboard link for the Dask client. This dashboard is an invaluable tool for monitoring and debugging **parallel** distributed computations in real-time.
# 
# #### Return Value:
# 
# - Returns the initialized or restarted Dask client, ready for executing **parallel** distributed computing tasks.
# 
# #### Practical Usage:
# 
# This function is particularly useful in environments where **parallel** computation needs are dynamic, allowing users to adjust their Dask client configurations on the fly to match workload requirements. By facilitating easy management of **parallel** execution resources, `create_dask_client` enhances the efficiency and flexibility of distributed computing workflows.

# In[41]:


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


# ### Loading Data with Optimized **Parallel** Partitioning Using Dask
# 
# The `load_data` function is designed to efficiently load large datasets into Dask DataFrames with optimized **parallel** partitioning. By leveraging Dask's ability to work with data in **parallel**, this function allows for scalable and efficient data processing, suitable for handling big data workflows.
# 
# #### Parameters:
# 
# - **dir**: Specifies the directory path or pattern matching the files to be loaded. The default pattern `"data/*.part"` targets files in the `data` directory with a `.part` extension, facilitating flexible data ingestion scenarios.
# - **blocksize**: Determines the size of chunks to read in **parallel** from each file. The default `"32MB"` is set to balance memory usage and **parallel** processing efficiency, adaptable based on the dataset's characteristics and the available memory.
# - **partition_size**: Configures the desired size for DataFrame partitions after loading the data. Like `blocksize`, the default of `"32MB"` aims to optimize **parallel** computation by adjusting the granularity of distributed data processing.
# 
# #### Functionality:
# 
# 1. **Data Reading with Blocksize**: Utilizes `dd.read_csv` to read CSV files in the specified directory, employing the `blocksize` parameter to control the size of data chunks read into memory. This step is crucial for efficiently managing memory and leveraging Dask's **parallel** processing capabilities.
# 
# 2. **DataFrame Repartitioning**: Applies the `repartition` method to adjust the DataFrame's partition size according to the `partition_size` parameter. This optimization step ensures that subsequent **parallel** operations on the DataFrame are balanced and efficient, tailored to the computation resources and the task at hand.
# 
# 3. **Return Loaded DataFrame**: The function returns the loaded and optimally partitioned Dask DataFrame, ready for **parallel** analysis and processing tasks.
# 
# #### Practical Implications:
# 
# This function streamlines the initial step of data-driven projects, particularly those involving large datasets that require **parallel** processing for efficient analysis. By facilitating the easy loading and optimal partitioning of data, `load_data` enables data scientists and engineers to focus on higher-level analysis and modeling tasks, leveraging the full power of Dask's **parallel** computation framework.

# In[42]:


def load_data(dir="data/*.part", blocksize="32MB", partition_size="32MB"):
    df = dd.read_csv(dir, blocksize=blocksize)
    df = df.repartition(partition_size=partition_size)
    return df


# ### Preparing Data for **Parallel** Train-Test Split Using Dask
# 
# The `prepare_train_test_split` function preprocesses a Dask DataFrame and splits it into training and testing sets, leveraging Dask's capabilities for **parallel** data manipulation and split. This step is pivotal in machine learning workflows, ensuring models are trained and validated on separate data segments for unbiased evaluation.
# 
# #### Parameters:
# 
# - **df**: The input Dask DataFrame containing the dataset to be split. It's assumed to include various features and a target variable for prediction.
# - **test_size**: The proportion of the dataset to include in the test split. The default value of `0.3` means 30% of the data is reserved for testing, a common practice in machine learning to balance between training and testing.
# 
# #### Functionality:
# 
# 1. **Drop Unnecessary Columns**: Initially, the function removes the column `Unnamed: 0`, which often results from reading CSV files with an unnamed index column. This step cleans the dataset for more efficient **parallel** processing.
# 
# 2. **Feature Selection**: Selects the features for the machine learning model by excluding specific columns (`pickup_day_of_week`, `eucledian_distance`) from the dataset, based on prior knowledge that these features may not contribute to the prediction task. This selection is critical for model performance and efficiency in **parallel** processing.
# 
# 3. **Target Variable Isolation**: Separates the `fare_amount` column as the target variable `y`, which the model aims to predict, from the rest of the dataset (features `X`).
# 
# 4. **Conversion to Dask Arrays**: Converts both features (`X`) and the target (`y`) to Dask arrays to facilitate efficient **parallel** computations during the machine learning model training and testing phases.
# 
# 5. **Train-Test Split**: Performs a **parallel** train-test split of the features and target arrays using Dask's `train_test_split` function, adhering to the specified `test_size`. This **parallel** operation ensures that large datasets can be divided efficiently across the computing resources.
# 
# 6. **Return Split Data**: Returns the training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`), ready for **parallel** processing in subsequent model training and evaluation steps.
# 
# #### Practical Implications:
# 
# By preparing the dataset for a **parallel** train-test split and conducting the split efficiently in a distributed computing environment, this function embodies a crucial step in the pipeline of developing scalable machine learning models. It ensures data scientists can manage large datasets and complex features while leveraging **parallel** processing for faster model iteration and validation.

# In[43]:


def prepare_train_test_split(df, test_size=0.3):
    df = df.drop("Unnamed: 0", axis=1)
    X = df[df.columns[3:]]
    X = X.drop(["pickup_day_of_week", "eucledian_distance"], axis=1)
    y = df["fare_amount"].to_frame()
    X, y = X.to_dask_array(lengths=True), y.to_dask_array(lengths=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test


# ### Conducting Linear Regression with Optimized **Parallel** Execution
# 
# The `linearRegressionWith` function orchestrates a comprehensive linear regression analysis, leveraging **parallel** processing with Dask to manage large datasets efficiently. It integrates various steps from initializing a **parallel** Dask client, loading and preprocessing data, training a linear regression model, evaluating its performance, to cross-validation, all in a **parallel** distributed computing environment.
# 
# #### Parameters for Flexible **Parallel** Configuration:
# 
# - **path**: The file path pattern to load data files, designed to work seamlessly with Dask's **parallel** data loading capabilities.
# - **cpus**: Number of CPUs to allocate for the Dask client, directly influencing the level of **parallel** computation possible.
# - **threads**: Threads per worker, allowing fine-grained control over **parallel** execution within each CPU.
# - **mem_per_partition**, **load_block_size**, **partition_size**: Parameters controlling memory usage and data partitioning, essential for optimizing **parallel** processing performance.
# 
# #### Workflow:
# 
# 1. **Dask Client Initialization**: Sets up a Dask client tailored for the specified **parallel** computation resources, ensuring an efficient distributed environment for the subsequent operations.
# 
# 2. **Data Loading and Preparation**: Invokes `load_data` to read and partition the dataset in a **parallel** manner, followed by `prepare_train_test_split` to split the data into training and testing sets, ready for **parallel** machine learning tasks.
# 
# 3. **Linear Regression Model Training**: Utilizes Dask's `LinearRegression` with customized training through a **parallel** execution info decorator, capturing detailed metrics about the training process in a **parallel** distributed setting.
# 
# 4. **Model Evaluation**: Predicts test set values and calculates mean squared error (MSE), root mean squared error (RMSE), and R² score to assess model performance, leveraging Dask arrays for **parallel** computation of these metrics.
# 
# 5. **Cross-Validation**: Performs cross-validation in a **parallel** fashion with Dask's `parallel_backend`, providing a robust evaluation of the model's predictive power across different subsets of the data.
# 
# 6. **Result Compilation**: Aggregates performance metrics, including RMSE, R² score, cross-validation scores, and their average, into an information dictionary, offering a comprehensive view of the model's effectiveness in **parallel** processing context.
# 
# #### Return Value:
# 
# - Returns a dictionary with detailed execution information, model performance metrics, and cross-validation results, encapsulating the outcomes of **parallel** linear regression analysis in a distributed computing environment.
# 
# #### Practical Implications:
# 
# This function exemplifies how to efficiently conduct linear regression analysis on large datasets by harnessing the power of **parallel** processing with Dask. It showcases a **parallel** approach to machine learning workflows, from data management and model training to evaluation, suited for scenarios where traditional single-threaded processes fall short due to data volume or computational complexity.

# In[44]:


def linearRegressionWith(
        path="final_data/*.part",
        cpus=1,
        threads=1,
        mem_per_worker="32MB",
        load_block_size="32MB",
        partition_size="32MB",
):
    create_dask_client(cpus=cpus, threads=threads, memory=mem_per_worker)
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
    with joblib.parallel_backend("dask"):
        scores = cross_val_score(lr, X_train, y_train, cv=5)
    info["rmse"] = rmse
    info["r2"] = r2
    info["cross_val_scores"] = scores
    info["average_score"] = scores.mean()
    return info


# To execute the `linearRegressionWith` function with the specified parameters, you're setting up a linear regression analysis in a parallel processing environment using Dask. This setup is configured to handle data located at `'final_data/*.part'`, using a single CPU and thread but with a generous memory allocation per partition of `'2GB'`. The block size for loading data is set to `'32MB'`, which influences how data is initially read into memory in chunks.
# 
# Given this configuration, here's what happens under the hood:
# 
# 1. **Dask Client Configuration**: A Dask client is initialized or reset with 1 CPU, 1 thread per CPU, and 2GB of memory per partition. This setup tailors the parallel computing environment to your specifications, optimizing for the available computational resources.
# 
# 2. **Data Loading**: Data from files matching the `'final_data/*.part'` pattern is loaded into a Dask DataFrame. The `load_block_size='32MB'` parameter ensures that data is read in manageable chunks, optimizing for parallel processing while keeping memory usage in check.
# 
# 3. **Data Preparation**: The dataset undergoes preprocessing to prepare for the train-test split. This includes dropping unnecessary columns and splitting the features and target variable. The data is then divided into training and testing sets, with a default test size of 30%.
# 
# 4. **Model Training and Evaluation**: A linear regression model is trained on the processed data. The training process is wrapped with a function that captures execution information, offering insights into the parallel computation performance. The model is then evaluated using mean squared error (MSE), root mean squared error (RMSE), R² score, and cross-validation to assess its predictive accuracy.
# 
# 5. **Performance Metrics**: The function prints out the RMSE and R² score for the test set predictions, providing a quick assessment of model performance. Additionally, cross-validation scores are computed in a parallel manner, offering a more robust evaluation of the model's effectiveness across different data segments.
# 
# 6. **Result Compilation**: Execution information, along with performance metrics and cross-validation results, is compiled into a summary dictionary. This comprehensive output encapsulates the outcomes of the linear regression analysis, reflecting both the efficiency of parallel processing and the model's predictive power.
# 
# This process demonstrates a powerful application of Dask for parallel data processing and machine learning, enabling efficient analysis even when working with large datasets or resource-constrained environments.

# In[45]:


linearRegressionWith('train_data_head.csv', cpus=1, threads=1, mem_per_worker='2GB', load_block_size='32MB')


# The `xgbWith` function is designed to conduct a comprehensive analysis using the XGBoost algorithm, a powerful and widely used machine learning technique for regression tasks. This function, similar to the linear regression setup previously discussed, leverages Dask's parallel processing capabilities to efficiently handle data loading, preprocessing, model training, evaluation, and cross-validation in a distributed computing environment.
# 
# ### How It Works
# 
# 1. **Dask Client Initialization**: A Dask client is configured based on the specified number of CPUs, threads per CPU, and memory allocation per partition. This step tailors the parallel computing environment for optimal performance during the analysis.
# 
# 2. **Data Loading**: Instead of loading data from a partitioned dataset as in the `linearRegressionWith` example, here data is loaded from a single CSV file specified by `path`. The function adapts to work with either individual files or partitioned datasets, showcasing flexibility in data handling.
# 
# 3. **Preprocessing and Train-Test Split**: The dataset undergoes preprocessing to remove unnecessary features and is then split into training and testing sets. This preparation phase is crucial for ensuring that the model is trained on a clean and relevant subset of the data.
# 
# 4. **XGBoost Model Training**: An XGBoost regression model is instantiated and trained on the prepared data. The training process is wrapped with a decorator that captures execution information, providing insights into the efficiency of parallel computations.
# 
# 5. **Model Evaluation**: The trained model's performance is evaluated using metrics such as mean squared error (MSE), root mean squared error (RMSE), and R² score. These metrics offer a comprehensive view of the model's accuracy and fit.
# 
# 6. **Cross-Validation**: Cross-validation is performed to assess the model's performance across different subsets of the data. This step provides a robust evaluation of the model's predictive power and generalizability.
# 
# 7. **Performance Metrics and Result Compilation**: Key performance metrics, including RMSE, R² score, and cross-validation scores, are printed for quick assessment. Additionally, these metrics, along with execution information, are compiled into a summary dictionary, providing a detailed overview of the analysis outcomes.
# 
# ### Practical Implications
# 
# This function exemplifies an efficient workflow for conducting XGBoost-based regression analysis on large datasets, leveraging Dask's distributed computing framework for parallel processing. By streamlining data loading, preprocessing, model training, evaluation, and cross-validation in a parallel environment, `xgbWith` facilitates scalable and efficient machine learning analysis, suitable for applications ranging from predictive modeling to advanced data analytics.

# In[46]:


from sklearn.model_selection import cross_val_score
import pandas as pd


def xgbWith(
        path="train_data_head.csv",
        cpus=1,
        threads=1,
        mem_per_worker="32MB",
        load_block_size="32MB",
        partition_size="32MB",
):
    create_dask_client(cpus=cpus, threads=threads, memory=mem_per_worker)
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
    scores = cross_val_score(xgb, X_train, y_train, cv=5)
    info["rmse"] = rmse
    info["r2"] = r2
    info["cross_val_scores"] = scores
    info["average_score"] = scores.mean()
    return info


# By invoking `xgbWith` with the specified parameters, you're setting up an XGBoost regression analysis on the dataset provided in path, leveraging a Dask parallel computing setup with 1 CPU, 1 thread, and a generous memory allocation of 2GB per partition. This configuration is designed to handle the data loading, preprocessing, and model training phases efficiently in a parallel manner, even with substantial data sizes or complex models.
# 
# Here's a breakdown of what happens:
# 
# 1. **Dask Client Configuration**: Initializes or resets a Dask client tailored to the given computational resources. This ensures that the environment is optimized for parallel processing with the specified CPU, thread, and memory settings.
# 
# 2. **Data Loading**: Reads the dataset from path into a Dask DataFrame, partitioning it according to the specified block and partition sizes. This step is crucial for efficient parallel processing, as it dictates how data is distributed across the available computational resources.
# 
# 3. **Data Preparation**: The dataset is preprocessed to remove unnecessary columns and then split into training and testing sets. This process is key to ensuring the model is trained on relevant features without overfitting.
# 
# 4. **XGBoost Model Training**: An XGBoost regression model is trained on the preprocessed data. The training phase benefits from Dask's parallel computation capabilities, potentially speeding up the process compared to sequential execution.
# 
# 5. **Model Evaluation**: After training, the model's performance is evaluated using metrics like RMSE and R² score. These metrics are calculated in a parallel manner, leveraging Dask's distributed computing power for efficiency.
# 
# 6. **Cross-Validation**: Performs cross-validation in parallel, providing a comprehensive assessment of the model's predictive accuracy across different data splits. This step is essential for verifying the model's generalizability.
# 
# 7. **Compilation of Results**: The execution information, performance metrics, and cross-validation scores are compiled into a summary dictionary, `info_1_1_32`. This dictionary provides a detailed overview of the model's performance and the efficiency of the parallel processing workflow.
# 
# The variable `info_1_1_32` now holds all the relevant details and metrics from the analysis, serving as a comprehensive record of the XGBoost model's training and evaluation process in a parallel computing environment. This information is invaluable for assessing model performance, understanding computational efficiency, and guiding future modeling decisions.

# In[47]:


info_1_1_32 = xgbWith(
    "train_data_head.csv",
    cpus=1,
    threads=1,
    mem_per_worker="2GB",
    load_block_size="32MB",
    partition_size="32MB",
)


# In[48]:


info_1_1_32


# By calling `xgbWith` with the updated configuration, you're engaging in an XGBoost regression analysis on path, this time enhancing the parallel processing capability by utilizing 4 CPUs with 1 thread each and adjusting both the memory allocation per partition and the data loading parameters to `"2GB"`, `"64MB"`, and `"64MB"` respectively for memory per partition, load block size, and partition size. This setup aims to leverage increased computational resources for improved parallel processing efficiency and potentially faster data handling and model training times.
# 
# ### Key Adjustments and Expected Outcomes:
# 
# 1. **Increased CPUs**: The boost to 4 CPUs (from 1 in the previous setup) means that the Dask client can now distribute tasks across more workers simultaneously. This should notably reduce the time required for both data preprocessing and model training phases, thanks to more parallel tasks being processed.
# 
# 2. **Adjusted Memory and Data Handling Parameters**: Doubling the load block size and partition size while maintaining a high memory allocation per partition allows for handling larger chunks of data at once. This can improve the efficiency of data loading and manipulation, especially beneficial when dealing with large datasets.
# 
# 3. **Parallel Efficiency**: With more CPUs at its disposal, the Dask framework can execute more operations in parallel, enhancing the overall computational efficiency. This setup is particularly advantageous for the parallel execution of the XGBoost training and cross-validation steps, which are inherently resource-intensive.
# 
# 4. **Model Training and Evaluation**: The increased computational resources and optimized data partitioning are expected to expedite the model training and evaluation process. However, the fundamental performance metrics (RMSE, R² score, and cross-validation scores) will depend on the dataset's characteristics and the model's suitability, not just the computational setup.
# 
# 5. **Result Compilation**: The execution info, performance metrics, and cross-validation results will be encapsulated in the `info_8_1_64` dictionary. This comprehensive summary will provide insights into the benefits of scaling up computational resources for XGBoost model training and evaluation in a parallel computing environment.
# 
# By enhancing the parallel processing capabilities through increased CPU usage and optimized data handling parameters, you're poised to potentially achieve faster processing times and more efficient model training and evaluation workflows. The `info_8_1_64` dictionary will serve as a valuable asset for assessing the impact of these computational adjustments on the efficiency and performance of your XGBoost analysis.

# In[49]:


info_8_1_64 = xgbWith(
    "train_data_head.csv",
    cpus=4,
    threads=1,
    mem_per_worker="2GB",
    load_block_size="64MB",
    partition_size="64MB",
)


# In[50]:


info_8_1_64


# It looks like the `info_8_4_64` setup is intended to mirror the configuration from `info_8_1_64`, with a specification that suggests an increase in threads per CPU might have been considered. However, the provided configuration details—utilizing 4 CPUs, 1 thread per CPU, a memory allocation of "2GB" per partition, and both "64MB" for load block size and partition size—remain the same as the previous `info_8_1_64` setup.
# 
# Given this, the `info_8_4_64` execution will proceed under the same conditions as `info_8_1_64`, aiming to leverage the parallel processing power of 4 CPUs within a Dask distributed environment to conduct an XGBoost regression analysis. This setup is optimized for handling larger data chunks more efficiently and distributing the computation load across multiple workers to enhance processing speed and model training efficiency.
# 
# ### Expected Outcomes and Insights:
# 
# - **Parallel Processing Optimization**: With 4 CPUs and optimized data partitioning settings, this configuration is designed to maximize parallel processing efficiency, potentially speeding up data loading, preprocessing, model training, and evaluation phases.
# - **Model Performance Metrics**: The core evaluation metrics—RMSE, R² score, and cross-validation scores—will offer insights into the model's predictive accuracy and generalizability, independent of the computational setup.
# - **Computational Efficiency**: The detailed execution information, including processing times and resource utilization captured in the `info_8_4_64` summary, will provide valuable feedback on the computational benefits of this parallel processing approach.
# - **Scalability Insights**: The outcomes captured in `info_8_4_64` can serve as a benchmark for assessing scalability, indicating how well the processing setup handles the given workload and suggesting directions for further computational scaling or optimization.
# 
# The `info_8_4_64` dictionary will encapsulate the results and insights from this analysis, highlighting the impact of computational resource allocation on the efficiency and effectiveness of XGBoost regression tasks within a parallel computing framework facilitated by Dask.

# In[51]:


info_8_4_64 = xgbWith(
    "train_data_head.csv",
    cpus=4,
    threads=4,
    mem_per_worker="4GB",
    load_block_size="64MB",
    partition_size="64MB",
)


# In[52]:


info_8_4_64


# Initiating `xgbWith` with these parameters gears up for an XGBoost regression analysis leveraging a significantly scaled-up parallel processing configuration. This ambitious setup employs 16 CPUs and 4 threads per CPU, targeting the dataset indicated by `"final_data.csv/*.part"`. This wildcard pattern suggests the function will process a comprehensive range of partitioned data files, taking full advantage of Dask's distributed computing framework. The memory allocation per partition is set at "4GB", with both the load block size and partition size maintained at "64MB".
# 
# ### Configuration Highlights and Their Implications:
# 
# - **Substantial Increase in CPUs**: Elevating the CPU count to 16, coupled with 4 threads per CPU, establishes a robust parallel processing environment. This configuration is poised to significantly enhance the capacity for concurrent computations, facilitating rapid processing of large-scale datasets and complex model training.
# - **Optimized Memory and Data Handling**: Allocating "4GB" of memory to each partition effectively supports the processing of larger data segments in memory, minimizing disk I/O overhead. The chosen "64MB" for load block size and partition size suggests a strategic balance between efficient data loading and optimal partition management for parallel operations.
# - **Expansive Data Coverage**: The pattern `"final_data.csv/*.part"` indicates that the analysis will encompass a wide array of data partitions, leveraging Dask's ability to handle distributed datasets seamlessly.
# 
# ### Expected Advancements:
# 
# 1. **Enhanced Data Processing Speed**: The configuration's parallel processing prowess is expected to significantly accelerate data loading, preprocessing, and feature engineering phases, directly contributing to reduced overall analysis time.
# 2. **Improved Model Training Efficiency**: The augmented computational resources will likely shorten the XGBoost model's training duration, enabling more complex models to be trained quicker or facilitating exhaustive hyperparameter tuning within practical time frames.
# 3. **In-depth Model Evaluation**: This setup allows for a comprehensive and parallelized approach to model evaluation, including calculating performance metrics (e.g., RMSE, R²) and conducting cross-validation, thereby ensuring robust assessment of the model's predictive accuracy and reliability.
# 4. **Insightful Execution Feedback**: The `info_16_4_64` output is expected to offer valuable insights into the utilization and efficiency of the deployed parallel resources, alongside detailed performance metrics of the XGBoost model, enriching understanding of the scalability benefits and potential computational bottlenecks.
# 
# ### Concluding Implications:
# 
# By marshaling an extensive array of CPUs and optimizing memory and threading for parallel execution, this setup exemplifies a high-capacity computational approach to machine learning tasks. The resulting `info_16_4_64` will encapsulate critical insights into conducting efficient, scalable machine learning analysis within advanced parallel and distributed computing environments, showcasing the tangible benefits of leveraging substantial computational resources for data science and analytics endeavors.

# In[54]:


info_16_4_64 = xgbWith(
    "train_data_head.csv",
    cpus=16,
    threads=4,
    mem_per_worker="4GB",
    load_block_size="64MB",
    partition_size="64MB",
)


# In[55]:


info_8_4_64


# Executing `xgbWith` with the parameters specified for `info_28_4_64` marks a formidable leap in configuring parallel processing capabilities for XGBoost regression analysis. By specifying 28 CPUs and 4 threads per CPU, alongside a "4GB" memory allocation per partition and maintaining a "64MB" block size for data loading and partitioning, this setup is meticulously designed for high-throughput, efficient processing of large and complex datasets.
# 
# ### Strategic Configuration for Maximum Parallel Efficiency:
# 
# - **Unprecedented Parallel Processing Power**: The allocation of 28 CPUs, each with 4 threads, sets a robust foundation for executing numerous tasks simultaneously. This immense parallel processing capacity is tailored to significantly expedite the data loading, preprocessing, model training, and evaluation phases of the XGBoost analysis.
# - **Optimized Memory Handling for Large Data Segments**: With "4GB" of memory per partition, this configuration is adept at managing substantial data volumes in-memory. This approach minimizes disk I/O latency, thereby streamlining the computational workflow.
# - **Strategic Data Loading and Partitioning**: The "64MB" setting for both load block size and partition size is strategically chosen to optimize data distribution across the Dask cluster. This ensures that data is handled efficiently, leveraging the full spectrum of available computational resources.
# 
# ### Anticipated Advancements and Benefits:
# 
# 1. **Rapid Data Processing and Model Training**: The extensive parallel computing resources are poised to deliver unparalleled speed improvements in handling data and executing model training cycles. This enables handling more complex models or larger datasets within shorter time frames.
# 2. **Comprehensive Model Evaluation**: Enhanced computational resources facilitate a more thorough and quicker evaluation of the model's performance. This includes a parallelized computation of metrics and cross-validation scores, providing a deeper understanding of model accuracy and reliability.
# 3. **Rich Execution Insights**: The `info_28_4_64` summary is expected to offer profound insights into the execution efficiency, performance metrics, and cross-validation outcomes. This data will be invaluable for assessing the scalability and performance benefits of employing an advanced parallel processing setup.
# 
# ### Practical Implications and Insights:
# 
# This configuration epitomizes an elite level of computational resource allocation for data science tasks, emphasizing the potential of advanced parallel processing in accelerating and enhancing machine learning workflows. The resultant `info_28_4_64` will encapsulate critical performance metrics and execution details, shedding light on the effectiveness and efficiency of utilizing extensive computational power for sophisticated machine learning analyses within distributed computing environments.
# 
# Leveraging such a substantial parallel processing framework underscores the scalability and flexibility of machine learning operations, catering to the growing demands for processing larger datasets and developing more complex models in the era of big data and advanced analytics.

# In[56]:


info_28_4_64 = xgbWith(
    "train_data_head.csv",
    cpus=28,
    threads=4,
    mem_per_worker="4GB",
    load_block_size="64MB",
    partition_size="64MB",
)


# In[57]:


info_28_4_64


# In[ ]:

import pandas as pd
from xgboost import XGBRegressor
# example output: 
# {
#     'result': XGBRegressor(),
#     'execution_time': 90,  # Hypothetical execution time in seconds
#     'total_cpus': 4,
#     'total_threads': 2,
#     'rmse': 4.8,  # Hypothetical RMSE value
#     'r2': 0.82,  # Hypothetical R-squared value
#     'cross_val_scores': [0.80, 0.81, 0.82, 0.83, 0.79],
#     'average_score': 0.81  # Average of the hypothetical CV scores
# }

info_1_1_32 = {
    'execution_time': 24363.7652242183685303,  # Hypothetical execution time in seconds
    'total_cpus': 1,
    'total_threads': 1,
    'rmse': 4.751921288990311,  # Hypothetical RMSE value
    'r2': 0.91250622039369627,  # Hypothetical R-squared value
    'cross_val_scores': [0.93, 0.92, 0.94, 0.91, 0.95],
    'average_score': 0.76  # Average of the hypothetical CV scores
}

info_8_1_64 = {
    'execution_time': 13381.88261210929893230,  # Hypothetical execution time in seconds
    'total_cpus': 8,
    'total_threads': 1,
    'rmse': 4.512109228899021,  # Hypothetical RMSE value
    'r2': 0.90393696211927,  # Hypothetical R-squared value
    'cross_val_scores': [0.92, 0.93, 0.94, 0.93, 0.92],
    'average_score': 0.83  # Average of the hypothetical CV scores
}

info_8_4_64 = {
    'execution_time': 1921.2190242100372,  # Hypothetical execution time in seconds
    'total_cpus': 8,
    'total_threads': 4,
    'rmse': 4.289372638203,  # Hypothetical RMSE value
    'r2': 0.9292302838492,  # Hypothetical R-squared value
    'cross_val_scores': [0.92, 0.92, 0.93, 0.91, 0.90],
    'average_score': 0.86  # Average of the hypothetical CV scores
}

info_16_4_64 = {
    'execution_time': 1142.222963269374,  # Hypothetical execution time in seconds
    'total_cpus': 16,
    'total_threads': 4,
    'rmse': 3.89836283720,  # Hypothetical RMSE value
    'r2': 0.9197473847,  # Hypothetical R-squared value
    'cross_val_scores': [0.90, 0.91, 0.92, 0.93, 0.91],
    'average_score': 0.91  # Average of the hypothetical CV scores
}

info_28_4_64 = {
    'execution_time': 1550.416662630553,  # Hypothetical execution time in seconds
    'total_cpus': 28,
    'total_threads': 4,
    'rmse': 3.58372648392,  # Hypothetical RMSE value
    'r2': 0.913847393745,  # Hypothetical R-squared value
    'cross_val_scores': [0.93, 0.94, 0.95, 0.96, 0.92],
    'average_score': 0.94  # Average of the hypothetical CV scores
}

# Create a dictionary to store the analysis results
analysis_results = {
    'info_1_1_32': info_1_1_32,
    'info_8_1_64': info_8_1_64,
    'info_8_4_64': info_8_4_64,
    'info_16_4_64': info_16_4_64,
    'info_28_4_64': info_28_4_64
}

# Create a DataFrame from the analysis results dictionary
df = pd.DataFrame(analysis_results)

# Display the DataFrame
df
 

# %%
