# useful_functions.py

"""
Module containing useful functions for the project.
#### References:
- Eichner, A. (2023, December 1). Identifying and Handling Outliers in Pandas: A Step-By-Step Guide. Medium. https://python.plainenglish.io/identifying-and-handling-outliers-in-pandas-a-step-by-step-guide-fcecd5c6cd3b
Contribution: The function remove_outliers was adapted from the code provided in the article.

#### Libraries
- Package Pandas (2.2). (2024). [Python]. https://pandas.pydata.org/
- Package NumPy (1.23). (2023). [Pyhton]. https://numpy.org/ - Harris, C. R., Millman, K. J., Van Der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., Van Kerkwijk, M. H., Brett, M., Haldane, A., Del Río, J. F., Wiebe, M., Peterson, P., … Oliphant, T. E. (2020). Array programming with NumPy. Nature, 585(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2
- Droettboom, J. D. H., Michael. (2024). Package matplotlib (3.8.4) [Python]. https://matplotlib.org

"""
### Useful functions for the project ###


# Import libraries

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt


# Function to get the last value in the dataset for the month
def convert_df_to_monthly (dataframe, index_format):
    """
    Gets the last value for each month in the given dataframe.
    
    Args:
        dataframe (pd.DataFrame): The dataframe to be modified.
        index_format: The original dataframe index format 
    
    Returns:
        pd.DataFrame: The modified dataframe with the last value for the month.
    """
    # Convert the index to datetime with desired format
    dataframe.index = pd.to_datetime(dataframe.index, format = index_format)
    # Create a new dataframe with the last value for each month
    dataframe_monthly = dataframe.resample('ME').last()
    # Adjust the index format
    dataframe_monthly.index = dataframe_monthly.index.strftime('%Y-%m')

    return dataframe_monthly # Return the new dataframe


# Returns the last working day based on actual date
def define_end_period(date_format): 
    """
    Define the last period to get data based on a format

    Args:
        format: The desired format for the date

    Returns:
        str: The last period to get data
    """
    # Get today's date
    today = datetime.now()

    # Calculate the year and month before today's date
    end_period = (today - timedelta(days=30)).strftime(date_format)

    return end_period


# Function to plot the prediction vs test data
def plot_prediction_vs_test(target_variable, test, prediction, title):
    """
    Plots the prediction and test data.
    
    Args:
        train (pd.DataFrame): The training data.
        test (pd.DataFrame): The test data.
        prediction (pd.DataFrame): The prediction data.
        title (str): The title of the plot.
    """
    # Convert the series to dataframes
    test_df = pd.DataFrame({'date': pd.to_datetime(test.index), target_variable: test.values}).set_index('date').to_period('M')

    # Convert index to datetime objects
    test_df.index = test_df.index.to_timestamp()

    # Plot the forecast
    plt.figure(figsize=(10, 6))
    plt.plot(test_df[target_variable], label='Test') # Plot test data
    plt.plot(test_df.index, prediction, label='Prediction') # Plot predictions
    plt.legend()
    plt.title(title) # Include a title
    plt.ylabel('Brazilian primary balance - R$ million') # Include y-axis label
    plt.xlabel('Date') # Include x-axis label
    plt.show()


# Function to remove outliers
def remove_outliers(dataframe, threshold=0.20):
    """
    Replaces outliers in the dataframe with NAs based on the given threshold. IQR method is used.
    Args:
        dataframe (pd.DataFrame): The dataframe to be modified.
        threshold (float): The quantile threshold to identify outliers.

    Returns:
        pd.DataFrame: The dataframe after outliers removal.
    """
    # Iterate over each column in the dataframe
    for column in dataframe.columns:
        # Compute the first and third quartiles
        Q1 = dataframe[column].quantile(threshold)
        Q3 = dataframe[column].quantile(1 - threshold)
        IQR = Q3 - Q1
        
        # compute the lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Check if the values are outliers
        is_outlier = (dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)
        # Replace outliers with NaN 
        dataframe.loc[is_outlier, column] = np.nan
        
    return dataframe


# Function to fill missing values
def fill_missing_values(dataframe):
    """
    Fills missing values in the given dataframe.
    Steps:
        - Each column is converted to float.
        - NaNs are filled with the mean of the previous 6 observations. 
        - If not enough past data, uses the mean of available past observations.
        - If no past data is available, uses the mean of the next 6 observations,
        prioritizing filling from the end of the DataFrame to the start.
    
    Args:
        dataframe (pd.DataFrame): The dataframe to be modified.
    
    Returns:
        pd.DataFrame: The dataframe with missing values filled.
    """
    # Convert every column to float
    dataframe = dataframe.astype('float32')
    
    # Iterate over each column in the dataframe
    for column in dataframe.columns:
        # Forward fill: Fill NaNs with the mean of the previous 6 observations
        for index in range(len(dataframe)):
            if pd.isna(dataframe.loc[dataframe.index[index], column]):
                # Compute mean of past 6 or available observations
                past_mean = dataframe[column].iloc[max(0, index-6):index].mean()
                # Access the dataframe specific cell by index and column to fill the NaN
                dataframe.at[dataframe.index[index], column] = past_mean if pd.notnull(past_mean) else np.nan
                
        # Backward fill: For remaining NaNs, fill with the mean of the next 6 observations
        for index in range(len(dataframe)-1, -1, -1):  # Iterate in reverse order
            if pd.isna(dataframe.loc[dataframe.index[index], column]):
                # Compute mean of the next 6 or available observations
                next_mean = dataframe[column].iloc[index+1:index+7].mean()
                # Access the dataframe specific cell by index and column to fill the NaN
                dataframe.at[dataframe.index[index], column] = next_mean if pd.notnull(next_mean) else np.nan
    
    return dataframe