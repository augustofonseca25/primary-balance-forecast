# useful_functions.py

"""
Module containing useful functions for the project.
"""
### Useful functions for the project ###

# Import libraries

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
#from statsmodels.tsa.stattools import adfuller, acf
#from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
#import statsmodels.api as sm



# # Script to adjust the date format
# def adjust_index_date_format(dataframe):
#     """
#     Adjusts the date format of the index in the given dataframe.
    
#     Args:
#         dataframe (pd.DataFrame): The dataframe to be modified.
    
#     Returns:
#         pd.DataFrame: The modified dataframe with the adjusted date format.
#     """

#     # Convert the index to datetime with desired format
#     dataframe.index = pd.to_datetime(dataframe.index)

#     # Sort the index in ascending order
#     dataframe = dataframe.sort_index()

#     # Convert the index to the desired format
#     dataframe.index = dataframe.index.strftime('%d-%m-%Y')

#     return dataframe


# Script to get the last value in the dataset for the month
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


# def clean_dataset(dataframe):
#     """
#     Cleans the given dataframe by removing any infinite or missing values.
    
#     Args:
#         dataframe (pd.DataFrame): The dataframe to be cleaned.
    
#     Returns:
#         pd.DataFrame: The cleaned dataframe.
#     """

#     # Remove any missing values
#     clean_data = dataframe.replace([np.inf, -np.inf], np.nan).dropna()

#     return clean_data


# def test_stationarity_and_differenciate(df):
#     """
#     Tests for stationarity and difference if needed for the given dataframe.

#     Parameters:
#     - df: DataFrame to be tested and differenced if needed.

#     Returns:
#     - DataFrame after differencing.
#     - Dictionary indicating which variables were differenced and their initial values.
#     """
#     diff_dict = {}
#     for col in df.columns:
#         initial_value = df[col].iloc[0]
#         initial_index = df[col].index[0]
#         result = adfuller(df[col], autolag='AIC')
#         diff_seasonal = False
#         diff_1st = False
#         if result[1] > 0.05: # p-value > 0.05
#             df[col] = df[col].diff().dropna()
#             diff_1st = True
#         # Check for seasonality with autocorrelation at lag 12
#         acf_values = acf(df[col], nlags=12, fft=True)
#         if abs(acf_values[12]) > 0.75: # arbitrary threshold for significant seasonality
#             df[col] = df[col].diff(12).dropna()
#             diff_seasonal = True
#         diff_dict[col] = {'differenced': diff_1st, 'initial_value': initial_value, 'initial_index': initial_index, 'seasonal_differenced': diff_seasonal}
    
#     if diff_dict is not None: # If any variable was differenced
#         df = df.iloc[1:] # Remove the first row of the df which contains some NAs
#     return df, diff_dict


# def reverse_differencing(df, diff_dict):
#     """
#     Reverses the differencing of the variables in the dataframe based on the provided dictionary.

#     Parameters:
#     - df: DataFrame to be reversed.
#     - diff_dict: Dictionary indicating which variables were differenced and their initial values.

#     Returns:
#     - DataFrame after reversing differencing.
#     """

#     for col, info in diff_dict.items():
#         # Check if the col exists in the dataframe
#         if col not in df.columns:
#             continue
#         else:
#             if info['seasonal_differenced']:
#                 # revert seasonal differencing
#                 df[col] = df[col].shift(-12).fillna(0).cumsum() + info['initial_value']
#             if info['differenced']:
#                 # Revert first order differencing
#                 df[col] = df[col].cumsum()
#     return df

# def create_synthetic_data(dataframe, lags):
#     """
#     Creates a dataframe with lags for the given dataframe.
    
#     Args:
#         dataframe (pd.DataFrame): The dataframe to be modified.
#         lags (list): The list of lags to be created.
    
#     Returns:
#         pd.DataFrame: The dataframe with lags.
#     """
#     # Get the original features' names
#     features = dataframe.columns

#     # Initialize a list to hold the new feature DataFrames
#     new_features = []

#     # Create lag and rolling window features for all variables
#     for feature in features:
#         for lag in lags:
#             new_features.append(
#                 dataframe[feature].shift(lag).rename(
#                     f'{feature}_synthet_lag_{lag}'))
#             new_features.append(
#                 dataframe[feature].rolling(window=12).mean().rename(
#                     f'{feature}_synthet_roll_mean_12'))
#             new_features.append(
#                 dataframe[feature].rolling(window=12).std().rename(
#                     f'{feature}_synthet_roll_std_12'))

#     # Convert the list of new features to a DataFrame
#     sintetic_data = pd.concat(new_features, axis=1)

#     return sintetic_data


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


# def plot_auto_correlation(model):
#     """
#     Plots auto-correlation.
    
#     Args:
#         model to get the residuals
#     """
#     # Plot ACF and PACF for the residuals
#     fig, ax = plt.subplots(1, 2, figsize=(12, 5))
#     sm.graphics.tsa.plot_acf(model, lags=40, ax=ax[0])
#     sm.graphics.tsa.plot_pacf(model, lags=40, ax=ax[1])

#     plt.tight_layout()
#     plt.show()

# Function to remove outliers
def remove_outliers(dataframe, threshold=0.20):
    """
    Replaces outliers in the dataframe with NAs based on the given threshold.
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