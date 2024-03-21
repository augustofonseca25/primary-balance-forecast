# useful_functions.py

"""
Module containing useful functions for the project.
"""
### Useful functions for the project ###

# Import libraries

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm



# Script to adjust the date format
def adjust_index_date_format(dataframe):
    """
    Adjusts the date format of the index in the given dataframe.
    
    Args:
        dataframe (pd.DataFrame): The dataframe to be modified.
    
    Returns:
        pd.DataFrame: The modified dataframe with the adjusted date format.
    """

    # Convert the index to datetime with desired format
    dataframe.index = pd.to_datetime(dataframe.index)

    # Sort the index in ascending order
    dataframe = dataframe.sort_index()

    # Convert the index to the desired format
    dataframe.index = dataframe.index.strftime('%d-%m-%Y')

    return dataframe


# Script to get the best value in the dataset for the month
def convert_df_to_monthly (dataframe, index_format):
    """
    Gets the last value for each month in the given dataframe.
    
    Args:
        dataframe (pd.DataFrame): The dataframe to be modified.
        index_format: The original dataframe index format 
    
    Returns:
        pd.DataFrame: The modified dataframe with the best value for the month.
    """

    dataframe.index = pd.to_datetime(dataframe.index, format = index_format)
    
    dataframe_monthly = dataframe.resample('M').last()

    dataframe_monthly.index = dataframe_monthly.index.strftime('%Y-%m')

    return dataframe_monthly


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


def clean_dataset(dataframe):
    """
    Cleans the given dataframe by removing any infinite or missing values.
    
    Args:
        dataframe (pd.DataFrame): The dataframe to be cleaned.
    
    Returns:
        pd.DataFrame: The cleaned dataframe.
    """

    # Remove any missing values
    clean_data = dataframe.replace([np.inf, -np.inf], np.nan).dropna()

    return clean_data


# def standardize_data(dataframe, type_choice='standardize', reverse=False, original_params=None):
#     """
#     Standardizes/normalizes the given dataframe and reverses the transformation if specified.
    
#     Args:
#         dataframe: The dataframe to be standardized/normalized.
#         type_choice (str): The type of standardization to be performed ('standardize' or 'normalize').
#         reverse (bool): If True, reverses the transformation.
#         original_params: The parameters used for standardization/normalization.
        
#     Returns:
#         DataFrame: The standardized/normalized dataframe.
#     """
#     numerical_cols = dataframe.select_dtypes(include=['float64', 'int64']).columns # Select numerical columns
    

#     # Check if it's to reverse the standardization/normalization
#     if not reverse:
#         if type_choice == 'standardize': # Check the type of standardization
#             scaler = StandardScaler()
#             scaled_data = scaler.fit_transform(dataframe[numerical_cols])
#             params = {'mean': scaler.mean_, 'scale': scaler.scale_} # to be used for reverse transformation
#         elif type_choice == 'normalize':
#             scaler = MinMaxScaler()
#             scaled_data = scaler.fit_transform(dataframe[numerical_cols])
#             params = {'min': scaler.data_min_, 'max': scaler.data_max_} # to be used for reverse transformation
#         dataframe[numerical_cols] = scaled_data # Update the numerical columns in the dataframe
#         return dataframe, params # returns the dataframe and the parameters used for standardization/normalization
#     else: # Reverse the transformation
#         params = original_params # Retrieve the original parameters incase of reverse transformation
#         if type_choice == 'standardize': # Check the type of the used standardization
#             mean = params['mean'] # Get the mean and scale values
#             scale = params['scale']
#             dataframe[numerical_cols] = dataframe[numerical_cols] * scale + mean # Reverse the transformation
#         elif type_choice == 'normalize': # Check the type of the used normalization
#             min_val = params['min'] # Get the min and max values
#             max_val = params['max']
#             dataframe[numerical_cols] = (dataframe[numerical_cols] * (max_val - min_val) + min_val) # Reverse the transformation
#         return dataframe


# def scale_data(train_dataframe, test_dataframe, type_choice='standardize'):
#     """
#     Scales the given training and test dataframes using standardization.
    
#     Args:
#         train_dataframe: DataFrame containing the training data.
#         test_dataframe: DataFrame containing the test data.
#         type_choice (str): The type of scaling to be performed ('standardize' or 'normalize').
        
#     Returns:
#         A tuple containing the scaled training dataframe, the scaled test dataframe, and the parameters used for the scaling.
#     """
#     numerical_cols = train_dataframe.select_dtypes(include=['float64', 'int64']).columns  # Select numerical columns
#     # Create a scaler object
#     scaler = StandardScaler() if type_choice == 'standardize' else MinMaxScaler(feature_range=(0,1))
    
#     # Fit and transform on the training data
#     scaled_train_data = scaler.fit_transform(train_dataframe[numerical_cols])
#     train_dataframe[numerical_cols] = scaled_train_data
    
#     # Transform the test data using the same parameters
#     scaled_test_data = scaler.transform(test_dataframe[numerical_cols])
#     test_dataframe[numerical_cols] = scaled_test_data
#     # Dictionary to store the parameters used for scaling to be used for reverse transformation
#     params = {'mean': scaler.mean_, 'scale': scaler.scale_} if type_choice == 'standardize' else {'min': scaler.data_min_, 'max': scaler.data_max_}
#     # Include index and column names in the scaled dataframes
#     # train_dataframe = pd.DataFrame(train_dataframe, columns=df_adjusted.columns)
#     # test_dataframe = pd.DataFrame(test_dataframe, columns=df_adjusted.columns)

#     return train_dataframe, test_dataframe, params


# def unscale_data(dataframe, original_params, type_choice='standardize'):
#     """
#     Reverses the scaling of the given dataframe using the provided original parameters.
    
#     Args:
#         dataframe: DataFrame that was scaled and now needs to be unscaled.
#         original_params (dict): The parameters used for the original scaling.
#         type_choice (str): Indicates whether the original transformation was 'standardize' or 'normalize'.
        
#     Returns:
#         DataFrame: The dataframe with the reversed scaling.
#     """
#     numerical_cols = dataframe.select_dtypes(include=['float64', 'int64']).columns  # Select numerical columns
    
#     if type_choice == 'standardize':
#         mean = original_params['mean']
#         scale = original_params['scale']
#         dataframe[numerical_cols] = (dataframe[numerical_cols] * scale) + mean
#     else:  # 'normalize'
#         min_val = original_params['min']
#         max_val = original_params['max']
#         dataframe[numerical_cols] = (dataframe[numerical_cols] * (max_val - min_val)) + min_val
    
#     return dataframe


def test_stationarity_and_differenciate(df, reverse=False, diff_dict=None):
    """
    Test each variable in the dataframe for stationarity using the Augmented Dickey-Fuller test.
    If the p-value is greater than 0.05, the variable will be differenced.
    Stores the initial value for possible reversal.
    
    Parameters:
    - df: DataFrame to be tested and possibly differenced.
    - reverse: If True, reverse the differencing based on a provided dictionary that includes initial values.
    - diff_dict: Dictionary indicating which variables were differenced and their initial values.
    
    Returns:
    - DataFrame after testing for stationarity and applying differencing if needed.
    - Dictionary indicating which variables were differenced and their initial values.
    """
    if reverse:
        if diff_dict is None:
            print('No dictionary provided for reversing differencing.')
            return df, None
        for col, info in diff_dict.items():
            if info['seasonal_differenced']:
                # revert seasonal differencing
                df[col] = df[col].shift(-12).fillna(0).cumsum() + info['initial_value']
            if info['differenced']:
                # Revert first order differencing
                df[col] = df[col].cumsum()
        return df
    # Is it's not reversing differencing, let's test for stationarity and difference if needed
    diff_dict = {}
    for col in df.columns:
        initial_value = df[col].iloc[0]
        initial_index = df[col].index[0]
        result = adfuller(df[col], autolag='AIC')
        diff_seasonal = False
        diff_1st = False
        if result[1] > 0.05: # p-value > 0.05
            df[col] = df[col].diff().dropna()
            diff_1st = True
        # Check for seasonality with autocorrelation at lag 12
        acf_values = acf(df[col], nlags=12, fft=True)
        if abs(acf_values[12]) > 0.75: # arbitrary threshold for significant seasonality
            df[col] = df[col].diff(12).dropna()
            diff_seasonal = True
        diff_dict[col] = {'differenced': diff_1st, 'initial_value': initial_value, 'initial_index': initial_index, 'seasonal_differenced': diff_seasonal}
    
    if diff_dict is not None: # If any variable was differenced
        df = df.iloc[1:] # Remove the first row of the df which contains some NAs
    return df, diff_dict


def create_synthetic_data(dataframe, lags):
    """
    Creates a dataframe with lags for the given dataframe.
    
    Args:
        dataframe (pd.DataFrame): The dataframe to be modified.
        lags (list): The list of lags to be created.
    
    Returns:
        pd.DataFrame: The dataframe with lags.
    """
    # Get the original features' names
    features = dataframe.columns

    # Initialize a list to hold the new feature DataFrames
    new_features = []

    # Create lag and rolling window features for all variables
    for feature in features:
        for lag in lags:
            new_features.append(
                dataframe[feature].shift(lag).rename(
                    f'{feature}_synthet_lag_{lag}'))
            new_features.append(
                dataframe[feature].rolling(window=12).mean().rename(
                    f'{feature}_synthet_roll_mean_12'))
            new_features.append(
                dataframe[feature].rolling(window=12).std().rename(
                    f'{feature}_synthet_roll_std_12'))

    # Convert the list of new features to a DataFrame
    sintetic_data = pd.concat(new_features, axis=1)

    return sintetic_data


def plot_forecast_vs_test(target_variable, train, test, forecast, title):
    """
    Plots the forecast and test data.
    
    Args:
        train (pd.DataFrame): The training data.
        test (pd.DataFrame): The test data.
        forecast (pd.DataFrame): The forecast data.
        title (str): The title of the plot.
    """
    # Convert the series to dataframes
    train_df = pd.DataFrame({'date': pd.to_datetime(train.index), target_variable: train.values}).set_index('date').to_period('M')
    test_df = pd.DataFrame({'date': pd.to_datetime(test.index), target_variable: test.values}).set_index('date').to_period('M')

    # Convert index to datetime objects
    train_df.index = train_df.index.to_timestamp()
    test_df.index = test_df.index.to_timestamp()

    # Plot the forecast
    plt.figure(figsize=(10, 6))
    #plt.plot(train_df[target_variable], label='Training')
    plt.plot(test_df[target_variable], label='Test')
    plt.plot(test_df.index, forecast, label='Forecast')
    plt.legend()
    plt.title(title)
    plt.ylabel(target_variable + ' Value')
    plt.xlabel('Date')
    plt.show()


def plot_auto_correlation(model):
    """
    Plots auto-correlation.
    
    Args:
        model to get the residuals
    """
    # Plot ACF and PACF for the residuals
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sm.graphics.tsa.plot_acf(model, lags=40, ax=ax[0])
    sm.graphics.tsa.plot_pacf(model, lags=40, ax=ax[1])

    plt.tight_layout()
    plt.show()


# def replace_zeros_with_previous(df, column_name):
#     """
#     Replaces zeros in a column with the previous non-zero value.
    
#     Args:
#         df (pd.DataFrame): The dataframe containing the column.
#         column_name (str): The name of the column to be modified.
    
#     Returns:
#         pd.DataFrame: The dataframe with zeros replaced by the previous non-zero value.
#     """
#     for i in range(1, len(df)):
#         if df[column_name].iloc[i] == 0:
#             df[column_name].iloc[i] = df[column_name].iloc[i-1]
#     return df


def remove_outliers(dataframe, threshold=0.20):
    """
    Replaces outliers in the dataframe with NAs
    Args:
        dataframe (pd.DataFrame): The dataframe to be modified.
        threshold (float): The quantile threshold to identify outliers.

    Returns:
        pd.DataFrame: The dataframe with outliers replaced.
    """
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


def fill_missing_values(dataframe):
    """
    Fills missing values in the given dataframe.
    
    Each column is converted to float.
    NaNs are filled with the mean of the previous 6 observations. 
    If not enough past data, uses the mean of available past observations.
    If no past data is available, uses the mean of the next 6 observations,
      prioritizing filling from the end of the DataFrame to the start.
    
    Args:
        dataframe (pd.DataFrame): The dataframe to be modified.
    
    Returns:
        pd.DataFrame: The dataframe with missing values filled.
    """
    # Convert every column to float
    dataframe = dataframe.astype('float32')
    
    for column in dataframe.columns:
        # Forward fill: Fill NaNs with the mean of the previous 6 observations
        for index in range(len(dataframe)):
            if pd.isna(dataframe.loc[dataframe.index[index], column]):
                # Compute mean of past 6 or available observations
                past_mean = dataframe[column].iloc[max(0, index-6):index].mean()
                dataframe.at[dataframe.index[index], column] = past_mean if pd.notnull(past_mean) else np.nan
                
        # Backward fill: For remaining NaNs, fill with the mean of the next 6 observations
        for index in range(len(dataframe)-1, -1, -1):  # Iterate in reverse order
            if pd.isna(dataframe.loc[dataframe.index[index], column]):
                # Compute mean of the next 6 or available observations
                next_mean = dataframe[column].iloc[index+1:index+7].mean()
                dataframe.at[dataframe.index[index], column] = next_mean if pd.notnull(next_mean) else np.nan
    
    return dataframe