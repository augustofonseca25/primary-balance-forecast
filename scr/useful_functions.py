# useful_functions.py

"""
Module containing useful functions for the project.
"""
### Useful functions for the project ###

# Import libraries

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from statsmodels.tsa.stattools import adfuller
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


def check_stationarity(time_series):
    """
    Checks the stationarity of the given time series.
    
    Args:
        time_series: The serie to be checked.
    
    Returns:
        p-value: if it's less than 0.05, the series is stationary, False otherwise.
    """

    # if len(clean_dataset(time_series)) > 0:

    # Perform the Dickey-Fuller test
    result = adfuller(time_series)
    p_value = result[1] #extract the result of the test    

    return p_value #threshold for the p-value
    
    # else:
    #     return np.nan  # Return NaN if all data points are missing or infinite


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


def standardize_data(dataframe, type_choice='standardize', reverse=False, original_params=None):
    """
    Standardizes/normalizes the given dataframe and reverses the transformation if specified.
    
    Args:
        dataframe: The dataframe to be standardized/normalized.
        type_choice (str): The type of standardization to be performed ('standardize' or 'normalize').
        reverse (bool): If True, reverses the transformation.
        original_params: The parameters used for standardization/normalization.
        
    Returns:
        DataFrame: The standardized/normalized dataframe.
    """
    numerical_cols = dataframe.select_dtypes(include=['float64', 'int64']).columns # Select numerical columns
    

    # Check if it's to reverse the transformation
    if not reverse:
        if type_choice == 'standardize': # Check the type of standardization
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(dataframe[numerical_cols])
            params = {'mean': scaler.mean_, 'scale': scaler.scale_} # to be used for reverse transformation
        elif type_choice == 'normalize':
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(dataframe[numerical_cols])
            params = {'min': scaler.data_min_, 'max': scaler.data_max_} # to be used for reverse transformation
        dataframe[numerical_cols] = scaled_data # Update the numerical columns in the dataframe
        return dataframe, params # returns the dataframe and the parameters used for standardization/normalization
    else: # Reverse the transformation
        params = original_params # Retrieve the original parameters incase of reverse transformation
        if type_choice == 'standardize': # Check the type of the used standardization
            mean = params['mean'] # Get the mean and scale values
            scale = params['scale']
            dataframe[numerical_cols] = dataframe[numerical_cols] * scale + mean # Reverse the transformation
        elif type_choice == 'normalize': # Check the type of the used normalization
            min_val = params['min'] # Get the min and max values
            max_val = params['max']
            dataframe[numerical_cols] = (dataframe[numerical_cols] * (max_val - min_val) + min_val) # Reverse the transformation
        return dataframe


def apply_differecing(dataframe, order=1, reverse=False, variables_to_diff=None, first_obs = None):
    """
    Applies or reverses differencing to the given dataframe based on the 'reverse' parameter.
    
    Args:
        dataframe (DataFrame): The dataframe to be differenced or reversed.
        order (int): The order of differencing to be applied.
        reverse (bool): If True, reverses the differencing; if False, applies differencing.
        variables_to_diff (list): The list of variables to be differenced or reversed.
        first_obs (DataFrame): The original dataframe to get the first observation for reversing differencing.
    
    Returns:
        pd.DataFrame: The differenced or reversed dataframe.
        list: The list of variables that were differenced.
        int: The order of differencing applied.
    """
    
    
    if not reverse:  # Apply differencing
        variables_to_diff = dataframe.columns.tolist()  # Start with all columns
        df_ori = dataframe.copy()
        for column in variables_to_diff:
            if column != dataframe.index.name: # Not the index
                p_value = check_stationarity(dataframe[column]) # Get the p_value for the ADF test
                if not np.isnan(p_value) and p_value > 0.05: # Check if the p-value is greater than 0.05 and valid
                    #dataframe[column] = dataframe[column].diff(order).fillna(0) # Take the first difference
                    dataframe[column] = dataframe[column].diff(order).fillna(0) # Take the first difference
                else:
                    variables_to_diff.remove(column) # Remove the variable from the list of variables to difference
        return dataframe, variables_to_diff, order, df_ori
    
    else:  # Reverse differencing
        if first_obs is None:
            raise ValueError("When reverse=True, 'first_obs' must be provided.")
        else:
            variables_to_reverse = list(variables_to_diff)
            for column in variables_to_reverse:
                #dataframe[column] = dataframe[column].cumsum().fillna(0)
                dataframe[column] = dataframe[column].cumsum() + first_obs[column].iloc[order-1]
            return dataframe




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