# useful_functions.py

"""
Module containing useful functions for the project.
"""
### Useful functions for the project ###

# Import libraries

import pandas as pd
from datetime import datetime, timedelta


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

# Script to get the best value fot the month
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

def define_end_period(format):
    # Define the last month to get data

    # Get today's date
    today = datetime.now()

    # Calculate the year and month before today's date
    end_period = (today - timedelta(days=30)).strftime(format)

    return end_period
