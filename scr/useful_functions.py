# useful_functions.py

"""
Module containing useful functions for the project.
"""
### Useful functions for the project ###

# Import libraries

import pandas as pd


# Script to adjust the date format
def adjust_index_date_format(dataframe):
    """
    Adjusts the date format of the index in the given dataframe.
    
    Args:
        dataframe (pd.DataFrame): The dataframe to be modified.
    
    Returns:
        pd.DataFrame: The modified dataframe with the adjusted date format.
    """
    
    # # Sort the index in ascending order
    # dataframe = dataframe.sort_index()

    # # Convert the index to datetime
    # dataframe.index = pd.to_datetime(dataframe.index)

    # # Convert the index to the desired format
    # dataframe.index = dataframe.index.strftime('%d-%m-%Y')

    # Convert the index to datetime with desired format
    dataframe.index = pd.to_datetime(dataframe.index)

    # Sort the index in ascending order
    dataframe = dataframe.sort_index()

    # Convert the index to the desired format
    dataframe.index = dataframe.index.strftime('%d-%m-%Y')

    return dataframe



