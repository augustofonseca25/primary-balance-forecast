# -*- coding: utf-8 -*-
"""import_IMF_api.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16kPBdC2phK33Lede6GdIQCpE-6hU0Fww

# Source: https://github.com/chriscarrollsmith/imfp

# Script to import commodities prices from IMF
"""

import pandas as pd
import requests
import imfp

import useful_functions

# Check the available databases
databases = imfp.imf_databases()

#Get the database for commodity prices
#databases[databases['description'].str.contains("Commodity")]

# Fetch list of valid parameters and input codes for commodity price database
params = imfp.imf_parameters("PCPS")

# Get key names from the params object
#params.keys()

#imfp.imf_parameter_defs("PCPS")

# View the data frame of valid input codes for the frequency parameter
#params['freq']

#params['commodity']

# Fetch the 'freq' input code for Monthly frequency
selected_freq = list(
    params['freq']['input_code'][params['freq']['description'].str.contains("Monthly")]
)

# List of substrings to check for
commodities_codes = ["PCOFFOTM", #Coffee, Other Mild Arabica
                   "PIORECR", #Iron Ore
                   "PMEAT", #PMEAT index
                   "POILBRE", #Brent Crude Oil
                   "PORANG", #Orange
                   "PSOYB", #Soybeans
                   "PSUGA" #Sugar index
                   ]

# Fetch the 'commodity' input code for coal
selected_commodity = list(
    params['commodity']['input_code'][params['commodity']['input_code'].isin(commodities_codes)]
)

# Fetch the 'unit_measure' input code for index
selected_unit_measure = list(
    params['unit_measure']['input_code'][params['unit_measure']['description'].str.contains("Index")]
)

# Request data from the API
df_commo = imfp.imf_dataset(database_id = "PCPS",
         freq = selected_freq, commodity = selected_commodity,
         unit_measure = selected_unit_measure,
         start_year = 2001, end_year = useful_functions.define_end_period("%Y"))

# Display the first few entries in the retrieved data frame
df_commo = df_commo[['time_period', 'commodity', 'obs_value']]
df_commo = df_commo.rename(columns={'time_period': 'Time', 'obs_value': 'value'})
df_commo['commodity'] = df_commo['commodity'].replace({
    'PCOFFOTM': 'Coffee',
    'PIORECR': 'Iron Ore',
    'PMEAT': 'Meat index',
    'POILBRE': 'Brent Crude Oil',
    'PORANG': 'Orange',
    'PSOYB': 'Soybeans',
    'PSUGA': 'Sugar'
})

# Pivot the data frame
df_commodities = df_commo.pivot(index='Time', columns='commodity', values='value')

# Adjust index date format
#df_commodities = useful_functions.adjust_index_date_format(df_rotated)

# Save the data frame to a CSV file
df_commodities.to_csv('../data/df_commodities.csv')