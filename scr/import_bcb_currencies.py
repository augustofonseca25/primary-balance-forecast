# -*- coding: utf-8 -*-
"""import_bcb_currencies.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VOEwBI5KIAp-cMi1cvHN0doRRBTS_q_W

### Este é um código p[ara acessar a API do BCB
"""

# Import packages

import pandas as pd
from bcb import currency
from datetime import datetime
import useful_functions

# Define currencies and date range

currencies = ['USD', 'EUR', 'CNY']

# Get the current date
current_date = pd.to_datetime('today').date()

# Find the last weekday
last_weekday = current_date - pd.tseries.offsets.BDay()
last_weekday_formatted = last_weekday.strftime('%Y-%m-%d')

# Define the start date - yyyy/mm/dd
start_date = '2001-01-01'

# Get the data
df_currencies_raw = currency.get(currencies, start = start_date, end = last_weekday_formatted)

# Create a copy of the DataFrame
df_currencies = df_currencies_raw.copy()

# Convert the dataset to monthly frequency getting the last valid value of each month
df_currencies_monthly = useful_functions.convert_df_to_monthly(df_currencies,'%Y-%m-%d')

# Adjust index date format
#df_currencies = adjust_index_date_format(df_currencies)

# Adjust the index name
df_currencies_monthly.rename_axis('Time', inplace=True)

# Save the DataFrame to an Excel file
df_currencies_monthly.to_csv('../data/df_currencies.csv', index=True)