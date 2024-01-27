# -*- coding: utf-8 -*-
"""import_budget_data.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZMqUiAKoYoi-a7H52EHvOGe5jXN5HSlZ

# This script imports brazilian budget execution data
Source of data: https://www.siop.planejamento.gov.br/modulo/login/index.html#/
Data collected in Jan, 25th, 2024.

Função 9 - Previdência, Função 10 - Saúde, Função 12 - Educação

GND 4 e 5 - Investimentos e inversões financeiras
GND 1 - Pessoal

Rp 1 - Obrigatórias
RP 2+ 3 - Discricionárias
Rp 6+7+8+9 - Emendas
"""

import pandas as pd
from useful_functions import *

# Read in the data
df_budget_raw = pd.read_excel('../data/dF_budget_raw_2001-2023_25.01.2024.xlsx')

# Rename columns
df_budget_raw.rename(columns={'Ano': 'year','Função': 'function', 'Grupo de Despesa': 'group', 'Resultado Primário': 'type', 'Empenhado': 'spent_value','Dotação Inicial': 'initial_value' }, inplace=True)

# Extract codes for each column
df_budget_raw['function_code'] = df_budget_raw['function'].str.split(' ', n = 1).str[0]
df_budget_raw['group_code'] = df_budget_raw['group'].str.split(' ', n = 1).str[0]
df_budget_raw['type_code'] = df_budget_raw['type'].str.split(' ', n = 1).str[0]

# Reorder columns
df_budget_raw = df_budget_raw.reindex(columns=['year', 'function', 'function_code', 'group', 'group_code', 'type', 'type_code', 'initial_value', 'spent_value'])

# Adjust the values format
for value in ['initial_value', 'spent_value']:
    df_budget_raw[value] = df_budget_raw[value].str.replace('.', '').str.replace(',', '.').astype(float)


# Filter

desired_functions = ['09', '10', '12'] #Function 9 - Pension, Function 10 - Health, Function 12 - Education

desired_groups = ['1', '4', '5'] # Group 1 - Personal, Group 4 - Investiments and group 5 - Financial changes

desired_types = ['1', '2', '3', '6', '7', '8', '9'] # Group 1 - Personal, Group 4 - Investiments and group 5 - Financial changes

#Create some copies of the original dataframe

# Filter by function
df_function = df_budget_raw[df_budget_raw['function_code'].isin(desired_functions)].copy()[['year', 'function', 'function_code', 'initial_value', 'spent_value']]
df_function.rename(columns={'function_code': 'code'}, inplace=True)

# Filter by group
df_group = df_budget_raw[df_budget_raw['group_code'].isin(desired_groups)].copy()[['year', 'group', 'group_code', 'initial_value', 'spent_value']]
df_group.rename(columns={'group_code': 'code'}, inplace=True)

# Filter by type
df_type = df_budget_raw[df_budget_raw['type_code'].isin(desired_types)].copy()[['year', 'type', 'type_code', 'initial_value', 'spent_value']]
df_type.rename(columns={'type_code': 'code'}, inplace=True)

# Some adjustments for function df

# Group df_function by year and sum the values
df_step= df_function.groupby(['year', 'code']).sum().reset_index()

# Create a new dictionary to store the filtered dataframes
filtered_dfs = {}

# Loop through the types of values to create the filtered dataframes
for i, value in enumerate(['initial_value', 'spent_value']):

    filtered_dfs[i]  = df_step[['year', 'code', value]].copy().pivot(index='year', columns='code', values=value) # filter and pivot the dataframe and them store it in the dictionary

    for column in filtered_dfs[i].columns:
        filtered_dfs[i].rename(columns={column: f"bud_fun_{column}_{value}"}, inplace=True) # rename the columns to avoid confusion

df_function = pd.concat(filtered_dfs.values(), axis=1) # concatenate the dataframes from the dictionary

# Some adjustments for group df

# Group df_group by year and sum the values
df_step= df_group.groupby(['year', 'code']).sum().reset_index()

# Create a new dictionary to store the filtered dataframes
filtered_dfs = {}

# Loop through the types of values to create the filtered dataframes
for i, value in enumerate(['initial_value', 'spent_value']):

    filtered_dfs[i]  = df_step[['year', 'code', value]].copy().pivot(index='year', columns='code', values=value) # filter and pivot the dataframe and them store it in the dictionary

    for column in filtered_dfs[i].columns:
        filtered_dfs[i].rename(columns={column: f"bud_group_{column}_{value}"}, inplace=True) # rename the columns to avoid confusion

# Concatenate the dataframes from the dictionary
df_group = pd.concat(filtered_dfs.values(), axis=1)

# Create new columns to store the sum of the initial values from the groups 4 and 5
df_group['bud_group_invest_initial_value'] = df_group[['bud_group_4_initial_value', 'bud_group_5_initial_value']].sum(axis=1)
df_group.drop(['bud_group_4_initial_value', 'bud_group_5_initial_value'], axis=1, inplace=True)

# Create new columns to store the sum of the spent values from the groups 4 and 5
df_group['bud_group_invest_spent_value'] = df_group[['bud_group_4_spent_value', 'bud_group_5_spent_value']].sum(axis=1)
df_group.drop(['bud_group_4_spent_value', 'bud_group_5_spent_value'], axis=1, inplace=True)

# Rename columns
df_group.rename(columns={'bud_group_1_initial_value': 'bud_group_personal_initial_value',
                        'bud_group_1_spent_value': 'bud_group_personal_spent_value'}, inplace=True)

# Some adjustments for type df

# Group df_type by year and sum the values
df_step= df_type.groupby(['year', 'code']).sum().reset_index()

# Create a new dictionary to store the filtered dataframes
filtered_dfs = {}

#   Loop through the types of values to create the filtered dataframes
for i, value in enumerate(['initial_value', 'spent_value']):

    filtered_dfs[i]  = df_step[['year', 'code', value]].copy().pivot(index='year', columns='code', values=value) # filter and pivot the dataframe and them store it in the dictionary

    for column in filtered_dfs[i].columns:
        filtered_dfs[i].rename(columns={column: f"bud_type_{column}_{value}"}, inplace=True) # rename the columns to avoid confusion

# Concatenate the dataframes from the dictionary
df_type = pd.concat(filtered_dfs.values(), axis=1)

# Create new columns to store the sum of the initial values from the types 6, 7, 8 and 9
df_type['bud_type_amendments_initial_value'] = df_type[['bud_type_6_initial_value', 'bud_type_7_initial_value', 'bud_type_8_initial_value', 'bud_type_9_initial_value']].sum(axis=1)
df_type['bud_type_disc_initial_value'] = df_type[['bud_type_2_initial_value', 'bud_type_3_initial_value']].sum(axis=1)
df_type.drop(['bud_type_6_initial_value', 'bud_type_7_initial_value', 'bud_type_8_initial_value', 'bud_type_9_initial_value', 'bud_type_2_initial_value', 'bud_type_3_initial_value'], axis=1, inplace=True)

# Create new columns to store the sum of the spent values from the types 6, 7, 8 and 9
df_type['bud_type_amendments_spent_value'] = df_type[['bud_type_6_spent_value', 'bud_type_7_spent_value', 'bud_type_8_spent_value', 'bud_type_9_spent_value']].sum(axis=1)
df_type['bud_type_disc_spent_value'] = df_type[['bud_type_2_spent_value', 'bud_type_3_spent_value']].sum(axis=1)
df_type.drop(['bud_type_6_spent_value', 'bud_type_7_spent_value', 'bud_type_8_spent_value', 'bud_type_9_spent_value', 'bud_type_2_spent_value', 'bud_type_3_spent_value'], axis=1, inplace=True)

# Rename columns
df_type.rename(columns={'bud_type_1_initial_value': 'bud_type_mandatory_initial_value',
                        'bud_type_1_spent_value': 'bud_type_mandatory_spent_value'}, inplace=True)


# Concatenate the dataframes
df_budget = pd.concat([df_function, df_group, df_type], axis=1)

# Convert time index to datetime and adjust its format.
df_budget.index = pd.to_datetime(df_budget.index, format='%Y')


# Change the day of the index to the last day of each year, since the data corresponds to the end of each year
#df_budget.index = pd.to_datetime(df_budget.index) + pd.DateOffset(years=1) - pd.DateOffset(days=1)
df_budget.index = pd.to_datetime(df_budget.index) + pd.DateOffset(month=12)

# # Convert the index to the desired format
df_budget.index = df_budget.index.strftime('%d-%m-%Y')

# Adjust the index name
df_budget.rename_axis('Time', inplace=True)


# Export the dataframe to an excel file
df_budget.to_csv('../data/df_budget.csv')