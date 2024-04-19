# Deep Learning in economic indicator forecasting:<br> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Hertie_School_of_Governance_logo.svg/1200px-Hertie_School_of_Governance_logo.svg.png" width="150px" align="right" /> The case of Brazil's primary balance. 

This repository contains all the code files used in the Master Thesis final project for the Master of Data Science program at the Hertie School in Berlin, Germany.

This project uses Autoregressive Integrated Moving Average (ARIMA) models as a baseline - due to their traditional use in economic forecasting - to compare with advanced deep learning models including Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU) in predicting Brazilian Primary Balance. To that end, the models were trained using a dataset comprising monthly data from 2001-2023, including 69 independent variables related to various macroeconomic and financial factors, alongside one dependent variable.

This project was implemented in Python using several different libraries. To organize and clearly state the references, each code script includes, at the beginning of the file, the references used, data sources, and applied packages.

This is the logical structure of this repository:  
Folders:
- data: contains all raw datasets as downloaded, the aggregated dataset (data_orig_parameters.csv), and the subset datasets after feature selection.
- src: contains code files organized as follows:
    - (1) Imports raw data (code files begin with '1_')
    - (2) Merges data and performs EDA and feature selection (code files begin with '2_')
    - (3) Trains models, including on file to grid parameters and other to test subsets (code files begin with '3_')
    - (4) The best GRU model for forecasting (code files begin with '4_').
      - subfolder models_parameters: Located within the 'src' folder, this subfolder contains all saved Keras files with the parameters of the models.

**Author:**
*   Augusto Fonseca [[Email](mailto:cesaraccf@gmail.com) | [GitHub](https://github.com/augustofonseca25) | [LinkedIn](https://www.linkedin.com/in/augustofonseca-brazil)]
