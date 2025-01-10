# Probababilistic electricity price forecasting for the Dutch EPEX market
Code for performing the methods LQRA, EQRA, DNN and DDNN for probabilistic electricity price forecasting for two 
different time spans. Some code is adopted from the following two papers:
1. Marcjasz, G., Narajewski, M., Weron, R., & Ziel, F. (2023). Distributional neural networks for electricity price 
forecasting. Energy Economics, 125, 106843.
2. Uniejewski, B., & Weron, R. (2021). Regularized quantile regression averaging for probabilistic electricity price 
forecasting. Energy Economics, 95, 105121.

## Preproces data
The data from the ENSOE platform was not directly usable for analysis. Therefore, to prepare the data for probabilistic 
electricity price prediction, the data was first pre-processed. To preprocess the data, the file preprocess_data.py must
be run. The result of the preprocessing is made available 
in the Data folder.

## Obtaining probabilistic electricity price forecasts
# Statistical time series methods
1. run point_forecasting_QRA.py to obtain the point forecasts for the QRA methods. This is the same for both LQRA and 
EQRA. Adjust this to the needed time span.
2. run QRA_regularized.py to obtain the quantiles of the probabilistic forecasting for every lambda for both LQRA and 
EQRA. Adjust this to the needed time span.
3. run BIC.py to obtain the quantiles of the probabilistic forecasting for LQRA(BIC) and EQRA(BIC-alpha).

# Neural network based methods

# Benchmarks

## Evaluation of (probabilistic) electricity price forecasts 
