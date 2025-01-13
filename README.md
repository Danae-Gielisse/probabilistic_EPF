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
### Statistical time series methods
1. run point_forecasting_QRA.py to obtain the point forecasts for the QRA methods. This is the same for both LQRA and 
EQRA. Adjust this to the needed time span.
2. run QRA_regularized.py to obtain the quantiles of the probabilistic forecasting for every lambda for both LQRA and 
EQRA. Adjust this to the needed time span.
3. run BIC.py to obtain the quantiles of the probabilistic forecasting for LQRA(BIC) and EQRA(BIC-alpha).

### Neural network based methods
1. run nn_ht_point.py, nn_ht_lasso.py, and nn_ht_enet.py for run 1, 2, 3 and 4 to obtain the best hyperparameter sets
for de DNN, DDNN-L and DDNN-E respectively. 
2. run nn_forecasting.py to obtain point forecasts and forecasts of the distribution parameters.
3. run params_to_percentile.py to convert the distribution parameters in 99 percentiles forecasts.
4. run ensemble.py to do the vertical- and horizontal aggregration.

### Benchmarks
1. run naive.py to create forecasts with the naive method.
2. run stat_benchmarks.py to create forecasts for Stat-QRM and Stat-qEns
3. run QRA_nn.py to create forecasts for DNN-QRA
4. run QRM_nn.py to create forecasts for DNN-QRM

## Evaluation of (probabilistic) electricity price forecasts 
- run evaluate_point_forecasts.py to obtain the MAE and the RMSE for all the methods.
- run evaluate_results_nn.py to calculate the CRPS, get the empirical coverage and perform the kupiec test for the
Neural Network based methods. 
- run evaluate_results_QRA.py to calculate the CRPS, get the empirical coverage and perform the kupiec test for the 
statistical time series based methods. 
- run evaluation_graphs.py to obtain graphs of the results. 