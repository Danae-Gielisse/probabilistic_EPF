import pandas as pd
import numpy as np
import statistics
from scipy.stats import chi2

# choose time span, regularization and distribution
time_span = 2
regularization = 'lasso' # choose lasso or enet
distribution = 'point'
if distribution == 'point':
    run_list = ['qra', 'qrm']
else:
    run_list = [1, 2, 3, 4, 'pEns', 'qEns']

# choose nominal coverage
percentage = 0.5
significance_levels = [0.01, 0.05, 0.1]


def create_CRPS_matrix(forecast):
    number_of_days = int(round(len(forecast) / 24))
    crps_df = pd.DataFrame(index=range(0, number_of_days), columns=range(0, 24))

    # fill CPRS dataframe with average pinball score for every day and hour
    for day in range(0, number_of_days):
        if day == 400:
            print('day 400 is done')
        for hour in range(0, 24):
            crps_df.loc[day, hour] = CRPS(day, hour, forecast)

    # add column with the average pinball score per day
    crps_df['average_pinball_score_day'] = crps_df.mean(axis=1)

    return crps_df


def CRPS(day, hour, forecast):
    # average over all tau's, to calculate the average pinball score
    tau = np.arange(1, 100) / 100
    pinball_list = []
    for q in tau:
        pinball_list.append(pinball_score(q, day, hour, forecast))
    return statistics.mean(pinball_list)


def pinball_score(tau, day, hour, forecast):
    # calculate the pinball score for every day, hour and tau combination
    idx = 24 * day + hour
    real_price = forecast.loc[idx, 'Price']
    quantile = int(round(tau * 100))
    column_quantile = 'Percentile_' + str(quantile)
    price_quantile_predicted = forecast.loc[idx, column_quantile]
    if real_price < price_quantile_predicted:
        pinball = (1-tau) * (price_quantile_predicted - real_price)
    else:
        pinball = tau * (real_price - price_quantile_predicted)

    return pinball


if distribution == 'jsu':
    # calculate CRPS for every run
    CRPS_result_list = []
    for run in run_list:
        forecast_path = f'../Results/percentiles_NN_time_span_{time_span}/percentiles_{distribution}{run}_{regularization}.csv'
        forecast = pd.read_csv(forecast_path, index_col=0)
        CRPS_df = create_CRPS_matrix(forecast)
        average_CRPS = CRPS_df['average_pinball_score_day'].mean()
        CRPS_result_list.append(average_CRPS)

    # create results dataframe
    CRPS_df = pd.DataFrame([CRPS_result_list], columns=run_list)
    crps_path_output = f'../Results/Evaluation_metrics/CRPS_{distribution}_{regularization}_ts{time_span}.csv'

    # save CRPS dataframe
    CRPS_df.to_csv(crps_path_output)
else: # if distribution equals point
    method_list = ['qra', 'qrm']
    CRPS_result_list = []
    for method in method_list:
        forecast_path = f'../Results/percentiles_NN_time_span_{time_span}/percentiles_{distribution}_{method}.csv'
        if method == 'qra':
            forecast = pd.read_csv(forecast_path)
        else:
            forecast = pd.read_csv(forecast_path, index_col=0)
        CRPS_df = create_CRPS_matrix(forecast)
        average_CRPS = CRPS_df['average_pinball_score_day'].mean()
        CRPS_result_list.append(average_CRPS)

    # create results dataframe
    CRPS_df = pd.DataFrame([CRPS_result_list], columns=method_list)
    crps_path_output = f'../Results/Evaluation_metrics/CRPS_{distribution}_ts{time_span}.csv'

    # save CRPS dataframe
    CRPS_df.to_csv(crps_path_output)



### methods for computing emperical coverage ###
'''
computes the emperical coverage for a forecast dataframe for a certain percentage
'''
def empirical_coverage(forecast, percentage):
    lowerbound_quantile = int(round(((1-percentage) / 2) * 100))
    upperbound_quantile = int(round(100 - lowerbound_quantile))
    quantile_low = 'Percentile_' + str(lowerbound_quantile)
    quantile_high = 'Percentile_' + str(upperbound_quantile)
    empirical_coverage_list = []
    for i in range(0, len(forecast)):
        lowerbound = forecast.loc[i, quantile_low]
        upperbound = forecast.loc[i, quantile_high]
        real_price = forecast.loc[i, 'Price']
        if lowerbound <= real_price <= upperbound:
            empirical_coverage_list.append(1)
        else:
            empirical_coverage_list.append(0)

    return statistics.mean(empirical_coverage_list), empirical_coverage_list


'''
computes the emperical coverage for a specific hour
'''
def empirical_coverage_hour(forecast, percentage, hour):
    idx = np.arange(hour, len(forecast), 24)
    hour_forecast = forecast.loc[idx].reset_index(drop=True)
    lowerbound_quantile = int(round(((1-percentage) / 2) * 100))
    upperbound_quantile = int(round(100 - lowerbound_quantile))
    quantile_low = 'Percentile_' + str(lowerbound_quantile)
    quantile_high = 'Percentile_' + str(upperbound_quantile)
    empirical_coverage_list_hour = []
    for i in range(0, len(hour_forecast)):
        lowerbound_hour = hour_forecast.loc[i, quantile_low]
        upperbound_hour = hour_forecast.loc[i, quantile_high]
        real_price = hour_forecast.loc[i, 'Price']
        if lowerbound_hour <= real_price <= upperbound_hour:
            empirical_coverage_list_hour.append(1)
        else:
            empirical_coverage_list_hour.append(0)

    return statistics.mean(empirical_coverage_list_hour), empirical_coverage_list_hour

'''
Performs the kupiec test
'''
def kupiec_test(emperical_coverage_list, confidence_level):
    # calculate the number of times that the real price is in the interval
    n1 = sum(emperical_coverage_list)
    # calculate the number of times that the real price is outside the interval
    n0 = len(emperical_coverage_list) - n1
    # define the total number of observations
    total_observations = len(emperical_coverage_list)

    # calculate nominal coverage (c) and observed failure rate (π)
    c = 1 - confidence_level
    pi = n1 / total_observations

    # compute the likelihood ratio (LR) statistic
    lr_uc = -2 * (
            n0 * np.log(1 - c) + n1 * np.log(c) -
            n0 * np.log(1 - pi) - n1 * np.log(pi)
    )

    # compute the p-value
    p_value = 1 - chi2.cdf(lr_uc, df=1)

    return p_value


kupiec_results = []

for significance_level in significance_levels:
    level_results = []
    for run in run_list:
        if distribution == 'point':
            forecast_path = f'../Results/percentiles_NN_time_span_{time_span}/percentiles_{distribution}_{run}.csv'
            forecast = pd.read_csv(forecast_path)
        else:
            forecast_path = f'../Results/percentiles_NN_time_span_{time_span}/percentiles_{distribution}{run}_{regularization}.csv'
            forecast = pd.read_csv(forecast_path, index_col=0)


        # perform kupiec test
        number_of_passes_dict = {}
        # perform the kupiec test for all 24 hours
        kupiec_list = []
        for hour in range(0, 24):
            ec_hour, coverage_list_hour = empirical_coverage_hour(forecast, percentage, hour)
            kupiec = kupiec_test(coverage_list_hour, 1-percentage)
            kupiec_list.append(kupiec)

        # count number of passes
        number_of_passes = 0
        for j in kupiec_list:
            if j <= significance_level:
                number_of_passes += 1
        level_results.append(number_of_passes)

    # append the level results to the kupiec_list
    kupiec_results.append(level_results)

# create df with number of passes for each run
kupiec_df = pd.DataFrame(kupiec_results, columns=run_list, index=[f"sig_{sl}" for sl in significance_levels])
if distribution == 'point':
    kupiec_output_path = f'../Results/Evaluation_metrics/kupiec_passes_{distribution}_nc{percentage}_ts{time_span}.csv'
else:
    kupiec_output_path = f'../Results/Evaluation_metrics/kupiec_passes_{distribution}_{regularization}_nc{percentage}_ts{time_span}.csv'
# save dataframe
kupiec_df.to_csv(kupiec_output_path)

coverage_results = []
for run in run_list:
    if distribution == 'point':
        forecast_path = f'../Results/percentiles_NN_time_span_{time_span}/percentiles_{distribution}_{run}.csv'
        forecast = pd.read_csv(forecast_path)
    else:
        forecast_path = f'../Results/percentiles_NN_time_span_{time_span}/percentiles_{distribution}{run}_{regularization}.csv'
        forecast = pd.read_csv(forecast_path, index_col=0)
    # calculate empirical coverage
    coverage, _ = empirical_coverage(forecast, percentage)
    # add coverage to results list
    coverage_results.append(coverage)

# create results dataframe
ec_df = pd.DataFrame([coverage_results], columns=run_list)
if distribution == 'point':
    ec_path_output = f'../Results/Evaluation_metrics/empirical_coverage_{distribution}_nc{percentage}_ts{time_span}.csv'
else:
    ec_path_output = f'../Results/Evaluation_metrics/empirical_coverage_{distribution}_{regularization}_nc{percentage}_ts{time_span}.csv'

# save CRPS dataframe
ec_df.to_csv(ec_path_output)
