"""
Most of this code is adopted from:
Marcjasz, G., Narajewski, M., Weron, R., & Ziel, F. (2023). Distributional neural networks for electricity price
forecasting. Energy Economics, 125, 106843.

Run this file to create point forecasts and forecasts for the distribution parameters.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import optuna
import logging
import tf_keras as tfk

# choose run, time span and regularization
run = 1
time_span = 1
regularization = 'lasso' # choose lasso or enet

# choose distribution
distribution = 'JSU'
paramcount = {'Normal': 2,
              'JSU': 4,
              'Point': None,
}

# define input size
INP_SIZE = 220
cty = 'NL'

# create folders if not already existed
if not os.path.exists(f'../Results/point_forecasts_NN_time_span_{time_span}'):
    os.mkdir(f'../Results/point_forecasts_NN_time_span_{time_span}')

if not os.path.exists(f'../Results/distparams_NN_time_span_{time_span}'):
    os.mkdir(f'../Results/distparams_NN_time_span_{time_span}')

# print for which country and distribution the run is for
print(cty, distribution)

# give error if incorrect country or incorrect distribution
if cty != 'NL':
    raise ValueError('Incorrect country')
if distribution not in paramcount:
    raise ValueError('Incorrect distribution')

# read data file
data = pd.read_csv(f'../Data/processed data/data.csv', index_col=0)
selected_columns = ['price', 'load_forecast', 'total_generation', 'EUA_price', 'API2_coal_price', 'ttf_gas_price']
data = data[selected_columns]
data.index = [datetime.strptime(e, '%Y-%m-%d %H:%M:%S') for e in data.index]

# define number of days for train and test for the different time spans
if time_span == 1:
    data = data[data.index < datetime(2021, 1, 1)]
    train_days = 912
    val_days = 312
    total_tv_days = train_days + val_days
else:
    data = data[data.index >= datetime(2020, 1, 1)]
    train_days = 856
    val_days = 286
    total_tv_days = train_days + val_days


def runoneday(inp):
    params, dayno = inp
    print('Day ' + str(dayno) + ' started')
    df = data.iloc[dayno*24:dayno*24+total_tv_days*24+24] # df met data van de 1224 dagen voor de dag zelf en dag zelf
    # prepare the input/output dataframes
    Y = np.zeros((total_tv_days, 24))
    # Yf = np.zeros((1, 24)) # no Yf for rolling prediction
    for d in range(total_tv_days):
        Y[d, :] = df.loc[df.index[d*24:(d+1)*24], 'price'].to_numpy()
    Y = Y[7:, :] # skip first 7 days
    # for d in range(1):
    #     Yf[d, :] = df.loc[df.index[(d+1092)*24:(d+1093)*24], 'Price'].to_numpy()
    X = np.zeros((total_tv_days+1, INP_SIZE))
    for d in range(7, total_tv_days+1):
        X[d, :24] = df.loc[df.index[(d-1)*24:(d)*24], 'price'].to_numpy() # D-1 price
        X[d, 24:48] = df.loc[df.index[(d-2)*24:(d-1)*24], 'price'].to_numpy() # D-2 price
        X[d, 48:72] = df.loc[df.index[(d-3)*24:(d-2)*24], 'price'].to_numpy() # D-3 price
        X[d, 72:96] = df.loc[df.index[(d-7)*24:(d-6)*24], 'price'].to_numpy() # D-7 price
        X[d, 96:120] = df.loc[df.index[(d)*24:(d+1)*24], df.columns[1]].to_numpy() # D load forecast
        X[d, 120:144] = df.loc[df.index[(d-1)*24:(d)*24], df.columns[1]].to_numpy() # D-1 load forecast
        X[d, 144:168] = df.loc[df.index[(d-7)*24:(d-6)*24], df.columns[1]].to_numpy() # D-7 load forecast
        X[d, 168:192] = df.loc[df.index[(d)*24:(d+1)*24], df.columns[2]].to_numpy() # D RES sum forecast
        X[d, 192:216] = df.loc[df.index[(d-1)*24:(d)*24], df.columns[2]].to_numpy() # D-1 RES sum forecast
        X[d, 216] = df.loc[df.index[(d-2)*24:(d-1)*24:24], df.columns[3]].to_numpy()[0] # D-2 EUA
        X[d, 217] = df.loc[df.index[(d-2)*24:(d-1)*24:24], df.columns[4]].to_numpy()[0] # D-2 API2_Coal
        X[d, 218] = df.loc[df.index[(d-2)*24:(d-1)*24:24], df.columns[5]].to_numpy()[0] # D-2 TTF_Gas
        X[d, 219] = data.index[d].weekday()
    # '''
    # input feature selection
    colmask = [False] * INP_SIZE
    if params['price_D-1']:
        colmask[:24] = [True] * 24
    if params['price_D-2']:
        colmask[24:48] = [True] * 24
    if params['price_D-3']:
        colmask[48:72] = [True] * 24
    if params['price_D-7']:
        colmask[72:96] = [True] * 24
    if params['load_D']:
        colmask[96:120] = [True] * 24
    if params['load_D-1']:
        colmask[120:144] = [True] * 24
    if params['load_D-7']:
        colmask[144:168] = [True] * 24
    if params['RES_D']:
        colmask[168:192] = [True] * 24
    if params['RES_D-1']:
        colmask[192:216] = [True] * 24
    if params['EUA']:
        colmask[216] = True
    if params['Coal']:
        colmask[217] = True
    if params['Gas']:
        colmask[218] = True
    if params['Dummy']:
        colmask[219] = True
    X = X[:, colmask]
    Xf = X[-1:, :]
    X = X[7:-1, :]
    # begin building a model
    inputs = tfk.Input(X.shape[1],) # <= INP_SIZE as some columns might have been turned off
    # batch normalization
    batchnorm = True
    if batchnorm:
        norm = tfk.layers.BatchNormalization()(inputs)
        last_layer = norm
    else:
        last_layer = inputs
    # dropout
    dropout = params['dropout']
    if dropout:
        rate = params['dropout_rate']
        drop = tfk.layers.Dropout(rate)(last_layer)
        last_layer = drop
    # regularization of 1st hidden layer, 
    #activation - output, kernel - weights/parameters of input
    regularize_h1_activation = params['regularize_h1_activation']
    regularize_h1_kernel = params['regularize_h1_kernel']
    h1_activation_rate = (0.0 if not regularize_h1_activation 
                          else params['h1_activation_rate_l1'])
    h1_kernel_rate = (0.0 if not regularize_h1_kernel 
                      else params['h1_kernel_rate_l1'])
    # define 1st hidden layer with regularization
    hidden = tfk.layers.Dense(params['neurons_1'],
                                activation=params['activation_1'],
                                kernel_regularizer=tfk.regularizers.L1(h1_kernel_rate),
                                activity_regularizer=tfk.regularizers.L1(h1_activation_rate))(last_layer)
    # regularization of 2nd hidden layer, 
    #activation - output, kernel - weights/parameters of input
    regularize_h2_activation = params['regularize_h2_activation']
    regularize_h2_kernel = params['regularize_h2_kernel']
    h2_activation_rate = (0.0 if not regularize_h2_activation 
                          else params['h2_activation_rate_l1'])
    h2_kernel_rate = (0.0 if not regularize_h2_kernel 
                      else params['h2_kernel_rate_l1'])
    # define 2nd hidden layer with regularization
    hidden = tfk.layers.Dense(params['neurons_2'],
                                activation=params['activation_2'],
                                kernel_regularizer=tfk.regularizers.L1(h2_kernel_rate),
                                activity_regularizer=tfk.regularizers.L1(h2_activation_rate))(hidden)
    if paramcount[distribution] is None:
        outputs = tfk.layers.Dense(24, activation='linear')(hidden)
        model = tfk.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tfk.optimizers.legacy.Adam(params['learning_rate']),
                      loss='mae',
                      metrics='mae')
    else:
        # now define parameter layers with their regularization
        param_layers = []
        param_names = ["loc", "scale", "tailweight", "skewness"]
        for p in range(paramcount[distribution]):
            regularize_param_kernel = params['regularize_'+param_names[p]]
            param_kernel_rate = (0.0 if not regularize_param_kernel 
                                 else params[str(param_names[p])+'_rate_l1'])
            param_layers.append(tfk.layers.Dense(
                24, activation='linear', # kernel_initializer='ones',
                kernel_regularizer=tfk.regularizers.L1(param_kernel_rate))(hidden))
        # concatenate the parameter layers to one
        linear = tfk.layers.concatenate(param_layers)
        # define outputs
        if distribution == 'Normal':
            outputs = tfp.layers.DistributionLambda(
                    lambda t: tfd.Normal(
                        loc=t[..., :24],
                        scale = 1e-3 + 3 * tf.math.softplus(t[..., 24:])))(linear)
        elif distribution == 'JSU':
            outputs = tfp.layers.DistributionLambda(
                    lambda t: tfd.JohnsonSU(
                        loc=t[..., :24],
                        scale=1e-3 + 3 * tf.math.softplus(t[..., 24:48]),
                        tailweight= 1 + 3 * tf.math.softplus(t[..., 48:72]),
                        skewness=t[..., 72:]))(linear)
        else:
            raise ValueError(f'Incorrect distribution {distribution}')
        model = tfk.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tfk.optimizers.legacy.Adam(params['learning_rate']),
                      loss=lambda y, rv_y: -rv_y.log_prob(y),
                      metrics='mae')
    # define callbacks
    callbacks = [tfk.callbacks.EarlyStopping(patience=50, restore_best_weights=True)]
    perm = np.random.permutation(np.arange(X.shape[0]))
    VAL_DATA = .2
    trainsubset = perm[:int((1 - VAL_DATA)*len(perm))]
    valsubset = perm[int((1 - VAL_DATA)*len(perm)):]
    model.fit(X[trainsubset], Y[trainsubset], epochs=1500, validation_data=(X[valsubset], Y[valsubset]), callbacks=callbacks, batch_size=32, verbose=False)

    if paramcount[distribution] is not None:
        dist = model(Xf)
        if distribution == 'Normal':
            getters = {'loc': dist.loc, 'scale': dist.scale}
        elif distribution == 'JSU':
            getters = {'loc': dist.loc, 'scale': dist.scale, 
                       'tailweight': dist.tailweight, 'skewness': dist.skewness}
        params = {k: [float(e) for e in v.numpy()[0]] for k, v in getters.items()}
        df_params = pd.DataFrame({
            'day': [dayno] * 24,  # 24 rijen met dezelfde dagwaarde
            'hour': list(range(24)),  # Uren 0 t/m 23
            'loc': params['loc'],  # Waarden uit 'loc'
            'scale': params['scale'],  # Waarden uit 'scale'
            'tailweight': params['tailweight'],  # Waarden uit 'tailweight'
            'skewness': params['skewness']  # Waarden uit 'skewness'
        })
        pred = model.predict(np.tile(Xf, (10000, 1)))
        predDF = pd.DataFrame(index=df.index[-24:])
        predDF['real'] = df.loc[df.index[-24:], 'price'].to_numpy()
        predDF['forecast'] = pd.NA
        predDF.loc[predDF.index[:], 'forecast'] = pred.mean(0)
    else: # for point predictions
        predDF = pd.DataFrame(index=df.index[-24:])
        predDF['real'] = df.loc[df.index[-24:], 'price'].to_numpy()
        predDF['forecast'] = pd.NA
        predDF.loc[predDF.index[:], 'forecast'] = model.predict(Xf)[0]
        df_params = None
    return df_params, predDF


optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
if regularization == 'enet':
    study_name = f'FINAL_NL_selection_prob_{distribution.lower()}{run}_{regularization}_ts{time_span}'
else:
    study_name = f'FINAL_NL_selection_prob_{distribution.lower()}{run}_ts{time_span}'
storage_name = f'sqlite:///../Results/hyperparameter_tuning/{study_name}'
study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
print(study.trials_dataframe())
best_params = study.best_params
print(best_params)

inputlist = [(best_params, day) for day in range((len(data) // 24) - total_tv_days)] # lijst met (parameters, dag_test)

start_day = 0
inputlist = inputlist[start_day:]
first_iteration = True

dicts = [d[0] for d in inputlist]
labels = [d[1] for d in inputlist]
df = pd.DataFrame(dicts)
print(len(inputlist))

predictions_output_folder = f'../Results/point_forecasts_NN_time_span_{time_span}/{distribution.lower()}{run}_{regularization}.csv'
parameters_output_folder = f'../Results/distparams_NN_time_span_{time_span}/parameter_df_{distribution.lower()}{run}_{regularization}.csv'

for e in inputlist:
    df_parameters, point_prediction_df = runoneday(e)
    if first_iteration:
        if paramcount[distribution] is not None:
            df_parameters.to_csv(parameters_output_folder, mode='w', header=True, index=False)
        point_prediction_df.to_csv(predictions_output_folder, mode='w', header=True, index=False)
        first_iteration = False
    else:
        if paramcount[distribution] is not None:
            df_parameters.to_csv(parameters_output_folder, mode='a', header=False, index=False)
        point_prediction_df.to_csv(predictions_output_folder, mode='a', header=False, index=False)

