"""
Most of this code is adopted from:
Marcjasz, G., Narajewski, M., Weron, R., & Ziel, F. (2023). Distributional neural networks for electricity price
forecasting. Energy Economics, 125, 106843.

Run this file to perform the hyperparameter tuning for the DDNN lasso methods.
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

# choose run and time span
run = 1
time_span = 2

# choose distribution
distribution = 'JSU'
paramcount = {'Normal': 2,
              'JSU': 4,
              'Point': None
}

if time_span == 1:
    train_days = 912
    val_days = 312
    total_tv_days = train_days + val_days
    # chosen for 13 retrains and a validation window of 24 for first time span
    val_multi = 13
    val_window = val_days // val_multi
else:
    train_days = 856
    val_days = 286
    total_tv_days = train_days + val_days
    val_multi = 13
    val_window = val_days // val_multi

if not os.path.exists(f'../Results/hyperparameter_tuning'):
    os.mkdir(f'../Results/hyperparameter_tuning')

INP_SIZE = 220
activations = ['sigmoid', 'relu', 'elu', 'tanh', 'softplus', 'softmax']

binopt = [True, False]

cty = 'NL'

print(cty, distribution)

if cty != 'NL':
    raise ValueError('Incorrect country')
if distribution not in paramcount:
    raise ValueError('Incorrect distribution')

# read data file
data = pd.read_csv(f'../Data/processed data/data.csv', index_col=0)
selected_columns = ['price', 'load_forecast', 'total_generation', 'EUA_price', 'API2_coal_price', 'ttf_gas_price']
data = data[selected_columns]
data.index = [datetime.strptime(e, '%Y-%m-%d %H:%M:%S') for e in data.index]
if time_span == 1:
    data = data.iloc[:total_tv_days*24]
else:
    data = data.iloc[35064:35064+total_tv_days*24]


def objective(trial):
    try:
        # prepare the input/output dataframes
        Y = np.zeros((total_tv_days, 24))
        Yf = np.zeros((val_days, 24))
        for d in range(total_tv_days):
            Y[d, :] = data.loc[data.index[d * 24:(d + 1) * 24], 'price'].to_numpy()
        for d in range(val_days):
            Yf[d, :] = data.loc[data.index[(d + train_days) * 24:(d + train_days + 1) * 24], 'price'].to_numpy()
        X = np.zeros((total_tv_days, INP_SIZE))
        for d in range(7, total_tv_days):
            X[d, :24] = data.loc[data.index[(d-1)*24:(d)*24], 'price'].to_numpy() # D-1 price
            X[d, 24:48] = data.loc[data.index[(d-2)*24:(d-1)*24], 'price'].to_numpy() # D-2 price
            X[d, 48:72] = data.loc[data.index[(d-3)*24:(d-2)*24], 'price'].to_numpy() # D-3 price
            X[d, 72:96] = data.loc[data.index[(d-7)*24:(d-6)*24], 'price'].to_numpy() # D-7 price
            X[d, 96:120] = data.loc[data.index[(d)*24:(d+1)*24], data.columns[1]].to_numpy() # D load forecast
            X[d, 120:144] = data.loc[data.index[(d-1)*24:(d)*24], data.columns[1]].to_numpy() # D-1 load forecast
            X[d, 144:168] = data.loc[data.index[(d-7)*24:(d-6)*24], data.columns[1]].to_numpy() # D-7 load forecast
            X[d, 168:192] = data.loc[data.index[(d)*24:(d+1)*24], data.columns[2]].to_numpy() # D RES sum forecast
            X[d, 192:216] = data.loc[data.index[(d-1)*24:(d)*24], data.columns[2]].to_numpy() # D-1 RES sum forecast
            X[d, 216] = data.loc[data.index[(d-2)*24:(d-1)*24:24], data.columns[3]].to_numpy()[0] # D-2 EUA
            X[d, 217] = data.loc[data.index[(d-2)*24:(d-1)*24:24], data.columns[4]].to_numpy()[0] # D-2 API2_Coal
            X[d, 218] = data.loc[data.index[(d-2)*24:(d-1)*24:24], data.columns[5]].to_numpy()[0] # D-2 TTF_Gas
            X[d, 219] = data.index[d].weekday()
        # '''
        # input feature selection
        colmask = [False] * INP_SIZE
        if trial.suggest_categorical('price_D-1', binopt):
            colmask[:24] = [True] * 24
        if trial.suggest_categorical('price_D-2', binopt):
            colmask[24:48] = [True] * 24
        if trial.suggest_categorical('price_D-3', binopt):
            colmask[48:72] = [True] * 24
        if trial.suggest_categorical('price_D-7', binopt):
            colmask[72:96] = [True] * 24
        if trial.suggest_categorical('load_D', binopt):
            colmask[96:120] = [True] * 24
        if trial.suggest_categorical('load_D-1', binopt):
            colmask[120:144] = [True] * 24
        if trial.suggest_categorical('load_D-7', binopt):
            colmask[144:168] = [True] * 24
        if trial.suggest_categorical('RES_D', binopt):
            colmask[168:192] = [True] * 24
        if trial.suggest_categorical('RES_D-1', binopt):
            colmask[192:216] = [True] * 24
        if trial.suggest_categorical('EUA', binopt):
            colmask[216] = True
        if trial.suggest_categorical('Coal', binopt):
            colmask[217] = True
        if trial.suggest_categorical('Gas', binopt):
            colmask[218] = True
        if trial.suggest_categorical('Dummy', binopt):
            colmask[219] = True
        X = X[:, colmask]
        # '''
        Xwhole = X.copy()
        Ywhole = Y.copy()
        metrics_sub = []
        for train_no in range(val_multi):
            print(f"\nStarting validation multi {train_no + 1}/{val_multi}...\n")
            start = val_window * train_no
            X = Xwhole[start:train_days+start, :]
            Xf = Xwhole[train_days+start:train_days+start+val_window, :]
            Y = Ywhole[start:train_days+start, :]
            Yf = Ywhole[train_days+start:train_days+start+val_window, :]
            X = X[7:train_days, :]
            Y = Y[7:train_days, :]
            X = X.astype(np.float32)
            Y = Y.astype(np.float32)
            Xf = Xf.astype(np.float32)
            Yf = Yf.astype(np.float32)
            # begin building a model
            inputs = tfk.Input(shape=(X.shape[1],), dtype=tf.float32) # <= INP_SIZE as some columns might have been turned off
            # batch normalization
            # we decided to always normalize the inputs
            batchnorm = True
            if batchnorm:
                norm = tfk.layers.BatchNormalization(dtype=tf.float32)(inputs)
                last_layer = norm
            else:
                last_layer = inputs
            # dropout
            dropout = trial.suggest_categorical('dropout', binopt)
            if dropout:
                rate = trial.suggest_float('dropout_rate', 0, 1)
                drop = tfk.layers.Dropout(rate)(last_layer)
                last_layer = drop
            # regularization of 1st hidden layer,
            regularize_h1_activation = trial.suggest_categorical('regularize_h1_activation', binopt)
            regularize_h1_kernel = trial.suggest_categorical('regularize_h1_kernel', binopt)
            h1_activation_rate = (0.0 if not regularize_h1_activation
                                  else trial.suggest_float('h1_activation_rate_l1', 1e-5, 1e1, log=True))
            h1_kernel_rate = (0.0 if not regularize_h1_kernel
                              else trial.suggest_float('h1_kernel_rate_l1', 1e-5, 1e1, log=True))
            # define 1st hidden layer with regularization
            hidden = tfk.layers.Dense(trial.suggest_int('neurons_1', 16, 1024, log=False),
                                        activation=trial.suggest_categorical('activation_1', activations),
                                        kernel_regularizer=tfk.regularizers.L1(h1_kernel_rate),
                                        activity_regularizer=tfk.regularizers.L1(h1_activation_rate))(last_layer)
            # regularization of 2nd hidden layer,
            regularize_h2_activation = trial.suggest_categorical('regularize_h2_activation', binopt)
            regularize_h2_kernel = trial.suggest_categorical('regularize_h2_kernel', binopt)
            h2_activation_rate = (0.0 if not regularize_h2_activation
                                  else trial.suggest_float('h2_activation_rate_l1', 1e-5, 1e1, log=True))
            h2_kernel_rate = (0.0 if not regularize_h2_kernel
                              else trial.suggest_float('h2_kernel_rate_l1', 1e-5, 1e1, log=True))
            # define 2nd hidden layer with regularization
            hidden = tfk.layers.Dense(trial.suggest_int('neurons_2', 16, 1024, log=False),
                                        activation=trial.suggest_categorical('activation_2', activations),
                                        kernel_regularizer=tfk.regularizers.L1(h2_kernel_rate),
                                        activity_regularizer=tfk.regularizers.L1(h2_activation_rate),
                                        dtype=tf.float32)(hidden)
            if paramcount[distribution] is None:
                outputs = tfk.layers.Dense(24, activation='linear')(hidden)
                model = tfk.Model(inputs=inputs, outputs=outputs)
                model.compile(optimizer=tfk.optimizers.legacy.Adam(trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)),
                              loss='mae',
                              metrics=['mae'])
            else:
                # now define parameter layers with their regularization
                param_layers = []
                param_names = ["loc", "scale", "tailweight", "skewness"]
                for p in range(paramcount[distribution]):
                    regularize_param_kernel = trial.suggest_categorical('regularize_'+param_names[p], binopt)
                    param_kernel_rate = (0.0 if not regularize_param_kernel
                                         else trial.suggest_float(param_names[p]+'_rate_l1', 1e-5, 1e1, log=True))
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
                            scale=1e-3 + 3 * tf.math.softplus(t[..., 24:])))(linear)
                elif distribution == 'JSU':
                    outputs = tfp.layers.DistributionLambda(
                            lambda t: tfd.JohnsonSU(
                                loc=tf.cast(t[..., :24], tf.float32),
                                scale=tf.cast(1e-3 + 3 * tf.math.softplus(t[..., 24:48]), tf.float32),
                                tailweight=tf.cast(1 + 3 * tf.math.softplus(t[..., 48:72]), tf.float32),
                                skewness=tf.cast(t[..., 72:], tf.float32)
                            ))(linear)
                else:
                    raise ValueError(f'Incorrect distribution {distribution}')
                model = tfk.Model(inputs=inputs, outputs=outputs)
                model.compile(optimizer=tfk.optimizers.legacy.Adam(trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)),
                              loss=lambda y, rv_y: -rv_y.log_prob(y),
                              metrics=['mae'])
            # define callbacks
            callbacks = [tfk.callbacks.EarlyStopping(patience=50, restore_best_weights=True)]
            model.fit(X, Y, epochs=1500, validation_data=(Xf, Yf), callbacks=callbacks, batch_size=32, verbose=0)

            metrics = model.evaluate(Xf, Yf) # for point its a list of one [loss, MAE]
            metrics_sub.append(metrics[0])
            # we optimize the returned value, -1 will always take the model with best MAE
        return np.mean(metrics_sub)

    except Exception as e:
        print(f"Trial failed with error: {str(e)}")
        raise optuna.TrialPruned()


# begin optuna study
optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
study_name = f'FINAL_NL_selection_prob_{distribution.lower()}{run}_ts{time_span}'
storage_name = f'sqlite:///../Results/hyperparameter_tuning/{study_name}'
study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
study.optimize(objective, n_trials=350, show_progress_bar=True)
best_params = study.best_params
print(best_params)
print(study.trials_dataframe())
