#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from pandas import concat
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras import backend as K

import tensorflow as tf


# In[6]:


class AutoLSTM():
    
    
    '''
    LSTM Model

    Attributes
    ----------
    self.data: dataframe, the main dataset worked on
    self.n_features: int, number of features
    self.lags: list of length 24, length of past months used to predict the target for each model
    self.leads: list of length 24, representing month predicted forward by each model
    self.models: list of length 24, each entry is a trained model with different lead (from 1 to 24)
    self.predX: dataframe, produced in get_pred_data(), used in get_predict()
    self.truey: list, produced in get_pred_data(), used in get_predict()
    self.scaler: scaler, a fitted minmax-scaler
    self.scaled: array, scaled self.data
    self.values_24: list of length 24，
    
    Params
    ----------
    data_name: str, name of the dataset.
    target_name: str, name of target variable.
    '''
    
    def __init__(self, data_name, target_name):   
        #import data
        curr_path = os.getcwd()
        input_path = os.path.join(curr_path, data_name)

        #set date as index column
        data = pd.read_excel(input_path, index_col=0)
        data.dropna(inplace = True)
        
        #set attributes
        self.data = data
        self.n_features = len(data.columns) - 1
    
    '''
    series_to_supervised()
    the function takes a time series and frames it as a supervised learning dataset.
    Modified from: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
    
    Params
    ----------
    data: dataframe, the input time series dataset
    n_in: int, number of month to include backward for each row in the output dataframe
    n_out: int, number of month to include forward for each row in the output dataframe
    dropnan: boolean, whether to drop rows include nan
    if_target: boolean, whether to include target itself as a feature
    
    Return
    ----------
    agg: dataframe, the dataset after reframed
    '''
    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True, if_target=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        df_without_target = df.loc[:, df.columns[1:]]
        cols, names = list(), list()
        if if_target:
            for i in range(n_in, 0, -1):
                cols.append(df.shift(i))
                names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
            for i in range(0, n_out):
                cols.append(df.shift(-i))
                if i == 0:
                    names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
                else:
                    names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
            agg = concat(cols, axis=1)
            agg.columns = names
        else:
            for i in range(n_in, 0, -1):
                cols.append(df_without_target.shift(i))
                names += [('var%d(t-%d)' % (j+1, i)) for j in range(1, n_vars)]
            for i in range(0, n_out):
                cols.append(df.shift(-i))
                if i == 0:
                    names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
                else:
                    names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
            agg = concat(cols, axis=1)
            agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    
    '''
    get_predict()
    Get future prediction of the target
    
    Params
    last_month: str, the month before the first month you want to predict. 
        For example, enter '2020-10'then the prediction will begin from 2020-11
    ----------
    
    Return
    pred_y_list: list of length 24, the predicted target value
    self.truey: list of length 24, the corresponding true target value, if exist
    ----------
    '''     
    def get_predict(self, last_month, forward=24):
        pred_y_list = []

        for i in range(forward):
            # get predict input
            self.get_pred_data(i, last_month)

            model = self.models[i]
            #values = self.values_24[i]

            test_X = self.predX

            # reshape input to be 3D [samples, timesteps, features]
            test_X = test_X.reshape((1, 1, test_X.shape[1]))
            
            pred_y = model.predict(test_X)

            test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
            pred_y = pred_y.reshape((len(pred_y), 1))

            inv_yhat = np.concatenate((pred_y, test_X[:, 1:self.n_features+1]), axis=1)
            inv_yhat = self.scaler.inverse_transform(inv_yhat)
            inv_yhat = inv_yhat[:,0]
            # invert scaling for actual

            pred_y_list.append(inv_yhat[0])

        return pred_y_list, self.truey

    '''
    get_backtesting()
    conduct backtesting using all of the 24 trained models
    
    Return
    pred_y_list: array, the predicted target value of all period for the 24 models
    true_y_list: array, the corresponding true target value for the 24 models
    ----------
    '''      
    def get_backtesting(self):
        pred_y_list = []
        true_y_list = []

        for i in range(len(self.leads)):
            model = self.models[i]
            value = self.values_24[i]
            train_X, train_y = value[:, :-1], value[:, -1]
            train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

            pred_y = model.predict(train_X)

            train_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))

            inv_yhat = np.concatenate((pred_y, train_X[:, 1:self.n_features+1]), axis=1)
            inv_yhat = self.scaler.inverse_transform(inv_yhat)
            inv_yhat = inv_yhat[:,0]
            # invert scaling for actual

            train_y = train_y.reshape((len(train_y), 1))
            inv_y = np.concatenate((train_y, train_X[:, 1:self.n_features+1]), axis=1)
            inv_y = self.scaler.inverse_transform(inv_y)
            inv_y = inv_y[:,0]

            pred_y_list.append(inv_yhat)
            true_y_list.append(inv_y)

        return pred_y_list, true_y_list

    '''
    get_pred_data()
    Produce self.predX and self.truey
    
    Params
    ----------
    i: int, representing the lead of the current model
    last_month: str, the month before the first month you want to predict
        For example, enter '2020-10'then the prediction will begin from 2020-11
    ''' 
    def get_pred_data(self, i, last_month):
        index_num = self.data.index.get_loc(last_month)
        self.truey = self.data.iloc[int(index_num.start)+1:int(index_num.start)+25,0].values
        # last reframed data for prediction input
        reframed_predX = self.series_to_supervised(self.scaled, self.lags[i], self.leads[0], False, True)
        reframed_predX.drop(reframed_predX.columns[range(reframed_predX.shape[1] - self.n_features, reframed_predX.shape[1])], axis=1, inplace=True)
        reframed_predX.drop(reframed_predX.columns[range(reframed_predX.shape[1] - 1 - (self.leads[0] - 1) * (self.n_features + 1), reframed_predX.shape[1]-1)], axis=1, inplace=True)

        self.predX = reframed_predX.iloc[index_num,0:-1].values

    '''
    run()
    Train models
    
    Params
    use_target: boolean, whether to use target itself when training models
    lags: list of length 24, representing the lag used in each model
    leads: list of length 24, usually range(1,25), representing the leads for each model
    ----------

    '''
    def run(self, use_target=True, lags=[], leads=[]): 

        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(self.data)

        self.scaler = scaler
        self.scaled = scaled
        self.lags = lags
        self.leads = leads
        self.values_24 = []
        self.models = []
        
        n = 1

        for i in range(len(leads)):
            lag = lags[i]
            lead = leads[i]

            # flatten data
            reframed = self.series_to_supervised(scaled, lag, lead, True, use_target)

            # drop columns we don't want to predict
            reframed.drop(reframed.columns[range(reframed.shape[1] - self.n_features, reframed.shape[1])], axis=1, inplace=True)
            reframed.drop(reframed.columns[range(reframed.shape[1] - 1 - (lead - 1) * (self.n_features + 1), reframed.shape[1]-1)], axis=1, inplace=True)

            values = reframed.values

            self.values_24.append(values)
            self.n = n

            train = values

            # split into input and outputs
            train_X, train_y = train[:, :-1], train[:, -1]

            # reshape input to be 3D [samples, timesteps, features]
            train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

            # create and fit the LSTM network
            model = Sequential()
            model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
            model.add(Dropout(0.1))
            model.add(Dense(1))
            
            def root_mean_squared_error(y_true, y_pred):
                return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
            model.compile(optimizer='adam', loss=root_mean_squared_error)
            
            result = model.fit(train_X, train_y, verbose=0, epochs=25, batch_size=72)
            self.models.append(model)
            self.train_result = result

