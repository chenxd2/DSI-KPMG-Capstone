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



class AutoLSTM():
    
    '''
    Vector Autoregression model

    Attributes
    ----------
    self.data_backup: dataframe, a backup copy of the input dataset
    self.data: dataframe, the main dataset worked on
    self.n: int, length (unit in months) of target to predict
    self.df_result: dataframe, stores the predicted target and the true target
    self.lag: int, number of past months used to predict the target
    self.rmse: rounded RMSE of the prediction
    self.target: str, name of target variable
    self.model: record a CNN trained model
    self.train_result: record model fit result loss
    
    Params
    ----------
    data_name: str, name of the dataset. Notice the input dataset must contain a column named 'Date'
    target_name: str, name of target variable
    drop_cols: list of strings, names of columns to drop
    '''
    def __init__(self, data_name, target_name):   
        #import data
        curr_path = os.getcwd()
        input_path = os.path.join(curr_path, data_name)

        #set date as index column
        data = pd.read_excel(input_path, index_col=0)
 
        # data.drop(drop_cols, axis=1, inplace=True)
        data.dropna(inplace = True)
        # data.reset_index(drop=True, inplace=True)
        
        #set attributes
        self.data = data
 
        self.n = 0
        self.df_result = 0
        self.lag = 0
        self.rmse = 0
        self.target = target_name
        self.n_features = len(data.columns) - 1
        self.model = 0
        self.train_result = 0

    
    def print_date_range(self):
        #Print Range of Date column
        print('Date Range: ', self.data_backup['Date'].iloc[0], '--', self.data_backup['Date'].iloc[-1])
    
    # convert series to supervised learning
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


    def get_pred_data(self, i, last_month):
        index_num = self.data.index.get_loc(last_month)
        self.truey = self.data.iloc[int(index_num.start)+1:int(index_num.start)+25,0].values
        # last reframed data for prediction input
        reframed_predX = self.series_to_supervised(self.scaled, self.lags[i], self.leads[0], False, True)
        reframed_predX.drop(reframed_predX.columns[range(reframed_predX.shape[1] - self.n_features, reframed_predX.shape[1])], axis=1, inplace=True)
        reframed_predX.drop(reframed_predX.columns[range(reframed_predX.shape[1] - 1 - (self.leads[0] - 1) * (self.n_features + 1), reframed_predX.shape[1]-1)], axis=1, inplace=True)

        self.predX = reframed_predX.iloc[index_num,0:-1].values


    def run(self, use_target=True, lags=[], leads=[]): 
        ''' Run CNN
        
        Params
        ----------
        pred_begin_date: date, the begin date of prediction
        n: int, length of prediction (unit in months)      
        lag: int, num of past months to consider
        if_target: boolean, whether to include target as a predictor
        '''

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
        
            
            # pred_y = self.model.predict(test_X)

            # reverse standardization
            #test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

            # invert scaling for forecast
            # test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
            # pred_y = pred_y.reshape((len(pred_y), 1))

            # inv_yhat = np.concatenate((pred_y, test_X[:, 1:self.n_features+1]), axis=1)
            # inv_yhat = scaler.inverse_transform(inv_yhat)
            # inv_yhat = inv_yhat[:,0]
            # # invert scaling for actual

            # test_y = test_y.reshape((len(test_y), 1))
            # inv_y = np.concatenate((test_y, test_X[:, 1:self.n_features+1]), axis=1)
            # inv_y = scaler.inverse_transform(inv_y)
            # inv_y = inv_y[:,0]

            # pred_y_list.append(inv_yhat)
            # true_y_list.append(inv_y[0])


        # df_result = pd.DataFrame(pred_y_list, columns=[self.target + '_pred'])
        # df_result[self.target] = true_y_list
        # df_result['Date'] = timearray
        # df_result.set_index(['Date'],inplace=True)
        # self.df_result=df_result
        
        # # calculate RMSE
        # rmse = np.sqrt(mean_squared_error(pred_y_list, true_y_list))
        # print('Test RMSE: %.3f' % rmse)
        # self.rmse = round(rmse,2)
