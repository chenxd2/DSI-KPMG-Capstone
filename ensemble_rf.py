#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import concat
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import argrelextrema
from sklearn.metrics import confusion_matrix


# In[147]:


class AutoRF():
    
    
    '''
    Random Forest Model

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
        data = pd.read_excel(input_path, index_col=0)
        
        #drop columns and na
        #data.drop(drop_cols, axis=1, inplace=True)
        data.dropna(inplace = True)
        # data.reset_index(drop=True, inplace=True)
        
        #set attributes
        self.data = data
        self.n_features = len(data.columns) - 1
        
    '''
    preprocess()
    generate varaible 'IsExpanding' based on the business cycle of target variable
    
    Params
    ----------
    dataset: dataframe, the original dataframe
    
    Return
    ----------
    dataset: dataframe, the dataset with a new 'IsExpanding' column
    '''
    
    def preprocess(self, dataset):
        ilocs_min = argrelextrema(dataset['SP500-EPS-Index'].values, np.less_equal, mode = 'wrap', order = 12)[0]
        ilocs_max = argrelextrema(dataset['SP500-EPS-Index'].values, np.greater_equal, mode = 'wrap', order = 12)[0]
        
        # encode expanding period as 1, contracting period as 0
        is_expanding = []
        i = 0
        is_expanding.extend(np.repeat(0, ilocs_min[i]))

        while i < len(ilocs_min) - 1:
            num_expanding = ilocs_max[i] - ilocs_min[i]
            num_contracting = ilocs_min[i + 1] - ilocs_max[i]
            is_expanding.extend(np.repeat(1, num_expanding))
            is_expanding.extend(np.repeat(0, num_contracting))
            i += 1

        is_expanding.extend(np.repeat(1, ilocs_max[i] - ilocs_min[i]))
        is_expanding.extend(np.repeat(0, dataset.shape[0] - ilocs_max[i]))
        dataset.insert(0, 'IsExpanding', is_expanding)
        return dataset
    
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

        
        # last reframed data for prediction input
        reframed_predX = self.series_to_supervised(self.data, self.lags[i], self.leads[0], False, False)
        reframed_predX.drop(reframed_predX.columns[range(reframed_predX.shape[1] - self.n_features, reframed_predX.shape[1])], axis=1, inplace=True)
        reframed_predX.drop(reframed_predX.columns[range(reframed_predX.shape[1] - 1 - (self.leads[0] - 1) * (self.n_features + 1), reframed_predX.shape[1]-1)], axis=1, inplace=True)
        
        self.truey = reframed_predX.iloc[int(index_num.start)+1:int(index_num.start)+25, -1].values
        self.predX = reframed_predX.iloc[index_num,0:-1].values
    
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
            
            self.get_pred_data(i, last_month)
            model = self.models[i]
            

            test_X = self.predX
            test_X = test_X.reshape((1, test_X.shape[1]))
            
            pred_y = model.predict(test_X)

            pred_y_list.append(pred_y[0])
        true_y_list = self.truey
        return pred_y_list, true_y_list
    
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
        
            pred_y = model.predict(train_X)
        
            pred_y_list.append(pred_y)
            true_y_list.append(train_y)

        return pred_y_list, true_y_list

    '''
    run()
    Train models
    
    Params
    use_target: boolean, whether to use target itself when training models
    lags: list of length 24, representing the lag used in each model
    leads: list of length 24, usually range(1,25), representing the leads for each model
    ----------

    '''
    def run(self, lags=[], leads=[]):

        self.lags = lags
        self.leads = leads
        self.values_24 = []
        self.models = []

        n = 1

        self.data = self.preprocess(self.data)
        self.data.drop('SP500-EPS-Index', axis=1, inplace=True)
        self.data.dropna(inplace = True)
        
        for i in range(len(leads)):

            lag = lags[i]
            lead = leads[i]
            # flatten data
            reframed = self.series_to_supervised(self.data, lag, lead, True, False)
            # drop columns we don't want to predict
            reframed.drop(reframed.columns[range(reframed.shape[1] - self.n_features, reframed.shape[1])], axis=1, inplace=True)
            reframed.drop(reframed.columns[range(reframed.shape[1]-1-(self.n_features+1)*(lead-1), reframed.shape[1]-1)], axis=1, inplace=True)

            values = reframed.values
            self.values_24.append(values)
            self.n = n

            train = values

            # split into input and outputs
            train_X, train_y = train[:, :-1], train[:, -1]

            # create and fit the RF
            model = RandomForestClassifier(max_depth=5)

            result = model.fit(train_X, train_y)
            self.models.append(model)

