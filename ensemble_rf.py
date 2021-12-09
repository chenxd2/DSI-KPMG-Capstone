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

class AutoRF():
    '''
    Auto Random Forest model

    Attributes:
        self.data:         pandas.DataFrame --  the main dataset worked on
        self.n:            int -- length (unit in months) of target to predict
        self.target:       str -- name of target variable
        self.n_features:   int -- num of features to use
        self.truey:        list -- a list containing the true target variable
        self.predX:        pandas.DataFrame -- a dataframe containing the predictors for the model
        self.lags:         list -- each element in the list represents the num of months back used by the correspondin model
        self.values_24:    list -- a list of length 24, containing the reshaped values of self.series_to_supervised
        self.models:       list -- a list of lenght 24, containing models of different leads
        self.train_result: list -- the fit() result of models of different leads
    '''
    
    def __init__(self, data_name, target_name):  
        '''
        Initiate the class

        Input:
            data_name:   str -- name of the dataset. Notice the input dataset must contain a column named 'Date'
            target_name: str --  name of target variable
        '''

        #import data
        curr_path = os.getcwd()
        input_path = os.path.join(curr_path, data_name)
        data = pd.read_excel(input_path, index_col=0)
        
        #drop columns and na
        data.dropna(inplace = True)
        
        #set attributes
        self.data = data
        self.n = 0
        self.rmse = 0
        self.target = target_name
        self.n_features = len(data.columns) - 1


    def preprocess(self, dataset):
        '''
        Preprocess the training dataset. Add 'IsExpanding' column to the training dataset as the target variable 
        for the model based on calculated local min and max.

        Input:
            dataset:   pandas.DataFrame -- training dataset

        Output:
            dataset:   pandas.DataFrame -- same dataset with an extra 'IsExpanding' colum
        '''
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
    
    
    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True, if_target=True):
        '''
        Convert the dataset into a rolling window for the supervised learning.

        Input:
            data:      pandas.DataFrame -- training dataset
            n_in:      int, default=1 -- number of month to be converted into one rolling window
            n_out:     int, default=1 -- number of month in the future as the target variable for the rolling windows
            dropnan:   bool, default=True -- whether to drop nan values in the rolling windows
            if_target: bool, default=True -- whether to include target variable as a predictor in the rolling windows

        Output:
            agg:       pandas.DataFrame -- rolling windows of the same length concatenated into one single DataFrame
        ''' 
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
    

    def get_pred_data(self, i, last_month):
        '''
        Helper function to store the true target variables and training data to the class

        Input:
            i:              int -- specifies which lag to use
            last_month:     datetime64 -- the last month of the training data plus one month to retrieve 
                                          the true target variable
        '''
        index_num = self.data.index.get_loc(last_month)

        
        # last reframed data for prediction input
        reframed_predX = self.series_to_supervised(self.data, self.lags[i], self.leads[0], False, False)
        reframed_predX.drop(reframed_predX.columns[range(reframed_predX.shape[1] - self.n_features, reframed_predX.shape[1])], axis=1, inplace=True)
        reframed_predX.drop(reframed_predX.columns[range(reframed_predX.shape[1] - 1 - (self.leads[0] - 1) * (self.n_features + 1), reframed_predX.shape[1]-1)], axis=1, inplace=True)
        
        self.truey = reframed_predX.iloc[int(index_num.start)+1:int(index_num.start)+25, -1].values
        self.predX = reframed_predX.iloc[index_num,0:-1].values
    

    def get_predict(self, last_month, forward=24):
        '''
        Get the prediction result and the true target variable from the model

        Input:
            last_month:     datetime64 -- the last month of the training data plus one month to retrieve 
                                          the true target variable
            forward:        int, default=24 -- number of months in the future to predict

        Output:
            pred_y_list:    list -- a list containing the predicted target variable from the model
            true_y_list:    list -- a list containing the true target variable from the training dataset
        '''

        pred_y_list = []
        
        for i in range(forward):
            
            self.get_pred_data(i, last_month)
            model = self.models[i]
            
            test_X = self.predX
            
            # reshape input to be 3D [samples, timesteps, features]
            test_X = test_X.reshape((1, test_X.shape[1]))
            
            pred_y = model.predict(test_X)

            pred_y_list.append(pred_y[0])
        true_y_list = self.truey
        return pred_y_list, true_y_list
    

    def get_backtesting(self):
        '''
        Specific funciton for backtesting purpose only

        Input:
            None

        Output:
            pred_y_list:    list -- a list containing the predicted target variable from the model
            true_y_list:    list -- a list containing the true target variable from the training dataset
        '''

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


    def run(self, lags, leads):
        ''' 
        Run Random Forest
        
        Input:    
            lags:        list -- each element in the list represents the num of months back used by the correspondin model                   
            leads:       list -- the length of the list represents the num of months forward the model is trying to predict,
                                 starting from 1 
        '''

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

            # create and fit the LSTM network
            model = RandomForestClassifier(max_depth=5)

            result = model.fit(train_X, train_y)
            self.models.append(model)
            self.train_result = result




