import os
import sys

root_path = os.path.abspath(os.path.join('../..')) # <- adjust such that root_path always points at the root project dir (i.e. if current file is two folders deep, use '../..'). 
if root_path not in sys.path:
    sys.path.append(root_path)

import database_server.db_utilities as dbu 

import numpy as np
import pandas as pd
# import pca
from sklearn.decomposition import PCA





""" This class contains functions for the following steps of the ML pipeline:

    - query database for source data (full data set or only subset required for requested prediction)
    - create feature and target sets
        - data prep steps, moving average computation, onehot encoding, normalization, etc.
    - train test split
    - model training (for various models accepting the same feature set)

    - model evaluation ? """

class TradML:
    def __init__(self):
        # potentially: read in excel file with data prep info

        pass

    def load_data(self):
        query_str = f"""
                    SELECT ms.*, 
                        m.schedule_date, m.schedule_time, m.schedule_round, m.schedule_day,
                        w.annual_wages_eur AS annual_wage_team, 
                        w.annual_wages_eur/w.n_players AS annual_wage_player_avg
                    FROM matchstats ms 
                    LEFT JOIN matches m ON ms.match_id = m.id
                    LEFT JOIN teamwages w ON ms.team_id = w.team_id
                    AND     ms.season_str = w.season_str;
                    """ # random order! (sorting is done in data prep)

        df = dbu.select_query(query_str) # note: can take 5-10 seconds for complete data set
        return df
    
    def generate_features(self, df=None, apply_pca=False, fitted_pca=None): # should probably add a bunch of parameters here for easier hyperparameter tuning
        
        ## ORDER OF STEPS IN THIS FUNCTION IS REALLY IMPORTANT! ORDER BELOW IS SUBJECT TO CHANGE ##

        ## IS IT POSSIBLE TO MAKE THIS FUNCTION WORK FOR BOTH TRAINING AND PREDICTION FEATURES?

        ## WHEN AND HOW TO DEAL WITH NA VALUES IN BASE DATA / FEATURE SET? ##

        ## ID COLUMNS -> MAKE CATEGORICAL AT THE BEGINNING? ###


        # load data
        if df is None:
            df = self.load_data()
        # sort (important for moving averages to be computed correctly!)
        df = df.sort_values(['team_id', 'schedule_date', 'schedule_time'], ascending=[True, True, True])
        
        ### feature additions ###

        # points variable based on result column
        df['points'] = df['result'].map({'W': 3, 'L': 0, 'D': 1})

        # head2head 


        ### moving averages computation ###

        # define ma feature columns (all numerical? note that df loaded from db varies between ints and floats depending on where NA values are present)
        # except head2head feature
        #ma_cols = df.select_dtypes(include=['int', 'float']).columns.drop(['head2head'])
        
        # compute moving averages and add as new columns to df
        ma_features = df.groupby(['team_id'])[ma_cols].apply(lambda x: x.shift(1).ewm(alpha=alpha, min_periods=min_periods).mean())

        # drop old columns 

        ### drop columns not desired for training ###
        # for now: all categoricals?

        ### onehot encoding ###
        # for now: skip since all categoricals have been dropped












        # if apply_pca: either load saved pca model or accept as parameter

        return features, targets
        
    def fit_pca(self, features, n_components=None, save_model=False, save_name=None):

        # Fit the PCA model with the determined number of components
        # (note: n_components can be a fraction between zero and one, in which case the number of components is determined via the explained variance threshold)
        pca_model = PCA(n_components=n_components)
        pca_model.fit(features)

        if save_model:
            # save pca model

            
        return pca_model


    def train_test_split(self, X, y, test_size=None, test_season=None): # think about whether randomizing is fine, if yes -> also add seed argument
        """ Split data into train and test sets. Either using the provided fraction or a fixed season for the test set."""

        return X_train, X_test, y_train, y_test
    
    def fit_xgb(self, X_train, y_train):
        return model
    
    # sklearn linear model overview: https://scikit-learn.org/stable/modules/linear_model.html
    def fit_reg(self, X_train, y_train):
        return model
    
    def fit_ridgereg(self, X_train, y_train):
        return model
    
    def fit_logreg(self, X_train, y_train):
        return model

    def load_saved_model(self):
        return model
    
    def save_model(self, model, folder_name, model_name=None):
        pass



