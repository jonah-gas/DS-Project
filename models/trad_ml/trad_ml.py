import os
import sys

root_path = os.path.abspath(os.path.join('../..')) # <- adjust such that root_path always points at the root project dir (i.e. if current file is two folders deep, use '../..'). 
if root_path not in sys.path:
    sys.path.append(root_path)

import database_server.db_utilities as dbu 

import numpy as np
import pandas as pd

from cleaning.data_cleaning import DataCleaning

# sklearn imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import datetime as dt

import pickle as pkl







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
    
    def generate_features(self, 
                          df=None, 
                          apply_pca=False, 
                          fitted_pca=None,
                          saved_pca_name='pca_test',
                          ma_alpha=0.35,
                          ma_min_periods=7,
                          max_na_cols = 0.1,
                          merge=True, # if True: merge two rows into one for every match (with almost twice the number of features)
                                      # otherwise: compute ratios/differences between home and away team features
                          targets=['gf', 'ga']): # should probably add a bunch of parameters here for easier hyperparameter tuning
        
        ## ORDER OF STEPS IN THIS FUNCTION IS REALLY IMPORTANT! ORDER BELOW IS SUBJECT TO CHANGE ##

        ## IS IT POSSIBLE TO MAKE THIS FUNCTION WORK FOR BOTH TRAINING AND PREDICTION FEATURES?

        ## WHEN AND HOW TO DEAL WITH NA VALUES IN BASE DATA / FEATURE SET? ##

        ## ID COLUMNS -> MAKE CATEGORICAL AT THE BEGINNING? ###


        ### load data ###
        if df is None:
            df = self.load_data()
        # sort (important for moving averages to be computed correctly!)
        df = df.sort_values(['team_id', 'schedule_date', 'schedule_time'], ascending=[True, True, True])

        ### data prep ###
        # get id cols
        id_cols = ['season_str', 'league_id', 'team_id', 'opponent_id', 'match_id']
        # convert id cols to categorical
        df[id_cols] = df[id_cols].astype('object')
        # define cols to omit from moving averages computation (will be appended to later)
        omit_from_ma_cols = []

        ### feature additions ###
        
        # points variable based on result column
        df['points'] = df['result'].map({'W': 3, 'L': 0, 'D': 1})

        # attendance feature
        omit_from_ma_cols.append('attendance')
        attendance_feature = df.groupby(['venue'])['attendance'].apply(lambda x: x.shift(1).rolling(window=ma_min_periods).mean())
        df = df.join(attendance_feature, rsuffix='_a')
        df = df.drop('attendance')
        #SPÄTER NUR HOME BEHALTEN 
        

        # formation feature (most common of last x matches)
        #omit_from_ma_cols.append('formation')
        # head2head 
        #omit_from_ma_cols.append('head2head')
        

        ### moving averages computation ###

        # define ma feature columns (all numerical? note that df loaded from db varies between ints and floats depending on where NA values are present)
        # except head2head feature
        ma_cols = df.select_dtypes(include=['float', 'int']).columns.drop(omit_from_ma_cols)
        
        # compute moving averages

        # NOTE: grouping by season_str as well here helps implement the "drop first x observations for each season" because we can simply drop na rows later.
        ma_features = df.groupby(['team_id', 'season_str'])[ma_cols].apply(lambda x: x.shift(1).ewm(alpha=ma_alpha, min_periods=ma_min_periods).mean())
        # add new columns to df
        df = df.join(ma_features, rsuffix='_ma')
        # drop old (pre-ma) columns (except targets and id cols)
        df = df.drop(ma_cols.drop(targets), axis=1)

        ### merge two rows into one for every match
        df = self._merge_data(df)

        ### one-hot encoding of categorical variables ###
        # drop all categoricals not encoded
        cats_to_drop = [c for c in df.select_dtypes(include=['object']).columns if c not in ['season_str', 'schedule_date']]
        df = df.drop(cats_to_drop, axis=1)
        # NOTE: DF STILL CONTAINS TWO NON-FEATURE COLUMNS (season_str and date) WHICH ARE NEEDED FOR SPLIT
        # maybe figure out better solution!

        ### handle missing values ###
        # drop rows with too many missing values 
       
        df.dropna(inplace=True) # for now: drop all rows with missing values in any col


        ### split into features and targets ###
        features, targets = df[df.columns.drop(targets)], df[targets]

        ### normalization ###
        # NOTE: EXCLUDE OBJECT COLUMNS FROM NORMALIZATION!
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        ### PCA ###
        # if apply_pca: either load saved pca model or accept as parameter
        if apply_pca:
            if fitted_pca is None:
                # load pca from pickle file
                pkl_path = os.path.join(root_path, 'models', 'trad_ml', 'saved_models', 'pca', f'{saved_pca_name}.pkl')
                with open(os.path.join(root_path, '') , 'rb') as f:
                    fitted_pca = pkl.load(f)

            features = fitted_pca.transform(features)
            
        return features, targets
    
    def _merge_data(self, data_raw_cleaned):
        """
        Merges the two sets of observations for one game and removes duplicated rows.

        Arguments:
            data_raw_cleaned (pd.DataFrame): Cleaned data.
        Returns:
            pd.DataFrame: Merged data.
        """
        # Filter rows where Venue is 'Home' and where Venue is 'Away'
        home_games = data_raw_cleaned[data_raw_cleaned['venue'] == 'Home']
        away_games = data_raw_cleaned[data_raw_cleaned['venue'] == 'Away']

        # Get the list of column names in home_games DataFrame and in away_games DataFrame
        column_names_home = home_games.columns.tolist()
        column_names_away = away_games.columns.tolist()

        # Create a dictionary mapping old column names to new column names for home_games and away_games
        new_column_names_home = {old_name: f"{old_name}_home" for old_name in column_names_home}
        new_column_names_away = {old_name: f"{old_name}_away" for old_name in column_names_away}

        # Use the rename method to rename columns in home_games and away_games with the new names
        home_games = home_games.rename(columns=new_column_names_home)
        away_games = away_games.rename(columns=new_column_names_away)

        # Merge home_games and away_games based on matching fbref_match_id columns
        merged_data = home_games.merge(away_games, left_on="match_id_home", right_on="match_id_away")

        # Find duplicated rows due to merging and delete one of them while renaming the other
        duplicated_cols = []
        dropped_cols = []
        for i, col in enumerate(merged_data.columns[:-1]):
            for j in range(i+1, len(merged_data.columns)):
                if merged_data[col].equals(merged_data.iloc[:, j]):
                    duplicated_cols.append((col, merged_data.columns[j]))

        for col_pair in duplicated_cols:
            if col_pair[1] not in dropped_cols:
                merged_data = merged_data.drop(col_pair[1], axis=1)
                dropped_cols.append(col_pair[1])

        for col_pair in duplicated_cols:
            col_name = col_pair[0]
            new_col_name = col_name[:-5]
            merged_data = merged_data.rename(columns={col_name: new_col_name})
        
        # Rename confusing column names
        merged_data = merged_data.rename(columns={"team_id": "home_id",
                                                  "opponent_id": "away_id"})
        # Delete unnecessary columns
        merged_data = merged_data.drop(["venue_home",
                                        "venue_away"], axis=1)

        return(merged_data)
    
    def fit_pca(self, features, n_components=None, save_model=False, save_name=None):

        # Fit the PCA model with the determined number of components
        # (note: n_components can be a fraction between zero and one, in which case the number of components is determined via the explained variance threshold)
        pca_model = PCA(n_components=n_components)
        pca_model.fit(features)

        if save_model:
            if save_name is None:
                save_name = f"pca_model_n={pca_model.n_components_}_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
            # define path to pca model folder
            save_path = os.path.join(root_path, 'models', 'trad_ml', 'saved_models', 'pca')
            # save to path
            with open(os.path.join(save_path, save_name), 'wb') as f:
                pkl.dump(pca_model, f)

        return pca_model
'''

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

    def load_pca(self):
        return pca_model
    
    def save_model(self, model, folder_name, model_name=None):
        pass
        
    

'''

