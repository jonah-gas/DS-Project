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
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

import datetime as dt

import pickle as pkl

# ignore setting with copy warning
pd.options.mode.chained_assignment = None  # default='warn'


"""This class implements all the steps involved in generating features from the data stored in our database.
   The constructor accepts a dictionary of parameters for feature generation. 
   The primary function of this class is generate_features(), which can produce features for training/testing or individual predictions."""

class FeatureGen:
    def __init__(self, params_dict, db_data=None):

        self.params = params_dict
        self.db_data = db_data # db data set for training or prediction <- loaded later if not provided

        # load data prep objects (-> or set None if not provided)
        self.data_prep_objects_path = os.path.join(root_path, 'models', 'trad_ml', 'saved_data_prep_objects')
        self.ohe = self._load_data_prep_object(self.params['ohe_name']) if self.params['ohe_name'] is not None else None
        self.scaler = self._load_data_prep_object(self.params['scaler_name']) if self.params['scaler_name'] is not None else None
        self.pca = self._load_data_prep_object(self.params['pca_name']) if self.params['pca_name'] is not None else None

    def load_data(self, home_team_id=None, away_team_id=None, return_df=False):

        if home_team_id is None and away_team_id is None:
            where_clause = "" # load all data (-> training)
        elif isinstance(home_team_id, int) and isinstance(away_team_id, int):
            where_clause = f"WHERE team_id = {home_team_id} OR team_id = {away_team_id}" # load data for one team (-> prediction)
        else:
            raise ValueError("Invalid input arguments for home_team_id and away_team_id. Must be either both None (-> training) or both integers (-> prediction).")
        
        query_str = f"""
                    SELECT ms.*, 
                        m.schedule_date, m.schedule_time, m.schedule_round, m.schedule_day,
                        w.annual_wages_eur AS annual_wage_team, 
                        w.annual_wages_eur/w.n_players AS annual_wage_player_avg
                    FROM matchstats ms 
                    LEFT JOIN matches m ON ms.match_id = m.id
                    LEFT JOIN teamwages w ON ms.team_id = w.team_id
                    AND     ms.season_str = w.season_str
                    {where_clause};
                    """ # random order! (sorting is done in data prep)
        self.db_data = dbu.select_query(query_str) # note: can take a few seconds for complete data set
        print(f"Loaded db data set with shape {self.db_data.shape}.")
        if return_df:
            return self.db_data

    
    # primary function of this class
    def generate_features(self, 
                          df=None,
                          incl_non_feature_cols=False, # if True: leave non-feature cols (ids etc.) included in feature set (and return their names as a third return value)
                          training=True # if True: fit and save data prep objects (ohe, scaler, pca) unless filenames are provided
                                         # otherwise (prediction): must load data prep objects 
                          ):
        """Generates features for either training or prediction. Uses feature generation parameters specified in self.params."""

        ### reject invalid input arguments #####################################################################################
        if not training: # prediction
            #if self.params['apply_ohe'] and self.ohe is None:
             #   raise ValueError("OneHotEncoder object name must be provided for prediction.")
            if self.params['apply_scaler'] and self.scaler is None:
                raise ValueError("StandardScaler object name must be provided for prediction.")
            if self.params['apply_pca'] and self.pca is None:
                raise ValueError("PCA object name must be provided for prediction.")

        ### load data ##########################################################################################################
        if df is None:
            if self.db_data is None:
                self.load_data()
            df = self.db_data.copy()

        ### define special columns #############################################################################################

        # get non-feature cols (ids, date, season, etc.)
        non_feature_cols = ['schedule_date', 'season_str', 'league_id', 'team_id', 'opponent_id', 'match_id']
        # define cols to omit from moving averages computation (will be appended to later)
        omit_from_ma_cols = non_feature_cols.copy()

        ### feature additions ##################################################################################################
        
        # points variable based on result column
        df['points'] = df['result'].map({'W': 3, 'L': 0, 'D': 1})

        # attendance feature
        #omit_from_ma_cols.append('attendance')

        # formation feature (most common of last x matches)
        #omit_from_ma_cols.append('formation')
        # head2head 
        #omit_from_ma_cols.append('head2head')

        ### moving averages computation ########################################################################################

        df = self._get_ma_features(df, omit_from_ma_cols)

        print(f"df shape after ma computation: {df.shape}")


        ### dealing with categorical variables #################################################################################

        # for now: don't encode any -> drop all categoricals 
        encoded_cols = []
        # drop all categoricals not encoded
        cats_to_drop = [c for c in df.select_dtypes(include=['object', 'category']).columns if c not in [*non_feature_cols, 'venue']]
        df = df.drop(cats_to_drop, axis=1)
        print(f"df shape after dropping categoricals: {df.shape}")

        ### merge two rows into one for every match ############################################################################

        if self.params['merge_type'] == 'wide':
            df = self._merge_wide(df, non_feature_cols)
        elif self.params['merge_type'] == 'diff_or_ratio':
            #df = self._merge_diff_or_ratio(df)
            pass
        
        print(f"df shape after merge: {df.shape}")

        ### drop rows with too many missing values ##############################################################################
        print(f"n rows with any na after merge: {df.isna().any(axis=1).sum()}")
        df.dropna(thresh=self.params['min_non_na_share'], inplace=True) # for now: drop all rows with missing values in any col
        print(f"df shape after dropping na rows over na threshold: {df.shape}")


        ### split into features (X)  and targets (y) ############################################################################

        X = df[[c for c in df.columns if c not in self.params['targets']]]
        print(f"features pre scaling: {X.shape}")
        y = df[self.params['targets']]

        ### train_test_split (if training) #####################################################################################

        if training:
            X_train, X_test, y_train, y_test = self._train_test_split(X, y, cutoff_date=self.params['tt_split_cutoff_date'], test_season=self.params['tt_split_test_season'])

        ### missing value imputation ###########################################################################################
        # note: only use information from the training set!

        # finally: drop any remaining rows with missing values
        if training:
            X_train.dropna(inplace=True)
            y_train = y_train.loc[X_train.index]
            X_test.dropna(inplace=True)
            y_test = y_test.loc[X_test.index]
            print(f"X_train, y_train shape after final na row drop: {X_train.shape, y_train.shape}")
            print(f"X_test, y_test shape after final na row drop: {X_test.shape, y_test.shape}")
        else:
            X.dropna(inplace=True)
            y = y.loc[X.index]
            print(f"X, y shape after final na row drop: {X.shape, y.shape}")
        
        

        ### standardize features ################################################################################################
        
        if self.params['apply_scaling']:
            # standardize all columns except non-feature cols (note: there shouldn't be any categorical feature cols left at this point)
            cols_to_be_standardized = [c for c in X.columns if c not in non_feature_cols]
            if training: 
                if self.scaler is None: # fit new scaler (and save it)
                    self.scaler = StandardScaler()
                    self.scaler.fit(X_train[cols_to_be_standardized])
                    scaler_name = f"scaler_ncols{len(cols_to_be_standardized)}_{self.params['merge_type']}"
                    self._save_data_prep_object(self.scaler, scaler_name)
                # apply scaler
                X_train[cols_to_be_standardized] = self.scaler.transform(X_train[cols_to_be_standardized])
                X_test[cols_to_be_standardized] = self.scaler.transform(X_test[cols_to_be_standardized])
                print(f"train/test features post scaling: {X_train.shape, X_test.shape}")
            else: # prediction -> scaler is guaranteed to exist
            
                X[cols_to_be_standardized] = self.scaler.transform(X[cols_to_be_standardized])
                print(f"features post scaling: {X.shape}")

        ### PCA ###############################################################################################################

        if self.params['apply_pca']:
            pca_cols = [c for c in X.columns if c not in non_feature_cols]
            if training and self.pca is None:
                # fit pca on training data and save
                self.pca = self._fit_pca(X_train[pca_cols], save=True)
            if training:
                # transform training and test data
                X_train_pca = self.pca.transform(X_train[pca_cols])
                X_test_pca = self.pca.transform(X_test[pca_cols])
                X_train_pca_df = pd.DataFrame(X_train_pca)
                X_test_pca_df = pd.DataFrame(X_test_pca)
                print(f"X_train_pca_df, X_test_pca_df shape: {X_train_pca_df.shape, X_test_pca_df.shape}")
                X_train = pd.concat([X_train_pca_df.reset_index(drop=True), X_train[non_feature_cols].reset_index(drop=True)], axis=1)
                X_test = pd.concat([X_test_pca_df.reset_index(drop=True), X_test[non_feature_cols].reset_index(drop=True)], axis=1)
                print(f"train/test features shape post pca: {X_train.shape, X_test.shape}")
            else: # prediction -> pca is guaranteed to exist
                X_pca = self.pca.transform(X[pca_cols])
                X_pca_df = pd.DataFrame(X_pca)
                X = pd.concat([X_pca_df.reset_index(drop=True), X[non_feature_cols].reset_index(drop=True)], axis=1)
                print(f"features post pca: {X.shape}")
        ### reset indices (to avoid confusion) #################################################################################
        if training:
            X_train.reset_index(drop=True, inplace=True)
            X_test.reset_index(drop=True, inplace=True)
            y_train.reset_index(drop=True, inplace=True)
            y_test.reset_index(drop=True, inplace=True)
        else:
            X.reset_index(drop=True, inplace=True)
            y.reset_index(drop=True, inplace=True)

        ### return appropriate data ############################################################################################
        cols_excl_nfc = [c for c in (X_train.columns if training else X.columns) if c not in non_feature_cols] # pca might have changed cols -> X/X_train distinction matters

        if training and incl_non_feature_cols:
            return X_train, X_test , y_train, y_test, non_feature_cols
        
        if training and not incl_non_feature_cols:
            return X_train[cols_excl_nfc], X_test[cols_excl_nfc], y_train, y_test
        
        if not training and incl_non_feature_cols:
                return X, y, non_feature_cols
        
        if not training and not incl_non_feature_cols:
            return X[cols_excl_nfc], y
            
    
    def _get_ma_features(self, df, omit_from_ma_cols):

        """ Transforms all float and int columns (except cols to omit) into (lagged) moving average features."""
        # define ma feature columns (all numeric cols except cols to omit)
        ma_cols = [c for c in df.select_dtypes(include=['float', 'int']) if c not in omit_from_ma_cols]
        
        # define grouping variables
        if self.params['ma_restart_each_season']:
            grouping_vars = ['team_id', 'season_str']
            # NOTE: grouping by season_str as well here helps implement the "drop first x observations for each season"-variation because we can simply drop na rows later.
        else:
            grouping_vars = ['team_id']
        
        # sort df ascending by (team_id, schedule_date) -> important!
        df = df.sort_values(['team_id', 'schedule_date', 'schedule_time'], ascending=[True, True, True])

        # compute moving averages
        ma_features = df.groupby(grouping_vars)[ma_cols].apply(lambda x: x.shift(1).ewm(alpha=self.params['ma_alpha'], min_periods=self.params['ma_min_periods']).mean())
        # add new columns to df
        df = df.join(ma_features, rsuffix='_ma')
        # drop old (pre-ma) columns (except targets and id cols)
        df = df.drop([c for c in ma_cols if c not in self.params['targets']], axis=1)
        return df

    def _merge_wide(self, df, non_feature_cols, training=True):
        """
        Merges the two sets of observations for one game and removes duplicated cols.

        Arguments:
            df (pd.DataFrame): Feature set (2 rows per match).
        Returns:
            pd.DataFrame: Merged feature set.
        """
        # Filter rows where Venue is 'Home' and where Venue is 'Away'
        home_games = df[df['venue'] == 'Home']
        away_games = df[df['venue'] == 'Away']

        # Get the list of column names in home_games DataFrame and in away_games DataFrame
        column_names_home = home_games.columns.tolist()
        column_names_away = away_games.columns.tolist()

        # Create a dictionary mapping old column names to new column names for home_games and away_games
        new_column_names_home = {old_name: f"{old_name}_home" for old_name in column_names_home}
        new_column_names_away = {old_name: f"{old_name}_away" for old_name in column_names_away}

        # Use the rename method to rename columns in home_games and away_games with the new names
        home_games = home_games.rename(columns=new_column_names_home)
        away_games = away_games.rename(columns=new_column_names_away)

        if training:
            # Merge home_games and away_games based on matching fbref_match_id columns
            merged_data = home_games.merge(away_games, left_on="match_id_home", right_on="match_id_away")
            print(60*'-')
            print(f"merged_data shape (before dropping duplicates): {merged_data.shape}")

            """
            # Find duplicated cols due to merging and delete one of them while renaming the other
            # note: this won't work for prediction because we only have one row -> duplicated cols could just be coincidence
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
            """
        else: # prediction
            # ???
            pass
        
        ### make sure non-feature cols are not duplicated (and remove suffix)
        # drop aways
        print(f"merged_data.shape before dropping aways: {merged_data.shape}")
        merged_data.drop([f"{c}_away" for c in [*non_feature_cols, *self.params['targets']]], axis=1, inplace=True) 
        print(f"merged_data.shape after dropping aways: {merged_data.shape}")
        # remove suffix from homes
        merged_data = merged_data.rename(columns={f"{c}_home": c for c in [*non_feature_cols, *self.params['targets']]}) # can we be sure this is always true?
        # Delete unnecessary columns
        merged_data = merged_data.drop(["venue_home", "venue_away" # redundant, position in dataframe after merge already holds information about venue
                                        ], axis=1)
        
        # DEBUG
        print(f"merged_data shape (after dropping duplicates): {merged_data.shape}")
        print(f"n cols ending with '_home': {len([c for c in merged_data.columns if c.endswith('_home')])}")
        print(f"n cols ending with '_away': {len([c for c in merged_data.columns if c.endswith('_away')])}")
        print(f"n cols with neiter '_home' nor '_away': {len([c for c in merged_data.columns if not c.endswith('_home') and not c.endswith('_away')])}")
        print(f"{[c for c in merged_data.columns if not c.endswith('_home') and not c.endswith('_away')]}")
        print(60*'-')

        return(merged_data)
    
    def _load_data_prep_object(self, obj_name):
        # check if file exists
        if not os.path.isfile(os.path.join(self.data_prep_objects_path, f"{obj_name}.pkl")):
            raise ValueError(f"Data prep object file '{obj_name}.pkl' does not exist in {self.data_prep_objects_path}.")
        else:
            with open(os.path.join(self.data_prep_objects_path, f"{obj_name}.pkl"), 'rb') as f:
                obj = pkl.load(f)
                return obj
    
    def _save_data_prep_object(self, obj, save_name=None):
        if save_name is None:
            save_name = f"{obj.__class__.__name__}_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
        with open(os.path.join(self.data_prep_objects_path, f"{save_name}.pkl"), 'wb') as f:
            pkl.dump(obj, f)

    def _fit_pca(self, features, save=True, save_name=None):

        # Fit the PCA model with the determined number of components
        # (note: n_components can be a fraction between zero and one, in which case the number of components is determined via the explained variance threshold)
        pca_model = PCA(n_components=self.params['pca_n_components'])
        pca_model.fit(features)

        if save:
            if save_name is None:
                save_name = f"pca_n={pca_model.n_components_}_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
            self._save_data_prep_object(pca_model, save_name=save_name)

        return pca_model

    def _train_test_split(self, X, y, cutoff_date=None, test_season=None): # think about whether randomizing is fine, if yes -> also add seed argument
        """ Split feature and target dfs into train and test sets. Either using a cutoff date or a fixed season for the test set."""

        if (cutoff_date is None and test_season is None) or (cutoff_date is not None and test_season is not None):
            raise ValueError("Either cutoff_date or test_season must be provided. (Neither or both are not allowed.)")
        if cutoff_date is not None:
            # split based on 'schedule_date' column in X
            X_train = X[X['schedule_date'] < cutoff_date]
            y_train = y[X['schedule_date'] < cutoff_date]
            X_test = X[X['schedule_date'] >= cutoff_date]
            y_test = y[X['schedule_date'] >= cutoff_date]
        else: # test_season is not None
            # split based on 'season_str' column in X
            X_train = X[X['season_str'] != test_season]
            y_train = y[X['season_str'] != test_season]
            X_test = X[X['season_str'] == test_season]
            y_test = y[X['season_str'] == test_season]
            
        return X_train, X_test, y_train, y_test


""" This class contains functions for the following steps of the ML pipeline:

    - model training (for various models accepting the same feature set)
    - model evaluation ? """

class Training:
    def __init__(self):
        pass


'''

    def fit_xgb(self, X_train, y_train):
        return model
    
    def fit_rf(self, X_train, y_train):
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

