import os
import sys

root_path = os.path.abspath(os.path.join('../..')) # <- adjust such that root_path always points at the root project dir (i.e. if current file is two folders deep, use '../..'). 
if root_path not in sys.path:
    sys.path.append(root_path)

import database_server.db_utilities as dbu 

import numpy as np
import pandas as pd

# sklearn imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# imports for saving/loading models and data prep objects
import pickle as pkl

# ignore setting with copy warning
pd.options.mode.chained_assignment = None  # default='warn'

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
The FeatureGen class implements all the steps involved in generating features from the data stored in our database.
The primary function of this class is generate_features(), which can produce features for training/testing or individual predictions.

Feature generation parameters are stored in FeatureGen.params (dict). The dictionary looks as follows: 

    feature_gen_params = {
        'ma_alpha': 0.35, # the higher alpha, the more weight is put on recent observations vs. older observations
        'ma_min_periods': 0,
        'ma_restart_each_season': True,

        'h2h_feature_cols': ['result_score'], # list of columns of which h2h features should be generated
        'h2h_alpha': 0.35, # head2head feature EWMA alpha

        'min_non_na_share': 0.9,

        'merge_type': 'wide', # how should feature rows of two teams be combined? -> one of ['wide', 'diff_or_ratio']

        'apply_ohe': False, # True -> one-hot encode selected features, False -> drop all categorical features
        'ohe_name': None, # load fitted ohe from file <- must not be None when generating prediction features!

        'tt_split_cutoff_date': None, # cutoff date is the most recent date to be included in training set
        'tt_split_test_season': '2022-2023',

        'apply_scaler': True,
        'scaler_name': None, # load fitted scaler from file <- must not be None when generating prediction features!

        'apply_pca': True,
        'pca_name': None, # load fitted pca from file (provide filename without .pkl suffix) <- must not be None when generating prediction features!
        'pca_n_components': 0.98, # only relevant if not loading fitted pca (note: n_components can be a fraction between zero and one, in which case the number of components is determined via the explained variance threshold)

        'targets': ['gf', 'ga'], # one of [['gf', 'ga'], ['xg', 'xga']] or list of any single stat column.
        'target_as_diff': False # if True (and two target columns were specified), target is provided as difference between the two columns
    }
"""
class FeatureGen:

    def __init__(self, params_dict=None, db_full_data=None, db_pred_data=None):
        """Constructor for FeatureGen class."""

        self.data_prep_objects_path = os.path.join(root_path, 'models', 'trad_ml', 'saved_data_prep_objects')

        # set self.params, self.run_name and load data prep objects according to params (self.ohe, self.scaler, self.pca)
        self.set_params(new_params_dict=params_dict) 
        self.db_full_data = db_full_data # db data set for training or prediction <- loaded later if not provided
        self.db_pred_data = db_pred_data # db data set for prediction (subset of db_full_data relevant for predictions) <- loaded later if not provided

        # set data prep objects according to params (-> load saved objects if name provided, otherwise set None)
        
        self._set_data_prep_objects() # set self.ohe, self.scaler, self.pca

    def set_params(self, new_params_dict, run_name=None):
        """Updates self.params with new parameters. Also updates self.run_name and loads data prep objects."""
        if new_params_dict is None:
            print("Warning: No feature generation parameters provided!")
        else:
            self.params = new_params_dict
            # update run name (-> used to identify saved models and data prep objects)
            self.run_name = run_name if run_name is not None else ''.join(np.random.choice(list("abcdefghijklmnopqrstuvwxyz"), 6, replace=True).tolist())
            # set data prep objects according to params
            self._set_data_prep_objects() 

    def load_data(self, home_team_id=None, away_team_id=None, return_df=False):
        """Loads relevant data from db into self.db_full_data (training mode) or self.db_pred_data (prediction mode)."""

        # determine training/prediction mode & reject invalid input arguments
        if home_team_id is None and away_team_id is None:
            training = True # training mode -> load full data set
        elif isinstance(home_team_id, int) and isinstance(away_team_id, int):
            training = False
        else:
            raise ValueError("Invalid input arguments for home_team_id and away_team_id. Must be either both None (-> training) or both integers (-> prediction).")
        
        # define query string (filter will be appended in prediction mode)
        query_str = f"""
                    SELECT ms.*, 
                        m.schedule_date, m.schedule_time, m.schedule_round, m.schedule_day,
                        w.annual_wages_eur AS annual_wage_team, 
                        w.annual_wages_eur/w.n_players AS annual_wage_player_avg
                    FROM matchstats ms 
                    LEFT JOIN matches m ON ms.match_id = m.id
                    LEFT JOIN teamwages w ON ms.team_id = w.team_id
                    AND     ms.season_str = w.season_str
                    """ # note: results won't be sorted! (sorting is done in data prep)
        
        # fetch data and store in class attribute
        if training:
            self.db_full_data = dbu.select_query(query_str + ";") # note: can take a few seconds for complete data set
            print(f" - training data set loaded from db, shape: {self.db_full_data.shape}")
        else:
            self.db_pred_data = dbu.select_query(query_str + f" WHERE ms.team_id = {home_team_id} OR ms.team_id = {away_team_id};")
            print(f" - prediction data set (home team id: {home_team_id}, away team id: {away_team_id}) loaded from db, shape: {self.db_pred_data.shape}")

        if return_df:
            return self.db_full_data if training else self.db_pred_data

    
    # primary function 
    def generate_features(self, 
                          df=None,
                          incl_non_feature_cols=False, # if True: leave non-feature cols (ids etc.) included in feature set (and return their names as a third return value)
                          home_team_id=None,
                          away_team_id=None,
                          print_logs=True):
        """
        Generates features for either training or prediction (if home_team_id and away_team_id are passed). 
        Uses feature generation parameters specified in self.params.
        If incl_non_feature_cols is True, returns 3 values: 
            features (incl. non-feature cols), targets and list of non-feature-col names.
        Otherwise, returns 2 values: 
            features (excl. non-feature cols) and targets.
        """

        ### determine training vs. prediction mode & reject invalid input arguments ############################################

        if isinstance(home_team_id, int) and isinstance(away_team_id, int):
            training = False # prediction mode -> create single feature row for prediction
        elif home_team_id is None and away_team_id is None:
            training = True # training mode -> create train/test feature dfs using full data set
        else:
            raise ValueError("Invalid input arguments for home_team_id and away_team_id. Must be either both None (-> training) or both integers (-> prediction).")
        
        if print_logs: print(f"{60*'*'}\nStarting {'training' if training else 'prediction'} feature generation (run_name: {self.run_name}).")

        # ensure params dict is set
        if self.params is None:
            raise ValueError("Must set feature generation parameters before calling generate_features().")

        # for prediction mode, verify that data prep objects are provided
        if not training: 
            if self.params['apply_ohe'] and self.ohe is None:
                raise ValueError("OneHotEncoder object name must be provided for prediction.")
            if self.params['apply_scaler'] and self.scaler is None:
                raise ValueError("StandardScaler object name must be provided for prediction.")
            if self.params['apply_pca'] and self.pca is None:
                raise ValueError("PCA object name must be provided for prediction.")
        # targets must be list
        if not isinstance(self.params['targets'], list):
            raise ValueError("Targets must be a list of one or more column names.")
        # if target should be returned as diff, 'targets' must be list of two col names
        if self.params['target_as_diff']:
            if len(self.params['targets']) != 2:
                raise ValueError("If target_as_diff is True, targets must be a list of two column names.")

        ### load data ##########################################################################################################
        if df is None:
            if training:
                if self.db_full_data is None:
                    self.load_data() # fetch full data set from db
                df = self.db_full_data.copy()
            else:
                if self.db_full_data is not None: 
                    # full data set is already loaded -> filter it to obtain prediction data set
                    self.db_pred_data = self.db_full_data[self.db_full_data['team_id'].astype(int).isin([home_team_id, away_team_id])].copy()
                    if print_logs: print(f" - prediction data set (home team id: {home_team_id}, away team id: {away_team_id}) filtered from full data set, shape: {self.db_pred_data.shape}")
                else:
                    self.load_data(home_team_id=home_team_id, away_team_id=away_team_id) # fetch only prediction-relevant data from db
                df = self.db_pred_data.copy()

        ### define special columns #############################################################################################

        # get non-feature cols (ids, date, season, etc.)
        non_feature_cols = ['schedule_date', 'season_str', 'league_id', 'team_id', 'opponent_id', 'match_id']
        # define cols to omit from moving averages computation (will be appended to later)
        omit_from_ma_cols = non_feature_cols.copy()

        ### feature additions ##################################################################################################
        
        # points variable based on result column
        df['result_score'] = df['result'].map({'W': 1, 'L': -1, 'D': 0})
    
        # head2head 
        df = self._get_head2head_features(df, training)
        h2h_cols = [c for c in df.columns if c.endswith('_h2h')]
        omit_from_ma_cols = omit_from_ma_cols + h2h_cols
        
        if print_logs:
            print(f" - df shape after feature additions: {df.shape}")
            print(f" - number of h2h_ cols: {len([c for c in df.columns if c.endswith('_h2h')])}")

        ### moving averages computation ########################################################################################

        df = self._get_ma_features(df, omit_from_ma_cols, training)
        if print_logs: print(f" - df shape after ma computation: {df.shape}")

        ### prediction mode only: filtering for the two most recent feature rows ###############################################

        if not training:
            ### handle h2h feature for predictions
            # get subsets of relevant rows
            home_vs_away_rows = df.loc[(df['team_id'] == home_team_id) & (df['opponent_id']==away_team_id), :]
            away_vs_home_rows = df.loc[(df['team_id'] == away_team_id) & (df['opponent_id']==home_team_id), :]

            if len(home_vs_away_rows) == 0: # no previous matches between the two -> set h2h values to pd.NA
                h2h_home_vs_away = [pd.NA] * len(h2h_cols)
                h2h_away_vs_home = [pd.NA] * len(h2h_cols)
            else:
                # extract the most recent relevant h2h feature set (from each team's perspective)
                h2h_home_vs_away = home_vs_away_rows.sort_values('schedule_date', ascending=True).tail(1)[h2h_cols].values.tolist()[0]
                h2h_away_vs_home = away_vs_home_rows.sort_values('schedule_date', ascending=True).tail(1)[h2h_cols].values.tolist()[0]

            # filter df for the two most recent feature rows (-> selects relevant ma features for predictions)
            df = df.sort_values(['team_id', 'schedule_date']).groupby(['team_id']).tail(1).reset_index(drop=True)
            
            # set extracted values as h2h feature values in df
            df.loc[df['team_id'] == home_team_id, h2h_cols] = h2h_home_vs_away
            df.loc[df['team_id'] == away_team_id, h2h_cols] = h2h_away_vs_home
            
            if print_logs: print(f" - df shape after filtering for most recent feature rows: {df.shape}")

            # define 'venue' (important for merge!) and non-feature cols 
            df.loc[df['team_id'] == home_team_id, ['opponent_id', 'venue']] = [away_team_id, 'Home']
            df.loc[df['team_id'] == away_team_id, ['opponent_id', 'venue']] = [home_team_id, 'Away']
            df.loc[:, 'match_id'] = -1 # indicates prediction
            df.loc[:, [c for c in non_feature_cols if c not in ['team_id', 'opponent_id', 'match_id']]] = pd.NA

        ### dealing with categorical variables #################################################################################

        # for now: don't encode any -> drop all categoricals 
        encoded_cols = []
        # drop all categoricals not encoded
        cats_to_drop = [c for c in df.select_dtypes(include=['object', 'category']).columns if c not in [*encoded_cols, *h2h_cols, *non_feature_cols, 'venue']]
        df = df.drop(cats_to_drop, axis=1)
        if print_logs: print(f" - df shape after encoding and dropping non-encoded categoricals: {df.shape}")

        ### merge two rows into one for every match ############################################################################

        if self.params['merge_type'] == 'wide':
            df = self._merge_wide(df, non_feature_cols)
        elif self.params['merge_type'] == 'diff_or_ratio':
            #df = self._merge_diff_or_ratio(df)
            pass
        
        if print_logs: print(f" - df shape after merge: {df.shape}")

        ### drop rows with too many missing values #############################################################################
        if print_logs: print(f" - n rows with any na after merge: {df.isna().any(axis=1).sum()}")
        df.dropna(thresh=self.params['min_non_na_share'], inplace=True) # for now: drop all rows with missing values in any col
        if print_logs: print(f" - df shape after dropping na rows over na threshold: {df.shape}")


        ### split into features (X)  and targets (y) ###########################################################################

        X = df[[c for c in df.columns if c not in self.params['targets']]]
        if print_logs: print(f" - X shape after feature/target split: {X.shape}")
        y = df[self.params['targets']] if training else None

        ### train_test_split (if training) #####################################################################################

        X_train, X_test, y_train, y_test = (None, None, None, None) if not training else self._train_test_split(X, y, cutoff_date=self.params['tt_split_cutoff_date'], test_season=self.params['tt_split_test_season'])
        if print_logs and training: print(f" - X_train, X_test, y_train, y_test shapes after train/test split: {X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}")
        ### missing value imputation ###########################################################################################

        # note: only use information from the training set!
        # set missing h2h to 0 
        h2h_cols_after_merge = [f"{c}_home" for c in h2h_cols] + [f"{c}_away" for c in h2h_cols]
        if training:
            X_train[h2h_cols_after_merge] = X_train[h2h_cols_after_merge].fillna(0)
            X_test[h2h_cols_after_merge] = X_test[h2h_cols_after_merge].fillna(0)
        else:
            X[h2h_cols_after_merge] = X[h2h_cols_after_merge].fillna(0)

        # finally: drop any remaining rows with missing values 
        if training:
            X_train.dropna(inplace=True)
            y_train = y_train.loc[X_train.index]
            X_test.dropna(inplace=True)
            y_test = y_test.loc[X_test.index]
            if print_logs: print(f" - X_train, X_test, y_train, y_test shapes after final NA row drop: {X_train.shape, X_test.shape, y_train.shape, y_test.shape}")
        else:
            # print warning if prediction feature row has missing values
            if X[[c for c in X.columns if c not in non_feature_cols]].isna().any(axis=1).sum() > 0:
                print(f"***WARNING: prediction feature row has missing values in {[c for c in X.columns if X[c].isna().any() and c not in non_feature_cols]}")

        ### standardize features ###############################################################################################
        
        if self.params['apply_scaler']:
            # standardize all columns except non-feature cols (note: there shouldn't be any categorical feature cols left at this point)
            cols_to_be_standardized = [c for c in X.columns if c not in non_feature_cols]
            if training: 
                if self.scaler is None: # fit new scaler (and save it)
                    self.scaler = StandardScaler()
                    self.scaler.fit(X_train[cols_to_be_standardized])
                    self._save_data_prep_object(self.scaler)
                # apply scaler
                X_train[cols_to_be_standardized] = self.scaler.transform(X_train[cols_to_be_standardized])
                X_test[cols_to_be_standardized] = self.scaler.transform(X_test[cols_to_be_standardized])
                if print_logs: print(f" - X_train, X_test shapes post scaling: {X_train.shape, X_test.shape}")
            else: # prediction -> scaler is guaranteed to exist
            
                X[cols_to_be_standardized] = self.scaler.transform(X[cols_to_be_standardized])
                if print_logs: print(f" - X shape post scaling: {X.shape}")

        ### PCA ################################################################################################################

        if self.params['apply_pca']:
            pca_cols = [c for c in X.columns if c not in non_feature_cols]

            if training:
                if self.pca is None: # fit new pca (and save it)
                    self.pca = PCA(n_components=self.params['pca_n_components'])
                    self.pca.fit(X_train[pca_cols])
                    self._save_data_prep_object(self.pca)

                # transform training and test data
                X_train_pca_df = pd.DataFrame(self.pca.transform(X_train[pca_cols]))
                X_test_pca_df = pd.DataFrame(self.pca.transform(X_test[pca_cols]))
                X_train = pd.concat([X_train[non_feature_cols].reset_index(drop=True), X_train_pca_df.reset_index(drop=True)], axis=1)
                X_test = pd.concat([X_test[non_feature_cols].reset_index(drop=True), X_test_pca_df.reset_index(drop=True)], axis=1)
                if print_logs: print(f" - X_train, X_test shapes post pca: {X_train.shape, X_test.shape}")

            else: # prediction -> pca is guaranteed to exist
                X_pca_df = pd.DataFrame(self.pca.transform(X[pca_cols]))
                X = pd.concat([X[non_feature_cols].reset_index(drop=True), X_pca_df.reset_index(drop=True)], axis=1)
                print(f" - X shape post pca: {X.shape}")

        ### reset indices (to avoid confusion) #################################################################################

        if training:
            X_train.reset_index(drop=True, inplace=True)
            X_test.reset_index(drop=True, inplace=True)
            y_train.reset_index(drop=True, inplace=True)
            y_test.reset_index(drop=True, inplace=True)
        else:
            X.reset_index(drop=True, inplace=True)

        ### implement 'target_as_diff' #########################################################################################
        if training:
            if self.params['target_as_diff']:
                c1, c2 = self.params['targets'][0], self.params['targets'][1] 
                if training:
                    y_train = (y_train[c1] - y_train[c2]).rename(f"{c1}_{c2}_diff")
                    y_test = (y_test[c1] - y_test[c2]).rename(f"{c1}_{c2}_diff")
                else:
                    y = (y[c1] - y[c2]).rename(f"{c1}_{c2}_diff")

        ### return appropriate data ############################################################################################

        cols_excl_nfc = [c for c in (X_train.columns if training else X.columns) if c not in non_feature_cols] # pca might have changed cols -> X/X_train distinction matters

        if print_logs: print(f"Feature generation complete (run: {self.run_name})")

        if training and incl_non_feature_cols:
            return X_train, X_test , y_train, y_test, non_feature_cols
        
        if training and not incl_non_feature_cols:
            return X_train[cols_excl_nfc], X_test[cols_excl_nfc], y_train, y_test
        
        if not training and incl_non_feature_cols:
                return X, non_feature_cols
        
        if not training and not incl_non_feature_cols:
            return X[cols_excl_nfc]
    
    def _get_ma_features(self, df, omit_from_ma_cols, training=True):
        """ 
        Transforms all float and int columns (except cols to omit) into moving average features.
        For training mode, MAs are lagged by 1 (to avoid data leakage) and all feature rows are returned.
        For prediction mode, MAs are not lagged and only the most recent feature row (for each team) is returned.
        """
        # define ma feature columns (all numeric cols except cols to omit)
        ma_cols = [c for c in df.select_dtypes(include=['float', 'int']) if c not in omit_from_ma_cols]
        
        # define grouping variables
        if self.params['ma_restart_each_season']:
            grouping_vars = ['team_id', 'season_str']
            # note: grouping by season_str as well here helps implement the "drop first x observations for each season"-variation because we can simply drop na rows later.
        else:
            grouping_vars = ['team_id']
        
        # define lag: in prediction mode we want to take the most recent observation into account (no lag), in training mode this is not allowed (lag=1)
        lag = 1 if training else 0

        # sort df ascending by (team_id, schedule_date) -> important!
        df = df.sort_values([*grouping_vars, 'schedule_date', 'schedule_time'], ascending=True)

        # compute moving averages
        ma_features = df.groupby(grouping_vars, group_keys=False)[ma_cols].apply(lambda x: x.shift(lag).ewm(alpha=self.params['ma_alpha'], min_periods=self.params['ma_min_periods']).mean())
        
        # add new columns to df
        df = df.join(ma_features, rsuffix='_ma')
        # drop old (pre-ma) columns (except targets and id cols)
        df = df.drop([c for c in ma_cols if c not in self.params['targets']], axis=1)

        return df
    
    def _get_head2head_features(self, df, training=True):
        """ 
        Adds head2head feature column to df.
        Computation is analogous to _get_ma_features() but with different grouping.
        """
        # note: the lag is applied before filtering for h2h matches.
        lag = 1 if training else 0

        # define grouping vars
        grouping_vars = ['team_id', 'opponent_id']

        # sort df ascending by grouping vars and schedule_date -> important!
        df = df.sort_values([*grouping_vars, 'schedule_date', 'schedule_time'], ascending=True)

        # compute moving averages
        h2h_features = df.groupby(grouping_vars, group_keys=False)[self.params['h2h_feature_cols']].apply(lambda x: x.shift(lag).ewm(alpha=self.params['h2h_alpha'], min_periods=0).mean())

        # add new columns to df
        df = df.join(h2h_features, rsuffix='_h2h')

        return df

    def _merge_wide(self, df, non_feature_cols):
        """
        Merges feature rows for home and away team into one row per match.
        """

        # note: In prediction mode df contains two rows which are to be merged and the order of the rows w.r.t. home team and away team is not known.
        # however, as long as we previously have set 'match_id' to -1 for both, and 'venue' to 'Home' and 'Away' respectively, we can perform the same merge as in training mode.

        # Filter rows where Venue is 'Home' and where Venue is 'Away'
        home_games = df[df['venue'] == 'Home']
        away_games = df[df['venue'] == 'Away']
        # Merge home_games and away_games based on matching fbref_match_id columns
        merged_df = home_games.merge(away_games, on='match_id', suffixes=('_home', '_away'))

        ### recover non-feature-cols, targets after merge
        # drop away-non-feature-cols and targets
        merged_df.drop([f"{c}_away" for c in [*non_feature_cols, *self.params['targets']] if f"{c}_away" in merged_df.columns], axis=1, inplace=True) 
        # remove suffix from home-non-feature-cols and targets
        merged_df.rename(columns={f"{c}_home": c for c in [*non_feature_cols, *self.params['targets']]}, inplace=True)
        
        # delete unnecessary columns
        merged_df.drop(['venue_home', 'venue_away' # redundant, position in dataframe after merge already holds information about venue
                       ], axis=1, inplace=True) 
        return(merged_df)
    
    def _set_data_prep_objects(self):
        """
        Loads data prep objects into instance variables according to the (file-)names provided in self.params. (Set None if no name is provided).
        """
        self.ohe = self._load_data_prep_object(self.params['ohe_name']) if self.params['ohe_name'] is not None else None
        self.scaler = self._load_data_prep_object(self.params['scaler_name']) if self.params['scaler_name'] is not None else None
        self.pca = self._load_data_prep_object(self.params['pca_name']) if self.params['pca_name'] is not None else None


    def _load_data_prep_object(self, obj_name):
        """
        Loads and returns the specified data prep object from the data_prep_objects_path.
        """
        # check if file exists
        if not os.path.isfile(os.path.join(self.data_prep_objects_path, f"{obj_name}.pkl")):
            raise ValueError(f"Data prep object file '{obj_name}.pkl' does not exist in {self.data_prep_objects_path}.")
        else:
            with open(os.path.join(self.data_prep_objects_path, f"{obj_name}.pkl"), 'rb') as f:
                obj = pkl.load(f)
                return obj
    
    def _save_data_prep_object(self, obj, save_name=None):
        """
        Saves the specified data prep object to the data_prep_objects_path.
        """
        if save_name is None:
            save_name = f"{obj.__class__.__name__}_{self.run_name}"
        with open(os.path.join(self.data_prep_objects_path, f"{save_name}.pkl"), 'wb') as f:
            pkl.dump(obj, f)

    def _train_test_split(self, X, y, cutoff_date=None, test_season=None): # think about whether randomizing is fine, if yes -> also add seed argument
        """
        Splits feature and target dfs into train and test sets. Either using a cutoff date or a fixed season for the test set.
        """
        if (cutoff_date is None and test_season is None) or (cutoff_date is not None and test_season is not None):
            raise ValueError("Either cutoff_date or test_season must be provided. (Neither or both are not allowed.)")
        if cutoff_date is not None:
            # split based on 'schedule_date' column in X
            X_train = X[X['schedule_date'] <= cutoff_date]
            y_train = y[X['schedule_date'] <= cutoff_date]
            X_test = X[X['schedule_date'] > cutoff_date]
            y_test = y[X['schedule_date'] > cutoff_date]
        else: # test_season is not None
            # split based on 'season_str' column in X
            X_train = X[X['season_str'] != test_season]
            y_train = y[X['season_str'] != test_season]
            X_test = X[X['season_str'] == test_season]
            y_test = y[X['season_str'] == test_season]
            
        return X_train, X_test, y_train, y_test


