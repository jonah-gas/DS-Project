import torch
import pandas as pd
import numpy as np
import pickle as pkl
import os
import sys
root_path = os.path.abspath(os.path.join('../..'))
if root_path not in sys.path:
    sys.path.append(root_path)
import itertools
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#################################################################################################################################
# Preprocess data
   
class preprocess():
    """A class to prepare the dataset."""
    def __init__(self, data_frame, train = False, pca = "no"):
        self.data_frame = data_frame
        
        self.data_frame = self.data_frame.loc[:,~self.data_frame.columns.duplicated()].copy() # drop duplicated columns
        
        # nominal encoding of categorical variables
        self.venue_dict = {}
        i = 0
        for where in self.data_frame.venue.unique():
            self.venue_dict[where] = i
            i += 1
        
        self.day_dict = {}
        i = 0
        for day in self.data_frame.schedule_day.unique():
            self.day_dict[day] = i
            i += 1
        
        self.result = {}
        i = 0
        for res in self.data_frame.result.unique():
            self.result[res] = i
            i += 1
        
        self.capt = {}
        i = 0
        for cap in set(list(self.data_frame.captain) + list(self.data_frame.captain)): #change
            self.capt[cap] = i
            i += 1
            
        self.formation = {}
        i = 0
        for form in self.data_frame.formation.unique(): #changed
            self.formation[form] = i
            i += 1
            
        self.referee = {}
        i = 0
        for ref in self.data_frame.referee.unique():
            self.referee[ref] = i
            i += 1
        
        self.season = {}
        i = 0
        for seas in self.data_frame.season_str.unique():
            self.season[seas] = i
            i += 1
        
        self.time = {}
        i = 0
        for tim in self.data_frame.schedule_time.unique():
            self.time[tim] = i
            i += 1
            
        self.ligue = {}
        i = 0
        for lig in self.data_frame.league_id.unique():
            self.ligue[lig] = i
            i += 1
            
            
        i = 0
        self.matchweek = {}
        self.data_frame.drop(self.data_frame[self.data_frame.schedule_round == "Relegation tie-breaker"].index, inplace = True) # delete relegation games
        self.data_frame.reset_index(drop = True, inplace = True)
        for week in self.data_frame.schedule_round.unique():
            self.matchweek[week] = int(week.split()[1])
            i += 1

        
        
        self.data_frame.schedule_day.replace(self.day_dict, inplace = True)
        self.data_frame.result.replace(self.result, inplace = True) #change
        self.data_frame.captain.replace(self.capt, inplace = True)
        self.data_frame.formation.replace(self.formation, inplace = True)
        self.data_frame.referee.replace(self.referee, inplace = True)
        self.data_frame.season_str.replace(self.season, inplace = True)
        self.data_frame.schedule_time.replace(self.time, inplace = True)
        self.data_frame.league_id.replace(self.ligue, inplace = True)
        self.data_frame.schedule_round.replace(self.matchweek, inplace = True)
        self.data_frame.venue.replace(self.venue_dict, inplace = True)
        
        self.features = ['shooting_standard_gls', 'shooting_standard_sh', 'shooting_standard_sot', 
                         'shooting_standard_sot_perc', 'shooting_standard_g_per_sh', 'shooting_standard_g_per_sot',
                         'shooting_standard_dist', 'shooting_standard_fk', 'shooting_standard_pk', 
                         'shooting_standard_pkatt', 'shooting_expected_npxg', 'shooting_expected_npxg_per_sh',
                         'shooting_expected_g_minus_xg', 'shooting_expected_npg_minus_xg', 'keeper_performance_sota',
                         'keeper_performance_saves', 'keeper_performance_save_perc', 'keeper_performance_cs',
                         'keeper_performance_psxg', 'keeper_performance_psxg_plus_minus', 'keeper_penaltykicks_pkatt', 
                         'keeper_penaltykicks_pksv', 'keeper_penaltykicks_pkm', 'keeper_launched_cmp', 'keeper_launched_att',
                         'keeper_launched_cmp_perc', 'keeper_passes_att', 'keeper_passes_thr', 'keeper_passes_launch_perc',
                         'keeper_passes_avglen', 'keeper_goalkicks_att', 'keeper_goalkicks_launch_perc', 
                         'keeper_goalkicks_avglen', 'keeper_crosses_opp', 'keeper_crosses_stp', 'keeper_crosses_stp_perc',
                         'keeper_sweeper_number_opa', 'keeper_sweeper_avgdist', 'passing_total_cmp', 'passing_total_att',
                         'passing_total_cmp_perc', 'passing_total_totdist', 'passing_total_prgdist', 'passing_short_cmp',
                         'passing_short_att', 'passing_short_cmp_perc', 'passing_medium_cmp', 'passing_medium_att',
                         'passing_medium_cmp_perc', 'passing_long_cmp', 'passing_long_att', 'passing_long_cmp_perc',
                         'passing_attacking_ast', 'passing_attacking_xag', 'passing_attacking_xa', 'passing_attacking_kp',
                         'passing_attacking_1_per_3', 'passing_attacking_ppa', 'passing_attacking_crspa',
                         'passing_attacking_prgp','passing_types_passtypes_live', 'passing_types_passtypes_dead',
                         'passing_types_passtypes_fk', 'passing_types_passtypes_tb', 'passing_types_passtypes_sw',
                         'passing_types_passtypes_crs', 'passing_types_passtypes_ti', 'passing_types_passtypes_ck', 
                         'passing_types_cornerkicks_in','passing_types_cornerkicks_out', 'passing_types_cornerkicks_str', 
                         'passing_types_outcomes_off','passing_types_outcomes_blocks', 'gca_scatypes_sca', 
                         'gca_scatypes_passlive','gca_scatypes_passdead', 'gca_scatypes_to', 'gca_scatypes_sh', 
                         'gca_scatypes_fld', 'gca_scatypes_def', 'gca_gcatypes_gca', 'gca_gcatypes_passlive', 
                         'gca_gcatypes_passdead', 'gca_gcatypes_to', 'gca_gcatypes_sh', 'gca_gcatypes_fld',  'gca_gcatypes_def', 
                         'defense_tackles_tkl', 'defense_tackles_tklw', 'defense_tackles_def3rd', 'defense_tackles_mid3rd', 
                         'defense_tackles_att3rd', 'defense_challenges_tkl', 'defense_challenges_att', 
                         'defense_challenges_tkl_perc', 'defense_challenges_lost', 'defense_blocks_blocks',
                         'defense_blocks_sh', 'defense_blocks_pass', 'defense_general_int', 'defense_general_tkl_plus_int',
                         'defense_general_clr', 'defense_general_err', 'possession_general_poss', 'possession_touches_touches',
                         'possession_touches_defpen', 'possession_touches_def3rd', 'possession_touches_mid3rd',
                         'possession_touches_att3rd', 'possession_touches_attpen', 'possession_touches_live', 
                         'possession_takeons_att', 'possession_takeons_succ', 'possession_takeons_succ_perc', 
                         'possession_takeons_tkld', 'possession_takeons_tkld_perc', 'possession_carries_carries', 
                         'possession_carries_totdist', 'possession_carries_prgdist', 'possession_carries_prgc', 
                         'possession_carries_1_per_3', 'possession_carries_cpa', 'possession_carries_mis',
                         'possession_carries_dis', 'possession_receiving_rec', 'possession_receiving_prgr',
                         'misc_performance_crdy', 'misc_performance_crdr', 'misc_performance_2crdy', 'misc_performance_fls',
                         'misc_performance_fld', 'misc_performance_off', 'misc_performance_og', 'misc_performance_recov',
                         'misc_aerialduels_won', 'misc_aerialduels_lost', 'misc_aerialduels_won_perc', 'keeper_penaltykicks_pka',
                         'attendance']
        
        # create new variable stadium capacity usage
        for team in self.data_frame.team_id.unique():
            # get maximum attendance in home stadium (implying that max attendance = sold out)
            max_attend = max(
                self.data_frame[(self.data_frame.team_id == team) & 
                                (self.data_frame.venue == self.venue_dict["Home"])
                               ].attendance)
            
            # create variable for home team with stad capacity
            self.data_frame.loc[(self.data_frame.team_id == team) & 
                                (self.data_frame.venue == self.venue_dict["Home"]),"stad_capac"] = self.data_frame[(self.data_frame.team_id == team) & 
            (self.data_frame.venue == self.venue_dict["Home"])
                                                                                                           ].attendance.apply(lambda x: x/max_attend)
            
            # create variable for away team with stad capacity
            self.data_frame.loc[(self.data_frame.opponent_id == team) &
                                (self.data_frame.venue == self.venue_dict["Away"]),"stad_capac"] = self.data_frame[(self.data_frame.opponent_id == team) & 
                                                                                                                   (self.data_frame.venue == self.venue_dict["Away"])].attendance.apply(lambda x: x/max_attend)
    
        # standard scaling of inputs
        if not os.path.isfile(os.path.join(root_path, "models", "neural_net", "standard_scaler.pkl")):
            object_ = StandardScaler()
        else:
            with open(os.path.join(root_path, "models", "neural_net", "standard_scaler.pkl"), 'rb') as f:
                object_ = pkl.load(f)
    
        liste = ['schedule_time', 'schedule_round', 'schedule_day', 'result', 'gf', 'ga', 'xg', 'xga', 'formation', 
         'referee', 'season_str', 'league_id', 'team_id', 'opponent_id', 'match_id', 'id', 'fbref_id', 
         'home_team_id', 'away_team_id', 'schedule_date', 'venue', 'captain',]

        cols_to_scale = list(set(list(self.data_frame.columns)).difference(liste))
        object_ = StandardScaler()
        self.data_frame.loc[:,cols_to_scale] = object_.fit_transform(self.data_frame.loc[:,cols_to_scale])
        self.save_object(object_, root_path, "standard_scaler")
        
        # one hot encoding of teams
        if not os.path.isfile(os.path.join(root_path, "models", "neural_net", "ohe_team.pkl")):
            ohe_team = OneHotEncoder(handle_unknown = "ignore")
        else:
            with open(os.path.join(root_path, "models", "neural_net", "ohe_team.pkl"), 'rb') as f:
                ohe_team = pkl.load(f)
      
            
        ohe_team = OneHotEncoder()
        
        to_ohe_team = self.data_frame.loc[:, ["team_id", "opponent_id"]]
        ohe_team.fit(to_ohe_team)
        self.save_object(ohe_team, root_path, "ohe_team")
                         
        codes = ohe_team.transform(to_ohe_team).toarray()
        feature_names = ohe_team.get_feature_names_out(['team_id', 'opponent_id'])
        

        self.data_frame = pd.concat([self.data_frame, pd.DataFrame(codes, columns = feature_names).astype(int)], axis=1)
        
        # one hot encoding of leagues
        if not os.path.isfile(os.path.join(root_path, "models", "neural_net", "onehot_ligue.pkl")):
            ohe_ligue = OneHotEncoder(max_categories = max(self.data_frame.season_str)+1) 
        else:
            with open(os.path.join(root_path, "models", "neural_net", "onehot_ligue.pkl"), 'rb') as f:
                ohe_ligue = pkl.load(f)   
                
        to_ohe_ligue = self.data_frame.loc[:,["league_id"]]
        ohe_ligue.fit(to_ohe_ligue)
        self.save_object(ohe_ligue, root_path, "onehot_ligue")
        
        codes = ohe_ligue.transform(to_ohe_ligue).toarray()
        feature_names = ohe_ligue.get_feature_names_out(['league_id'])

        self.data_frame = pd.concat([self.data_frame, pd.DataFrame(codes, columns = feature_names).astype(int)], axis=1)
        
        

    def data_frame(self):
        """ Returns the preprocessed dataframe"""
        
        return self.data_frame
        
    
    def return_dicts(self, dict_name):
        """ 
        Returns the dictionaries used for variable encoding.
        Input: String ('day', 'result', 'captain', 'formation', 'referee', 'season', 'venue)
        Output: Dictionary
            """
        
        dicts = {"day": self.day_dict,
                 "result": self.result,
                 "captain": self.capt,
                 "formation": self.formation,
                 "referee": self.referee,
                 "season": self.season,
                 "venue": self.venue_dict,
                 "A3ex&&7cFD": "This dictionary does not exist. Chose from 'day', 'result', 'captain', 'formation', 'referee', 'season' or 'venue'"}
        if dict_name not in dicts.keys():
            dict_name = "A3ex&&7cFD" 
        return dicts[dict_name]
    
    
    def save_object(self, obj, root_path, save_name):
        """
        Saves obj to a pickle file in directory path.
        """ 
        with open(os.path.join(root_path, "models", "neural_net", f"{save_name}.pkl"), 'wb') as f:
            pkl.dump(obj, f)

    def load_data_prep_object(self, obj_name):
        """
        Loads and returns the specified data prep object from the data_prep_objects_path.
        """
            # check if file exists
        if not os.path.isfile(os.path.join(f"{obj_name}.pkl")):
            raise ValueError(f"Data prep object file '{obj_name}.pkl' does not exist.")
        else:
            with open(os.path.join(f"{obj_name}.pkl"), 'rb') as f:
                obj = pkl.load(f)
                return obj

    def do_pca(self, df, perc_var, col_name1 = "shooting_standard_gls", col_name2 = 'misc_aerialduels_won_perc'):
        """
        Performs PCA on matchstats columns.
        """#
        try:
            pcs_matchstat = load_data_prep_object("./pca_matchstats")
        except:
            x = df.loc[:,col_name1:col_name2].fillna(0)
            pca_matchstat = PCA(n_components = perc_var)
            pcs_matchstat = pca_matchstat.fit_transform(x)
            save_object(pcs_matchstat, "pca_matchstats")
        principal_ms_df = pd.DataFrame(data = pcs_matchstat, columns = [f"feature_{p}" for p in range(pcs_matchstat.shape[1])])
        num_pcs = principal_ms_df.shape[1]
        print(num_pcs)
        columns_to_overwrite = list(df.loc[:, col_name1:col_name2].columns)
        df = df.drop(labels = columns_to_overwrite, axis = "columns")
        new_cols = columns_to_overwrite[:num_pcs-1] + columns_to_overwrite[-1:]
        #print(len(new_cols))
        df.loc[:, new_cols] = principal_ms_df.values
        return df
    
#################################################################################################################################
# Games dictionary

def game_dict(scale_df):
    """ Creates dictionaries containing every game in the (first four) training seasons and the (last) validation/test season."""
    
    # 1. iterate over first four seasons
    # 2. iterate over every week
    # 3. create dictionary for every week 
    # 4. create dataframe for specific week 
    # 5. iterate over new dataframe and add single game to dictionary
    # 6. return dict
    
    games_train = {}
    for seas in scale_df.season_str.unique()[:-1]:  
        for week in sorted(scale_df.schedule_round.unique()): 
            if week not in games_train.keys():
                games_train[week] = []  
            df = scale_df[(scale_df.season_str == seas) & (scale_df.schedule_round == week)] 
            for game in df.match_id:
                games_train[week].append(game)

    games_test = {}
    seas = scale_df.season_str.unique()[-1]
    for week in sorted(scale_df.schedule_round.unique()):
        if week not in games_test.keys():
            games_test[week] = []
        df = scale_df[(scale_df.season_str == seas) & (scale_df.schedule_round == week)]
        for game in df.match_id:
            games_test[week].append(game)
                
    return games_train, games_test



#################################################################################################################################
# Input tensors for LSTM

def inputs(games, clubs, rearrange_list, scale_df):
    """
    Creates input tensors and targets to feed to the LSTM
    Inputs: games, dictionary containing every game
            clubs, containing a dataframe for every club with all its games
            rearrange_list, a list of column names to reorder based on ones needs
            scale_df, the dataframe containing all games
    Output: list of lists  (1. all data to games club X played in current season till day Z 
                            2. all data to games club Y played in current season till day Z 
                            3. all results for club X in current season (result, goals scored, goals conceded, goal difference
                            4. all results for club Y in current season (result, goals scored, goals conceded, goal difference
    """
    
    
    lstm_inputs = [[], [], [], [], []] 
    
    
    for i in range(7,len(games)):  # start of prediction in game 7
        print(i)
        
        for game in games[i]:
            ## Team 1
            # get ids of participating teams in game 
            team1 = scale_df[scale_df.match_id == game].iloc[0].team_id
            team2 = scale_df[scale_df.match_id == game].iloc[1].team_id
            
            # get dataframe containing all games of club and reorder according to our need
            df_team1 = clubs[team1]
            df_team1 = df_team1[rearrange_list]

            # get season in which game took place
            season = df_team1[df_team1.match_id == game].season_str.values[0]  
            
            # create dataframe containing all games of season prior to the game we want to predict
            df_team1_past = df_team1.loc[ \
                df_team1[df_team1.season_str == season].iloc[0].name: # index of first game in seasons
                df_team1[df_team1.match_id == game].index.values[0] - 1,  # index of game we want to predict
                :]  # all columns
            #print(df_team1_past.shape)
            #print(df_team1[df_team1.match_id == game].index.values[0])
            ## Team 2
            # get dataframe containing all games of club and reorder according to our need
            df_team2 = clubs[team2]
            df_team2 = df_team2[rearrange_list]

            df_team2_past = df_team2.loc[
                df_team2[df_team2.season_str == season].iloc[0].name:  # index of first game in seasons
                df_team2[df_team2.match_id == game].index.values[0] - 1,  # index of game we want to predict
                :]  # all columns

            # create np array with zero to store data
            np_team1 = np.zeros([len(games), df_team1_past.loc[:,"xg":"schedule_round"].shape[1]])
            #print("np team 1", np_team1.shape)
            np_team2 = np.zeros([len(games), df_team2_past.loc[:,"xg":"schedule_round"].shape[1]])


            # insert data into array (back to front) such that all input into the lstm has the same length (padding)
            np_team1[- len(df_team1_past):, :-df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]] = df_team1_past.loc[:,"xg": "mean_points"]
            np_team2[- len(df_team2_past):, :-df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]] = df_team2_past.loc[:,"xg": "mean_points"]


            np_team1[- len(df_team1_past):, -df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]:] = df_team1.loc[df_team1[df_team1.season_str == season].iloc[1].name:df_team1[df_team1.match_id == game].index[0],"weekly_wages_eur":"schedule_round"]  
            np_team2[- len(df_team2_past):, -df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]:] = df_team2.loc[df_team2[df_team2.season_str == season].iloc[1].name:df_team2[df_team2.match_id == game].index[0],"weekly_wages_eur":"schedule_round"]  

            results1 = np.zeros((len(games), 4))
            results2 = np.zeros((len(games), 4))
            
            res1 = len(df_team1.loc[df_team1[df_team1.season_str == season].iloc[1].name:df_team1[df_team1.match_id == game].index[0],"result"])
            res2 = len(df_team2.loc[df_team2[df_team2.season_str == season].iloc[1].name:df_team2[df_team2.match_id == game].index[0],"result"])
            results1[-res1:, 0] = df_team1.loc[df_team1[df_team1.season_str == season].iloc[1].name:df_team1[df_team1.match_id == game].index[0],"result"]
            results1[-res1:, 1] = df_team1.loc[df_team1[df_team1.season_str == season].iloc[1].name:df_team1[df_team1.match_id == game].index[0],"gf"]
            results1[-res1:, 2] = df_team1.loc[df_team1[df_team1.season_str == season].iloc[1].name:df_team1[df_team1.match_id == game].index[0],"ga"]
            results1[-res1:, 3] = df_team1.loc[df_team1[df_team1.season_str == season].iloc[1].name:df_team1[df_team1.match_id == game].index[0],"goal_diff"]
            results2[-res2:, 0] = df_team2.loc[df_team2[df_team2.season_str == season].iloc[1].name:df_team2[df_team2.match_id == game].index[0],"result"]
            results2[-res2:, 1] = df_team2.loc[df_team2[df_team2.season_str == season].iloc[1].name:df_team2[df_team2.match_id == game].index[0],"gf"]
            results2[-res2:, 2] = df_team2.loc[df_team2[df_team2.season_str == season].iloc[1].name:df_team2[df_team2.match_id == game].index[0],"ga"]
            results2[-res2:, 3] = df_team2.loc[df_team2[df_team2.season_str == season].iloc[1].name:df_team2[df_team2.match_id == game].index[0],"goal_diff"]
            
            result1 = torch.nn.functional.one_hot(torch.tensor(results1[:,0]).long(), num_classes = 3)
            result2 = torch.nn.functional.one_hot(torch.tensor(results2[:,0]).long(), num_classes = 3)
            lstm_inputs[0].append(torch.tensor(np_team1))
            lstm_inputs[1].append(torch.tensor(np_team2))
            lstm_inputs[2].append(result1)
            lstm_inputs[3].append(result2)
            
    return lstm_inputs
        
def inputs_2seas(games, clubs, rearrange_list, scale_df):
    """
    Creates input tensors and targets to feed to the LSTM
    Inputs: games, dictionary containing every game
            clubs, containing a dataframe for every club with all its games
            rearrange_list, a list of column names to reorder based on ones needs
            scale_df, the dataframe containing all games
    Output: list of lists  (1. all data to games club X played in current season till day Z 
                            2. all data to games club Y played in current season till day Z 
                            3. all results for club X in current season (result, goals scored, goals conceded, goal difference
                            4. all results for club Y in current season (result, goals scored, goals conceded, goal difference
    """
    
    lstm_inputs = [[], [], [], [], []] 
    
    
    for i in range(7,len(games)):  # start of prediction in game 7
        print(i)
        
        for game in games[i]:
            
            if game in scale_df[scale_df.season_str == 0].match_id.unique():
                pass # skip first season as we want at least one season prior
            
            else:
                season = scale_df[scale_df.match_id == game].season_str.iloc[0] # get season in which game took place
                
                ## Team 1
                # get ids of participating teams in game 
                team1 = scale_df[scale_df.match_id == game].iloc[0].team_id
                team2 = scale_df[scale_df.match_id == game].iloc[1].team_id
                
                if (team1 in scale_df[scale_df.season_str == season - 1].team_id.unique()) and (team2 in scale_df[scale_df.season_str == season - 1].team_id.unique()): # if both teams played in last season
                    
                    # get dataframe containing all games of club and reorder according to our need
                    df_team1 = clubs[team1]
                    df_team1 = df_team1[rearrange_list]

                    
                    # create dataframe containing all games of season prior to the game we want to predict
                    df_team1_past = df_team1.loc[ \
                        df_team1[df_team1.season_str == season - 1].iloc[0].name: # index of first game in seasons
                        df_team1[df_team1.match_id == game].index.values[0] - 1,  # index of game priot to the one we want to predict
                        :]  # all columns

                    ## Team 2
                    # get dataframe containing all games of club and reorder according to our need
                    df_team2 = clubs[team2]
                    df_team2 = df_team2[rearrange_list]
                    df_team2_past = df_team2.loc[
                        df_team2[df_team2.season_str == season - 1].iloc[0].name:  # index of first game in seasons
                        df_team2[df_team2.match_id == game].index.values[0] - 1,  # index of game priot to the one we want to predict
                        :]  # all columns

                    # create np array with zero to store data
                    np_team1 = np.zeros([len(games)*2, df_team1_past.loc[:,"xg":"schedule_round"].shape[1]])
                    np_team2 = np.zeros([len(games)*2, df_team2_past.loc[:,"xg":"schedule_round"].shape[1]])


                    # insert data into array (back to front) such that all input into the lstm has the same length (padding)
                    np_team1[- len(df_team1_past):, :-df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]] = df_team1_past.loc[:,"xg": "mean_points"]
                    np_team2[- len(df_team2_past):, :-df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]] = df_team2_past.loc[:,"xg": "mean_points"]


                    # if last game of the season, no do not add as no future results to predict
                    if (df_team1.iloc[-1].name == df_team1_past.iloc[-1].name) or (df_team2.iloc[-1].name == df_team2_past.iloc[-1].name):
                        pass

                    # if not last game, add data from last game to input array
                    else:
                        # 
                        np_team1[- len(df_team1_past):, -df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]:] = df_team1.loc[df_team1[df_team1.season_str == season - 1].iloc[1].name:df_team1[df_team1.match_id == game].index.values[0],"weekly_wages_eur":"schedule_round"]  
                        np_team2[- len(df_team2_past):, -df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]:] = df_team2.loc[df_team2[df_team2.season_str == season - 1].iloc[1].name:df_team2[df_team2.match_id == game].index.values[0],"weekly_wages_eur":"schedule_round"]  

                        results1 = np.zeros((len(games) * 2, 4))
                        results2 = np.zeros((len(games) * 2, 4))
                        #print(result1)
                        res1 = len(df_team1.loc[df_team1[df_team1.season_str == season - 1].iloc[1].name:df_team1[df_team1.match_id == game].index.values[0],"result"])
                        res2 = len(df_team2.loc[df_team2[df_team2.season_str == season - 1].iloc[1].name:df_team2[df_team2.match_id == game].index.values[0],"result"])
                        results1[-res1:, 0] = df_team1.loc[df_team1[df_team1.season_str == season - 1].iloc[1].name:df_team1[df_team1.match_id == game].index.values[0],"result"]
                        results1[-res1:, 1] = df_team1.loc[df_team1[df_team1.season_str == season - 1].iloc[1].name:df_team1[df_team1.match_id == game].index.values[0],"gf"]
                        results1[-res1:, 2] = df_team1.loc[df_team1[df_team1.season_str == season - 1].iloc[1].name:df_team1[df_team1.match_id == game].index.values[0],"ga"]
                        results1[-res1:, 3] = df_team1.loc[df_team1[df_team1.season_str == season - 1].iloc[1].name:df_team1[df_team1.match_id == game].index.values[0],"goal_diff"]
                        results2[-res2:, 0] = df_team2.loc[df_team2[df_team2.season_str == season - 1].iloc[1].name:df_team2[df_team2.match_id == game].index.values[0],"result"]
                        results2[-res2:, 1] = df_team2.loc[df_team2[df_team2.season_str == season - 1].iloc[1].name:df_team2[df_team2.match_id == game].index.values[0],"gf"]
                        results2[-res2:, 2] = df_team2.loc[df_team2[df_team2.season_str == season - 1].iloc[1].name:df_team2[df_team2.match_id == game].index.values[0],"ga"]
                        results2[-res2:, 3] = df_team2.loc[df_team2[df_team2.season_str == season - 1].iloc[1].name:df_team2[df_team2.match_id == game].index.values[0],"goal_diff"]
                        
                        
                        result1 = torch.nn.functional.one_hot(torch.tensor(results1[:,0]).long(), num_classes = 3)
                        result2 = torch.nn.functional.one_hot(torch.tensor(results2[:,0]).long(), num_classes = 3)
                        lstm_inputs[0].append(torch.tensor(np_team1))
                        lstm_inputs[1].append(torch.tensor(np_team2))
                        lstm_inputs[2].append(result1)
                        lstm_inputs[3].append(result2)
                        
                        
                elif (team1 in scale_df[scale_df.season_str == season - 1].team_id.unique()) and (team2 not in scale_df[scale_df.season_str == season - 1].team_id.unique()):
                     # get dataframe containing all games of club and reorder according to our need
                    df_team1 = clubs[team1]
                    df_team1 = df_team1[rearrange_list]

                    # get season in which game took place
                    season = df_team1[df_team1.match_id == game].season_str.values[0]  

                    # create dataframe containing all games of season prior to the game we want to predict
                    df_team1_past = df_team1.loc[ \
                        df_team1[df_team1.season_str == season - 1].iloc[0].name: # index of first game in seasons
                        df_team1[df_team1.match_id == game].index.values[0] - 1,  # index of game we want to predict
                        :]  # all columns

                    ## Team 2
                    # get dataframe containing all games of club and reorder according to our need
                    df_team2 = clubs[team2]
                    df_team2 = df_team2[rearrange_list]

                    df_team2_past = df_team2.loc[
                        df_team2[df_team2.season_str == season].iloc[0].name:  # index of first game in seasons
                        df_team2[df_team2.match_id == game].index.values[0] - 1,  # index of game we want to predict
                        :]  # all columns

                    # create np array with zero to store data
                    np_team1 = np.zeros([len(games)*2, df_team1_past.loc[:,"xg":"schedule_round"].shape[1]])
                    np_team2 = np.zeros([len(games)*2, df_team2_past.loc[:,"xg":"schedule_round"].shape[1]])


                    # insert data into array (back to front) such that all input into the lstm has the same length (padding)
                    np_team1[- len(df_team1_past):, :-df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]] = df_team1_past.loc[:,"xg": "mean_points"]
                    np_team2[- len(df_team2_past):, :-df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]] = df_team2_past.loc[:,"xg": "mean_points"]


                    # if last game of the season, no do not add as no future results to predict
                    if (df_team1.iloc[-1].name == df_team1_past.iloc[-1].name) or (df_team2.iloc[-1].name == df_team2_past.iloc[-1].name):
                        pass


                    # if not last game, add data from last game to input array
                    else:
                        # 
                        np_team1[- len(df_team1_past):, -df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]:] = df_team1.loc[df_team1[df_team1.season_str == season - 1].iloc[1].name:df_team1[df_team1.match_id == game].index.values[0],"weekly_wages_eur":"schedule_round"]  
                        np_team2[- len(df_team2_past):, -df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]:] = df_team2.loc[df_team2[df_team2.season_str == season].iloc[1].name:df_team2[df_team2.match_id == game].index.values[0],"weekly_wages_eur":"schedule_round"]  

                        results1 = np.zeros((len(games) * 2, 4))
                        results2 = np.zeros((len(games) * 2, 4))
                        
                        res1 = len(df_team1.loc[df_team1[df_team1.season_str == season - 1].iloc[1].name:df_team1[df_team1.match_id == game].index.values[0],"result"])
                        res2 = len(df_team2.loc[df_team2[df_team2.season_str == season].iloc[1].name:df_team2[df_team2.match_id == game].index.values[0],"result"])
                        results1[-res1:, 0] = df_team1.loc[df_team1[df_team1.season_str == season - 1].iloc[1].name:df_team1[df_team1.match_id == game].index.values[0],"result"]
                        results1[-res1:, 1] = df_team1.loc[df_team1[df_team1.season_str == season - 1].iloc[1].name:df_team1[df_team1.match_id == game].index.values[0],"gf"]
                        results1[-res1:, 2] = df_team1.loc[df_team1[df_team1.season_str == season - 1].iloc[1].name:df_team1[df_team1.match_id == game].index.values[0],"ga"]
                        results1[-res1:, 3] = df_team1.loc[df_team1[df_team1.season_str == season - 1].iloc[1].name:df_team1[df_team1.match_id == game].index.values[0],"goal_diff"]
                        results2[-res2:, 0] = df_team2.loc[df_team2[df_team2.season_str == season].iloc[1].name:df_team2[df_team2.match_id == game].index.values[0],"result"]
                        results2[-res2:, 1] = df_team2.loc[df_team2[df_team2.season_str == season].iloc[1].name:df_team2[df_team2.match_id == game].index.values[0],"gf"]
                        results2[-res2:, 2] = df_team2.loc[df_team2[df_team2.season_str == season].iloc[1].name:df_team2[df_team2.match_id == game].index.values[0],"ga"]
                        results2[-res2:, 3] = df_team2.loc[df_team2[df_team2.season_str == season].iloc[1].name:df_team2[df_team2.match_id == game].index.values[0],"goal_diff"]
                        
                        
                        result1 = torch.nn.functional.one_hot(torch.tensor(results1[:,0]).long(), num_classes = 3)
                        result2 = torch.nn.functional.one_hot(torch.tensor(results2[:,0]).long(), num_classes = 3)
                        lstm_inputs[0].append(torch.tensor(np_team1))
                        lstm_inputs[1].append(torch.tensor(np_team2))
                        lstm_inputs[2].append(result1)
                        lstm_inputs[3].append(result2)
                        
                        
                elif (team1 not in scale_df[scale_df.season_str == season - 1].team_id.unique()) and (team2 in scale_df[scale_df.season_str == season - 1].team_id.unique()): # if team 1 did not play last season
                    
                     # get dataframe containing all games of club and reorder according to our need
                    df_team1 = clubs[team1]
                    df_team1 = df_team1[rearrange_list]

                    # get season in which game took place
                    season = df_team1[df_team1.match_id == game].season_str.values[0]  

                    # create dataframe containing all games of season prior to the game we want to predict
                    df_team1_past = df_team1.loc[ \
                        df_team1[df_team1.season_str == season].iloc[0].name: # index of first game in seasons
                        df_team1[df_team1.match_id == game].index.values[0] - 1,  # index of game prior to the one we want to predict
                        :]  # all columns

                    ## Team 2
                    # get dataframe containing all games of club and reorder according to our need
                    df_team2 = clubs[team2]
                    df_team2 = df_team2[rearrange_list]

                    df_team2_past = df_team2.loc[
                        df_team2[df_team2.season_str == season - 1].iloc[0].name:  # index of first game in last seasons
                        df_team2[df_team2.match_id == game].index.values[0] - 1,  # index of game prior to the one we want to predict
                        :]  # all columns

                    # create np array with zero to store data
                    np_team1 = np.zeros([len(games)*2, df_team1_past.loc[:,"xg":"schedule_round"].shape[1]])
                    np_team2 = np.zeros([len(games)*2, df_team2_past.loc[:,"xg":"schedule_round"].shape[1]])


                    # insert data into array (back to front) such that all input into the lstm has the same length (padding)
                    np_team1[- len(df_team1_past):, :-df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]] = df_team1_past.loc[:,"xg": "mean_points"]
                    np_team2[- len(df_team2_past):, :-df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]] = df_team2_past.loc[:,"xg": "mean_points"]


                    # if last game of the season, no do not add as no future results to predict
                    if (df_team1.iloc[-1].name == df_team1_past.iloc[-1].name) or (df_team2.iloc[-1].name == df_team2_past.iloc[-1].name):
                        pass

                    # if not last game, add data from last game to input array
                    else:
                        # 
                        np_team1[- len(df_team1_past):, -df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]:] = df_team1.loc[df_team1[df_team1.season_str == season].iloc[1].name:df_team1[df_team1.match_id == game].index.values[0],"weekly_wages_eur":"schedule_round"]  
                        np_team2[- len(df_team2_past):, -df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]:] = df_team2.loc[df_team2[df_team2.season_str == season - 1].iloc[1].name:df_team2[df_team2.match_id == game].index.values[0],"weekly_wages_eur":"schedule_round"]  

                        results1 = np.zeros((len(games) * 2, 4))
                        results2 = np.zeros((len(games) * 2, 4))
                        
                        res1 = len(df_team1.loc[df_team1[df_team1.season_str == season].iloc[1].name:df_team1[df_team1.match_id == game].index.values[0],"result"])
                        res2 = len(df_team2.loc[df_team2[df_team2.season_str == season - 1].iloc[1].name:df_team2[df_team2.match_id == game].index.values[0],"result"])
                        results1[-res1:, 0] = df_team1.loc[df_team1[df_team1.season_str == season].iloc[1].name:df_team1[df_team1.match_id == game].index.values[0],"result"]
                        results1[-res1:, 1] = df_team1.loc[df_team1[df_team1.season_str == season].iloc[1].name:df_team1[df_team1.match_id == game].index.values[0],"gf"]
                        results1[-res1:, 2] = df_team1.loc[df_team1[df_team1.season_str == season].iloc[1].name:df_team1[df_team1.match_id == game].index.values[0],"ga"]
                        results1[-res1:, 3] = df_team1.loc[df_team1[df_team1.season_str == season].iloc[1].name:df_team1[df_team1.match_id == game].index.values[0],"goal_diff"]
                        results2[-res2:, 0] = df_team2.loc[df_team2[df_team2.season_str == season - 1].iloc[1].name:df_team2[df_team2.match_id == game].index.values[0],"result"]
                        results2[-res2:, 1] = df_team2.loc[df_team2[df_team2.season_str == season - 1].iloc[1].name:df_team2[df_team2.match_id == game].index.values[0],"gf"]
                        results2[-res2:, 2] = df_team2.loc[df_team2[df_team2.season_str == season - 1].iloc[1].name:df_team2[df_team2.match_id == game].index.values[0],"ga"]
                        results2[-res2:, 3] = df_team2.loc[df_team2[df_team2.season_str == season - 1].iloc[1].name:df_team2[df_team2.match_id == game].index.values[0],"goal_diff"]
                       
                        result1 = torch.nn.functional.one_hot(torch.tensor(results1[:,0]).long(), num_classes = 3)
                        result2 = torch.nn.functional.one_hot(torch.tensor(results2[:,0]).long(), num_classes = 3)
                        lstm_inputs[0].append(torch.tensor(np_team1))
                        lstm_inputs[1].append(torch.tensor(np_team2))
                        lstm_inputs[2].append(result1)
                        lstm_inputs[3].append(result2)
                        #lstm_inputs[4].append(fut_feature1)
                        #lstm_inputs[5].append(fut_feature2)
                        
                else:
                    pass
    return lstm_inputs



def two_team_inputs(team1, team2, rearrange_list, scale_df, clubs):
    """
    Creates evaluation input to LSTM for two arbitrary teams
    Inputs: team1, team id of home team as integer
            team2, team id of away team as integer
            rearrange_list, list with column names for flexibility with new variables
            scale_df, dataframe with all games
            clubs, dictionary containing a dataframe for every club with all its games
    Output: list of lists  (1. all data to games club X played in last seasons till day Z 
                            2. all data to games club Y played in last seasons till day Z 
                            3. all results for club X in current season (result, goals scored, goals conceded, goal difference
                            4. all results for club Y in current season (result, goals scored, goals conceded, goal difference
    """
    
    lstm_inputs = [[], [], [], [], []] 
    season = int(max(scale_df.season_str))  # most recent season to base prediction on newest data
    
    if team1 not in scale_df[scale_df.season_str == season].team_id.unique():
        raise KeyError("Incorrect input: Team 1 did not play in the last season")
    if team2 not in scale_df[scale_df.season_str == season].team_id.unique():
        raise KeyError("Incorrect input: Team 2 did not play in the last season")
                ## Team 1
                # get ids of participating teams in game 
   
    games = scale_df.schedule_round.unique()
    if (team1 in scale_df[scale_df.season_str == season - 1].team_id.unique()) and (team2 in scale_df[scale_df.season_str == season - 1].team_id.unique()): # if both teams played in last season
        # get dataframe containing all games of club and reorder according to our need
        df_team1 = clubs[team1]
        df_team1 = df_team1[rearrange_list]
        # get season in which game took place


        # create dataframe containing all games of season prior to the game we want to predict
        df_team1_past = df_team1.loc[ \
            df_team1[df_team1.season_str == season - 1].iloc[0].name: # index of first game in seasons
            df_team1.iloc[-2].name,  # index of game prior to the one we want to predict
            :]  # all columns
        
        ## Team 2
        # get dataframe containing all games of club and reorder according to our need
        df_team2 = clubs[team2]
        df_team2 = df_team2[rearrange_list]
        #print("team2", df_team2.shape)
        df_team2_past = df_team2.loc[
            df_team2[df_team2.season_str == season - 1].iloc[0].name:  # index of first game in seasons
            df_team2.iloc[-2].name,  # index of game prior to the one we want to predict
            :]  # all columns
        
        # create np array with zero to store data
        np_team1 = np.zeros([len(games)*2, df_team1_past.loc[:,"xg":"schedule_round"].shape[1]])
        np_team2 = np.zeros([len(games)*2, df_team2_past.loc[:,"xg":"schedule_round"].shape[1]])
        #print("np_team1",np_team1.shape)

        # insert data into array (back to front) such that all input into the lstm has the same length (padding)
        np_team1[- len(df_team1_past):, :-df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]] = df_team1_past.loc[:,"xg": "mean_points"]
        np_team2[- len(df_team2_past):, :-df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]] = df_team2_past.loc[:,"xg": "mean_points"]


        # if last game of the season, no do not add as no future results to predict
        if (df_team1.iloc[-1].name == df_team1_past.iloc[-1].name) or (df_team2.iloc[-1].name == df_team2_past.iloc[-1].name):
            print(df_team1.iloc[-1].name)
            print(df_team1_past.iloc[-1].name)
            print("hello")
            pass

        # if not last game, add data from last game to input array
        else:
            # 
            np_team1[- len(df_team1_past):, -df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]:] = df_team1.loc[df_team1[df_team1.season_str == season - 1].iloc[1].name:,"weekly_wages_eur":"schedule_round"]  
            np_team2[- len(df_team2_past):, -df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]:] = df_team2.loc[df_team2[df_team2.season_str == season - 1].iloc[1].name:,"weekly_wages_eur":"schedule_round"]  

            results1 = np.zeros((len(games) * 2, 4))
            results2 = np.zeros((len(games) * 2, 4))
            
            res1 = len(df_team1.loc[df_team1[df_team1.season_str == season - 1].iloc[1].name:,"result"])
            res2 = len(df_team2.loc[df_team2[df_team2.season_str == season - 1].iloc[1].name:,"result"])
            results1[-res1:, 0] = df_team1.loc[df_team1[df_team1.season_str == season - 1].iloc[1].name:,"result"]
            results1[-res1:, 1] = df_team1.loc[df_team1[df_team1.season_str == season - 1].iloc[1].name:,"gf"]
            results1[-res1:, 2] = df_team1.loc[df_team1[df_team1.season_str == season - 1].iloc[1].name:,"ga"]
            results1[-res1:, 3] = df_team1.loc[df_team1[df_team1.season_str == season - 1].iloc[1].name:,"goal_diff"]
            results2[-res2:, 0] = df_team2.loc[df_team2[df_team2.season_str == season - 1].iloc[1].name:,"result"]
            results2[-res2:, 1] = df_team2.loc[df_team2[df_team2.season_str == season - 1].iloc[1].name:,"gf"]
            results2[-res2:, 2] = df_team2.loc[df_team2[df_team2.season_str == season - 1].iloc[1].name:,"ga"]
            results2[-res2:, 3] = df_team2.loc[df_team2[df_team2.season_str == season - 1].iloc[1].name:,"goal_diff"]
            
            # one hot encode results for prediction
            result1 = torch.nn.functional.one_hot(torch.tensor(results1[:,0]).long(), num_classes = 3)
            result2 = torch.nn.functional.one_hot(torch.tensor(results2[:,0]).long(), num_classes = 3)
            lstm_inputs[0].append(torch.tensor(np_team1))
            lstm_inputs[1].append(torch.tensor(np_team2))
            lstm_inputs[2].append(result1)
            lstm_inputs[3].append(result2)

    elif (team1 in scale_df[scale_df.season_str == season - 1].team_id.unique()) and (team2 not in scale_df[scale_df.season_str == season - 1].team_id.unique()): # if team 2 did not play in last season
        
        # get dataframe containing all games of club and reorder according to our need
        df_team1 = clubs[team1]
        df_team1 = df_team1[rearrange_list]

        # create dataframe containing all games of season prior to the game we want to predict
        df_team1_past = df_team1.loc[ \
            df_team1[df_team1.season_str == season - 1].iloc[0].name: # index of first game in seasons
            df_team1.iloc[-2].name,  # index of game we want to predict
            :]  # all columns

        ## Team 2
        # get dataframe containing all games of club and reorder according to our need
        df_team2 = clubs[team2]
        df_team2 = df_team2[rearrange_list]

        df_team2_past = df_team2.loc[
            df_team2[df_team2.season_str == season].iloc[0].name:  # index of first game in seasons
            df_team2.iloc[-2].name,  # index of game we want to predict
            :]  # all columns

        # create np array with zero to store data
        np_team1 = np.zeros([len(games)*2, df_team1_past.loc[:,"xg":"schedule_round"].shape[1]])
        np_team2 = np.zeros([len(games)*2, df_team2_past.loc[:,"xg":"schedule_round"].shape[1]])


        # insert data into array (back to front) such that all input into the lstm has the same length (padding)
        np_team1[- len(df_team1_past):, :-df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]] = df_team1_past.loc[:,"xg": "mean_points"]
        np_team2[- len(df_team2_past):, :-df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]] = df_team2_past.loc[:,"xg": "mean_points"]


        # if last game of the season, no do not add as no future results to predict
        if (df_team1.iloc[-1].name == df_team1_past.iloc[-1].name) or (df_team2.iloc[-1].name == df_team2_past.iloc[-1].name):
            pass


        # if not last game, add data from last game to input array
        else:
            # 
            np_team1[- len(df_team1_past):, -df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]:] = df_team1.loc[df_team1[df_team1.season_str == season - 1].iloc[1].name:,"weekly_wages_eur":"schedule_round"]  
            np_team2[- len(df_team2_past):, -df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]:] = df_team2.loc[df_team2[df_team2.season_str == season].iloc[1].name:,"weekly_wages_eur":"schedule_round"]  

            results1 = np.zeros((len(games) * 2, 4))
            results2 = np.zeros((len(games) * 2, 4))
            
            res1 = len(df_team1.loc[df_team1[df_team1.season_str == season - 1].iloc[1].name:,"result"])
            res2 = len(df_team2.loc[df_team2[df_team2.season_str == season].iloc[1].name:,"result"])
            results1[-res1:, 0] = df_team1.loc[df_team1[df_team1.season_str == season - 1].iloc[1].name:,"result"]
            results1[-res1:, 1] = df_team1.loc[df_team1[df_team1.season_str == season - 1].iloc[1].name:,"gf"]
            results1[-res1:, 2] = df_team1.loc[df_team1[df_team1.season_str == season - 1].iloc[1].name:,"ga"]
            results1[-res1:, 3] = df_team1.loc[df_team1[df_team1.season_str == season - 1].iloc[1].name:,"goal_diff"]
            results2[-res2:, 0] = df_team2.loc[df_team2[df_team2.season_str == season].iloc[1].name:,"result"]
            results2[-res2:, 1] = df_team2.loc[df_team2[df_team2.season_str == season].iloc[1].name:,"gf"]
            results2[-res2:, 2] = df_team2.loc[df_team2[df_team2.season_str == season].iloc[1].name:,"ga"]
            results2[-res2:, 3] = df_team2.loc[df_team2[df_team2.season_str == season].iloc[1].name:,"goal_diff"]
            
            result1 = torch.nn.functional.one_hot(torch.tensor(results1[:,0]).long(), num_classes = 3)
            result2 = torch.nn.functional.one_hot(torch.tensor(results2[:,0]).long(), num_classes = 3)
            lstm_inputs[0].append(torch.tensor(np_team1))
            lstm_inputs[1].append(torch.tensor(np_team2))
            lstm_inputs[2].append(result1)
            lstm_inputs[3].append(result2)
            

    elif (team1 not in scale_df[scale_df.season_str == season - 1].team_id.unique()) and (team2 in scale_df[scale_df.season_str == season - 1].team_id.unique()): # if team 1 did not play in last season
       
        # get dataframe containing all games of club and reorder according to our need
        df_team1 = clubs[team1]
        df_team1 = df_team1[rearrange_list]


        # create dataframe containing all games of season prior to the game we want to predict
        df_team1_past = df_team1.loc[ \
            df_team1[df_team1.season_str == season].iloc[0].name: # index of first game in seasons
            df_team1.iloc[-2].name,  # index of game we want to predict
            :]  # all columns

        ## Team 2
        # get dataframe containing all games of club and reorder according to our need
        df_team2 = clubs[team2]
        df_team2 = df_team2[rearrange_list]

        df_team2_past = df_team2.loc[
            df_team2[df_team2.season_str == season - 1].iloc[0].name:  # index of first game in seasons
            df_team2.iloc[-2].name,  # index of game we want to predict
            :]  # all columns

        # create np array with zero to store data
        np_team1 = np.zeros([len(games)*2, df_team1_past.loc[:,"xg":"schedule_round"].shape[1]])
        print(np_team1)
        np_team2 = np.zeros([len(games)*2, df_team2_past.loc[:,"xg":"schedule_round"].shape[1]])


        # insert data into array (back to front) such that all input into the lstm has the same length (padding)
        np_team1[- len(df_team1_past):, :-df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]] = df_team1_past.loc[:,"xg": "mean_points"]
        np_team2[- len(df_team2_past):, :-df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]] = df_team2_past.loc[:,"xg": "mean_points"]


        # if last game of the season, no do not add as no future results to predict
        if (df_team1.iloc[-1].name == df_team1_past.iloc[-1].name) or (df_team2.iloc[-1].name == df_team2_past.iloc[-1].name):
            pass

        # if not last game, add data from last game to input array
        else:
            # 
            np_team1[- len(df_team1_past):, -df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]:] = df_team1.loc[df_team1[df_team1.season_str == season].iloc[1].name:,"weekly_wages_eur":"schedule_round"]  
            np_team2[- len(df_team2_past):, -df_team1.loc[:, "weekly_wages_eur":"schedule_round"].shape[1]:] = df_team2.loc[df_team2[df_team2.season_str == season - 1].iloc[1].name:,"weekly_wages_eur":"schedule_round"]  

            results1 = np.zeros((len(games) * 2, 4))
            results2 = np.zeros((len(games) * 2, 4))
            
            res1 = len(df_team1.loc[df_team1[df_team1.season_str == season].iloc[1].name:,"result"])
            res2 = len(df_team2.loc[df_team2[df_team2.season_str == season - 1].iloc[1].name:,"result"])
            results1[-res1:, 0] = df_team1.loc[df_team1[df_team1.season_str == season].iloc[1].name:,"result"]
            results1[-res1:, 1] = df_team1.loc[df_team1[df_team1.season_str == season].iloc[1].name:,"gf"]
            results1[-res1:, 2] = df_team1.loc[df_team1[df_team1.season_str == season].iloc[1].name:,"ga"]
            results1[-res1:, 3] = df_team1.loc[df_team1[df_team1.season_str == season].iloc[1].name:,"goal_diff"]
            results2[-res2:, 0] = df_team2.loc[df_team2[df_team2.season_str == season - 1].iloc[1].name:,"result"]
            results2[-res2:, 1] = df_team2.loc[df_team2[df_team2.season_str == season - 1].iloc[1].name:,"gf"]
            results2[-res2:, 2] = df_team2.loc[df_team2[df_team2.season_str == season - 1].iloc[1].name:,"ga"]
            results2[-res2:, 3] = df_team2.loc[df_team2[df_team2.season_str == season - 1].iloc[1].name:,"goal_diff"]
            
            result1 = torch.nn.functional.one_hot(torch.tensor(results1[:,0]).long(), num_classes = 3)
            result2 = torch.nn.functional.one_hot(torch.tensor(results2[:,0]).long(), num_classes = 3)
            lstm_inputs[0].append(torch.tensor(np_team1))
            lstm_inputs[1].append(torch.tensor(np_team2))
            lstm_inputs[2].append(result1)
            lstm_inputs[3].append(result2)
            #lstm_inputs[4].append(fut_feature1)
            #lstm_inputs[5].append(fut_feature2)
    else:
        raise KeyError("One or both team/s did not play in last season")
    return lstm_inputs
#################################################################################################################################
# Custom dataset

class data_to_lstm():
    """
    Dataset class to feed data via dataloader to LSTM
    """   
    def __init__(self, mylist):
        self.output = []
        self.team_home = mylist[0]
        self.team_away = mylist[1]
        self.result1 = mylist[2]
        self.result2 = mylist[3]
        
        
        
        for i in range(len(self.team_home)):
            self.output.append(((self.team_home[i], self.team_away[i]), (self.result1[i], self.result2[i])))
        

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        return self.output[idx]

    
#################################################################################################################################
# LSTM
class Sport_pred_2LSTM_1(torch.nn.Module):
    """
    LSTM class for sequential data
    """
    def __init__(self,n_features, hidden, num_classes, num_layers = 1):
        super(Sport_pred_2LSTM_1, self).__init__()
        self.n_features = n_features 
        self.num_classes = num_classes # number of classes (win, draw, lose)
        self.n_hidden = hidden # number of hidden states
        self.n_layers = num_layers # number of LSTM layers (stacked)
        
        # two separate lstms to account for every teams history
        self.l_lstm1 = torch.nn.LSTM(input_size = n_features, 
                             hidden_size = self.n_hidden,
                             num_layers = self.n_layers,
                             batch_first = True)
        

        # linear layer to process outcomes
        self.l_linear1 = torch.nn.Linear(self.n_hidden, self.num_classes)
           
        
        self.soft = torch.nn.Softmax()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()


    def forward(self, x):
        # convert input to fit the model
        x = x.to(torch.float32)
        x = torch.nan_to_num(x, nan = 0.0)
        
        
        # run data through lstm and yield output
        lstm_out1,(ht1, ct1) = self.l_lstm1(x)#,(h01, c01))
        ht1 = ht1.squeeze()
        ct1 = ct1.squeeze()
        
        out = lstm_out1#
        
        # run lstm output through linear layer to predict outcome
        result = self.l_linear1(out)
       
        return result
    
# GRU instead of LSTM    
class Sport_pred_2GRU_1(torch.nn.Module):
    """
    GRU class for sequential data
    """
    def __init__(self,n_features, hidden, num_classes, num_layers = 1):
        super(Sport_pred_2GRU_1, self).__init__()
        self.n_features = n_features 
        self.num_classes = num_classes # number of classes (win, draw, lose)
        self.n_hidden = hidden # number of hidden states
        self.n_layers = num_layers # number of LSTM layers (stacked)
        
        # two separate lstms to account for every teams history
        self.l_lstm1 = torch.nn.GRU(input_size = n_features, 
                             hidden_size = self.n_hidden,
                             num_layers = self.n_layers,
                             batch_first = True)
        

        # linear layer to process outcomes
        self.l_linear1 = torch.nn.Linear(self.n_hidden, self.num_classes)
        
        
        self.soft = torch.nn.Softmax()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()


    def forward(self, x):
        # convert input to fit the model
        x = x.to(torch.float32)
        x = torch.nan_to_num(x, nan = 0.0)
        
        
        # run data through lstm and yield output
        lstm_out1,(ht1, ct1) = self.l_lstm1(x)
  
        out = lstm_out1
        
        # run lstm output through linear layer to predict outcome
        result = self.l_linear1(out)
        
        return result

   #################################################################################################################################
# Dictionary with dataframes of all clubs
def club_dict(scale_df):
    """
    Creates a dictionary that contains a dataframe with every games for every club
    Input:  scale_df, dataframe containing all games
    Output: dictionary containing all clubs and their games
    """
    clubs = {}
    for club in list(scale_df.team_id.unique()):
        club_df = scale_df[scale_df.team_id == club]
        club_df.sort_values(by = ['season_str', 'schedule_round'], inplace = True)
        club_df.reset_index(drop = True, inplace = True)
        clubs[club] = club_df
    return clubs



#################################################################################################################################
# Points and last games own and opponent
def points_and_co(clubs, result_dict):
    """
    Function to add points, mean points and last results against every opponent.
    Input:  scale_df, dataframe containing all games
            result_dict, dictionary containing the nominal encoding of the results
    Output: Added three variables to every dataframe:   points,  the points a club collected till respective game
                                                        mean_points, number of points the club collected on average in current season
                                                        last_results, resutlts of team against oppontent in previous games
                                                        goal_diff, the goal difference in game
    """
    
    # load standard scaler
    if not os.path.isfile(os.path.join("standard_scaler.pkl")):
            object_ = StandardScaler()
    else:
        with open(os.path.join("standard_scaler.pkl"), 'rb') as f:
            object_ = pkl.load(f)
            
    point_dict = {result_dict['W']: 3, result_dict['L']: 0, result_dict['D']: 1}  # dict to decode w/d/l nominal encoding to their respective points
    for club in clubs:
        new_column_points = []
        new_column_meanpoints = []
        last_year_game_res = {}
        last_year_game = []
        goal_diff_seasons = []
        z = 0
        for season in clubs[club].groupby("season_str"):
            season = season[1]
            new_col_seas_p = [0] * len(season)
            new_col_seas_mp = [0] * len(season)
            new_col_last_year_game = [0] * len(season)
            index = 0
            #print(season)
            last_year_game_res[z] = {}
            for _, matchday in season.iterrows():
                goal_diff_seasons.append(int(matchday.gf) - int(matchday.ga))
                if index > 0: # if not first game in season
                    new_col_seas_p[index] = new_col_seas_p[index - 1] + point_dict[matchday.result] # compute points
                    new_col_seas_mp[index] = (new_col_seas_mp[index - 1] + point_dict[matchday.result])/2 # compute mean points
                else: # if first game in season
                    new_col_seas_p[index] = point_dict[matchday.result] # compute points from first game
                    new_col_seas_mp[index] = point_dict[matchday.result] # compute points from first game

                if matchday.opponent_id not in last_year_game_res[z].keys(): # if no prior games against opponent
                    last_year_game_res[z][matchday.opponent_id] = [] # create dict entry 
                    last_year_game_res[z][matchday.opponent_id].append(matchday.result) # add result
                else: # if prior games exist
                    last_year_game_res[z][matchday.opponent_id].append(matchday.result)  # add results

                if z > 0: # if not first season
                    if (index > len(season)/2) & (matchday.opponent_id in last_year_game_res[z - 1].keys()):
                        new_col_last_year_game[index] = np.mean(last_year_game_res[z - 1][matchday.opponent_id] + last_year_game_res[z][matchday.opponent_id]) # if more than half season played ie there are already results for curr season take mean of last three games
                    elif (index <= len(season)/2) & (matchday.opponent_id in last_year_game_res[z - 1].keys()): # if first half of season
                        new_col_last_year_game[index] = np.mean(last_year_game_res[z - 1][matchday.opponent_id]) # take mean of last games
                    else:
                        pass
                index += 1
            new_column_points.append(new_col_seas_p)
            new_column_meanpoints.append(new_col_seas_mp)
            
            last_year_game.append(new_col_last_year_game)
            z += 1
            
        # create single list from list of lists and add to dataframe
        new_column_points = list(itertools.chain.from_iterable(new_column_points))
        new_column_meanpoints = list(itertools.chain.from_iterable(new_column_meanpoints))
        last_year_game = list(itertools.chain.from_iterable(last_year_game))
        clubs[club]["goal_diff"] = goal_diff_seasons
        clubs[club]["points"] = new_column_points
        clubs[club]["mean_points"] = new_column_meanpoints
        clubs[club]["last_results"] = last_year_game
        
        # standard scale new variables
        clubs[club].loc[:,['points', 'mean_points', "last_results"]] = object_.fit_transform(clubs[club].loc[:,['points', 'mean_points', "last_results"]])
    return clubs


def points_and_co_oppon(clubs, result_dict):
    """
    Function to also add opponents points, opponent mean points and opponent wages to the available data.
    Input:  scale_df, dataframe containing all games
            result_dict, dictionary containing the nominal encoding of the results
    Output: Added three variables to every dataframe:   oppon_points, number of points opponent collected in curr season
                                                        oppon_mean_points, average of points opponent collected in curr season
                                                        oppon_wages, weekly wages opponent pays
    """
    
    # load standard scaler
    if not os.path.isfile(os.path.join("standard_scaler.pkl")):
        object_ = StandardScaler()
    else:
        with open(os.path.join("standard_scaler.pkl"), 'rb') as f:
            object_ = pkl.load(f)
    point_dict = {result_dict['W']: 3, result_dict['L']: 0, result_dict['D']: 1} # dict to compute points collected
    for club in clubs:
        oppon_points = []
        oppon_meanpoints = []
        oppon_wages = []
        for season in clubs[club].groupby("season_str"):
            season = season[1]
            new_col_seas_oppon_p = [0] * len(season)
            new_col_seas_oppon_mp = [0] * len(season)
            new_col_seas_oppon_wg = [0] * len(season)
            index = 0
            #print(season)
            for _, matchday in season.iterrows():
                if len(clubs[matchday.opponent_id].loc[:clubs[matchday.opponent_id][clubs[matchday.opponent_id].match_id == matchday.match_id].index[0],]) > 1: # if oppontent club played more than one game prior to matchday
                    new_col_seas_oppon_p[index] = clubs[matchday.opponent_id].loc[:clubs[matchday.opponent_id][clubs[matchday.opponent_id].match_id == matchday.match_id].index[0],].iloc[-2,:].points  # get opponents points
                    new_col_seas_oppon_mp[index] = clubs[matchday.opponent_id].loc[:clubs[matchday.opponent_id][clubs[matchday.opponent_id].match_id == matchday.match_id].index[0],].iloc[-2,:].mean_points # get opponents mean 
                    if clubs[matchday.opponent_id].loc[:clubs[matchday.opponent_id][clubs[matchday.opponent_id].match_id == matchday.match_id].index[0],].iloc[-2,:].schedule_date == matchday.schedule_date: # sanity check
                        print("Alarm")
                else:
                    pass
                new_col_seas_oppon_wg[index] = clubs[matchday.opponent_id][clubs[matchday.opponent_id].match_id == matchday.match_id].weekly_wages_eur.values[0] # get weekly wages from last game
                index += 1
            oppon_points.append(new_col_seas_oppon_p)
            oppon_meanpoints.append(new_col_seas_oppon_mp)
            oppon_wages.append(new_col_seas_oppon_wg)
            
        # create single list from list of lists and add to dataframe
        oppon_points = list(itertools.chain.from_iterable(oppon_points))
        oppon_meanpoints = list(itertools.chain.from_iterable(oppon_meanpoints))
        oppon_wages = list(itertools.chain.from_iterable(oppon_wages))
        clubs[club]["oppon_points"] = oppon_points
        clubs[club]["oppon_mean_points"] = oppon_meanpoints
        clubs[club]["oppon_wages"] = oppon_wages
        
        # standard scale new variables
        clubs[club].loc[:,['oppon_points', 'oppon_mean_points']] = object_.fit_transform(clubs[club].loc[:,['oppon_points', 'oppon_mean_points']])
    return clubs



#################################################################################################################################

    
                                                                                                                                        
                                                                                                                                        
                                                                                                                                        
                                                                                                                                                                                                                                                                   
                                                                                                                                 