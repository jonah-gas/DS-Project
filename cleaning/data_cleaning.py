
import pandas as pd
import os
import sys

root_path = os.path.abspath(os.path.join('..')) # <- adjust such that root_path always points at the root project dir (i.e. if current file is two folders deep, use '../..'). 
if root_path not in sys.path:
    sys.path.append(root_path)

import database_server.db_utilities as dbu 

class DataCleaning:
    def __init__(self):
        # read in excel file with raw data cleaning information
        # (sheet_name=None will yield dict of sheet dfs: 'fbref_match', 'fbref_match_player')
        cleaning_info_path = os.path.join(root_path, 'cleaning', 'cleaning_info.xlsx')
        self.cleaning_info = pd.read_excel(cleaning_info_path, sheet_name=None)

    def clean_matchstats_for_db(self, df, conn=None):
        """Cleans the raw scraped data by dropping columns, removing rows with missing values, and renaming columns."""

        # rename columns
        df = self.clean_match_colnames(df)

        # pull cleaning info from cleaning_info df
        cols_to_drop = df.columns[self.cleaning_info['fbref_match']['drop_before_matchstats_insert']]
        cols_to_check_missing = df.columns[self.cleaning_info['fbref_match']['drop_row_if_missing_before_matchstats_insert']]

        # drop columns 
        df.drop(cols_to_drop, axis=1, inplace=True)

        ### data validation ###

        # drop exact duplicates (e.g. due to overlapping seasons in files in data archive folder)
        n_rows_before = df.shape[0]
        df.drop_duplicates(inplace=True)
        print(60*'*')
        print(f"matchstats cleaning: Dropped {n_rows_before - df.shape[0]} exact duplicates.")

        # drop rows with missing values in essential columns (ids, etc.)
        n_rows_before = df.shape[0]
        df.dropna(subset=cols_to_check_missing, inplace=True)
        print(60*'*')
        print(f"matchstats cleaning: Dropped {n_rows_before - df.shape[0]} rows due to missing values in columns {cols_to_check_missing}.")

        # drop rows with too many missing values
        n_rows_before = df.shape[0]
        thresh = 0.8
        df.dropna(thresh=thresh*df.shape[1], inplace=True)
        print(f"matchstats cleaning: Dropped {n_rows_before - df.shape[0]} rows due to missing values in more than {thresh*100}% of columns.")
        print(60*'*')

        # prepare names with apostrophes for SQL insert
        df['referee'] = df['referee'].str.replace("'", "''")
        df['captain'] = df['captain'].str.replace("'", "''")

        ### id matching ###

        # create matching dicts
        leagues_df = dbu.select_query("SELECT fbref_id, id FROM leagues;", conn=conn)
        leagues_dict = dict(zip(leagues_df['fbref_id'].astype(int), leagues_df['id']))
        teams_df = dbu.select_query("SELECT fbref_id, id FROM teams;", conn=conn)
        teams_dict = dict(zip(teams_df['fbref_id'], teams_df['id'].astype(int)))
        matches_df = dbu.select_query("SELECT fbref_id, id FROM matches;", conn=conn)
        matches_dict = dict(zip(matches_df['fbref_id'], matches_df['id'].astype(int)))

        # replace fbref ids with db ids
        df['league_id'] = df['league_id'].map(leagues_dict)
        df['team_id'] = df['team_id'].map(teams_dict)
        df['opponent_id'] = df['opponent_id'].map(teams_dict)
        df['match_id'] = df['match_id'].map(matches_dict)

        return df

    def get_teams_for_db(self, df, conn=None):
        """Isolates team names and ids from raw scraped data for db insert."""

        # get distinct (fbref-) team names and fbref league ids
        df = df.groupby('schedule_fbref_opponent_id').first().sort_values('schedule_fbref_league_id')[['schedule_Opponent', 'schedule_fbref_league_id']].reset_index()
        
        # get leagues table for country matching
        leagues_df = dbu.select_query("SELECT * FROM leagues;", conn=conn)
        
        # get matching dict
        leagues_dict = dict(zip(leagues_df['fbref_id'].astype(int), leagues_df['country']))
        
        # add country column
        df['country'] = df['schedule_fbref_league_id'].map(leagues_dict)
        
        # replace apostrophes in names with double apostrophes (SQL syntax)
        df['schedule_Opponent'] = df['schedule_Opponent'].str.replace("'", "''")
       
        # drop fbref league id column (not a db column)
        df = df.drop(columns=['schedule_fbref_league_id'])
       
        # rename columns to match db colnames
        df.rename(columns={'schedule_fbref_opponent_id': 'fbref_id', 
                           'schedule_Opponent': 'name'}, inplace=True)
        return df

    def get_matches_for_db(self, df, conn=None):
        """Isolates match information from raw scraped data for db insert."""

        # filter only rows from home perspective (caution: if tournaments (i.e. world cups) are in the data this might not work!)
        df = df[df['schedule_Venue']=='Home']
        # select relevant columns
        df = df[['schedule_fbref_match_id', 'schedule_fbref_league_id', 'schedule_fbref_squad_id', 'schedule_fbref_opponent_id', 'schedule_Date', 'schedule_Time', 'schedule_Round', 'schedule_Day']]

        # filter out matches set in the future (i.e. no schedule data yet)
        df = df[df['schedule_fbref_match_id']!='matchup']

        # get rid of rows with missing match id
        df = df[~df['schedule_fbref_match_id'].isna()]

        # get id matching dicts
        leagues_df = dbu.select_query("SELECT id, fbref_id FROM leagues;", conn=conn)
        leagues_dict = dict(zip(leagues_df['fbref_id'].astype(int), leagues_df['id']))
        teams_df = dbu.select_query("SELECT id, fbref_id FROM teams;", conn=conn)
        teams_dict = dict(zip(teams_df['fbref_id'], teams_df['id'].astype(int)))

        # add db id columns to df
        df['league_id'] = df['schedule_fbref_league_id'].map(leagues_dict)
        df['home_team_id'] = df['schedule_fbref_squad_id'].map(teams_dict)
        df['away_team_id'] = df['schedule_fbref_opponent_id'].map(teams_dict)

        # drop unneeded columns
        df.drop(columns=['schedule_fbref_league_id', 'schedule_fbref_squad_id', 'schedule_fbref_opponent_id'], inplace=True)

        # rename columns to match db colnames
        df.rename(columns={'schedule_fbref_match_id': 'fbref_id',
                            'schedule_Date': 'schedule_date',
                            'schedule_Time': 'schedule_time',
                            'schedule_Round': 'schedule_round',
                            'schedule_Day': 'schedule_day'}, inplace=True)
        # sort
        df = df.sort_values(['schedule_date', 'schedule_time'], ascending=[True, True])

        return df

    def clean_teamwages_for_db(self, df, conn=None):
        """Prepare team wages for db insert into teamwages table"""

        # drop exact duplicates
        n_rows_before = df.shape[0]
        df.drop_duplicates(inplace=True)
        print(60*'*')
        print(f"teamwages cleaning: Dropped {n_rows_before - df.shape[0]} exact duplicates.")

        # fix pct_estimated column
        df['pct_estimated'] = df['pct_estimated'].str.replace('%', '').astype(float)

        # id matching (new team_id column)
        teams_df = dbu.select_query("SELECT fbref_id, id FROM teams;", conn=conn)
        teams_dict = dict(zip(teams_df['fbref_id'], teams_df['id'].astype(int)))
        df['team_id'] = df['squad_id'].map(teams_dict)

        # drop unnecessary columns
        df.drop(columns=['squad_id', 'squad_name'], inplace=True)

        return df
    
    def clean_match_colnames(self, df):
        """Transforms column names of raw scraped data into names matching those in database."""

        # lowercase
        df.rename(columns=lambda x: x.lower(), inplace=True)

        # replace special characters
        df.rename(columns=lambda x: x.replace("+/-", "_plus_minus"), inplace=True)
        df.rename(columns=lambda x: x.replace('%', '_perc'), inplace=True)
        df.rename(columns=lambda x: x.replace("#", "number_"), inplace=True)
        df.rename(columns=lambda x: x.replace("/", "_per_"), inplace=True)
        df.rename(columns=lambda x: x.replace(":", ""), inplace=True)
        df.rename(columns=lambda x: x.replace("take-ons", "takeons"), inplace=True)
        df.rename(columns=lambda x: x.replace("-", "_minus_"), inplace=True)
        df.rename(columns=lambda x: x.replace("+", "_plus_"), inplace=True)
        df.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)

        # remove 'schedule' from all but a few selected columns
        keep_schedule = ['schedule_date', 'schedule_time', 'schedule_round', 'schedule_day']
        new_names = {}
        for col_name in df.columns:
            if col_name.startswith('schedule') and col_name not in keep_schedule:
                new_names[col_name] = col_name.replace('schedule_', '', 1)
        df.rename(columns=new_names, inplace=True)

        # rename 5 special columns (season & id)
        df.rename(columns={'fbref_season': 'season_str',
                           'fbref_league_id': 'league_id',
                           'fbref_squad_id': 'team_id',
                           'fbref_opponent_id': 'opponent_id',
                           'fbref_match_id': 'match_id'}, inplace=True)

        return df