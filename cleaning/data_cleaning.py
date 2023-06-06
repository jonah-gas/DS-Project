
import pandas as pd
import os

class DataCleaning:
    def __init__(self):
        # read in excel file with raw data cleaning information
        # (sheet_name=None will yield dict of sheet dfs: 'fbref_match', 'fbref_match_player')

        module_dir = os.path.dirname(os.path.abspath(__file__))
        cleaning_info_path = os.path.join(module_dir, 'cleaning_info.xlsx')
        self.cleaning_info = pd.read_excel(cleaning_info_path, sheet_name=None)

    
    
    def clean_raw_matchstats_for_db(self, df):
        """
        Cleans the raw data by dropping columns, removing rows with missing values, and renaming columns.

        Arguments:
            df (pd.DataFrame): Raw data to be cleaned.
        Returns:
            pd.DataFrame: Cleaned dataframe, ready for database insert.
        """
        ### rename columns ###
        df = self.clean_match_colnames(df)

        # pull info from cleaning_info df
        cols_to_drop = df.columns[self.cleaning_info['fbref_match']['drop_before_matchstats_insert']]
        cols_to_check_missing = df.columns[self.cleaning_info['fbref_match']['drop_row_if_missing_before_matchstats_insert']]

        ### drop columns ####
        df.drop(cols_to_drop, axis=1, inplace=True)

        ### validate data ###
        # drop rows with missing values in essential columns (ids, etc.)
        n_rows_before = df.shape[0]
        df.dropna(subset=cols_to_check_missing, inplace=True)
        print(f"Dropped {n_rows_before - df.shape[0]} rows due to missing values in columns {cols_to_check_missing}.")
        # drop rows with too many missing values
        n_rows_before = df.shape[0]
        thresh = 0.8
        df.dropna(thresh=thresh*df.shape[1], inplace=True)
        print(f"Dropped {n_rows_before - df.shape[0]} rows due to missing values in more than {thresh*100}% of columns.")

        # prepare names with apostrophes for SQL insert
        df['referee'] = df['referee'].str.replace("'", "''")
        df['captain'] = df['captain'].str.replace("'", "''")

        ### to do: replace fbref ids with db ids here ### (once module import problem is solved)

        return df

    def merge_data(self, data_raw_cleaned):
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
        merged_data = home_games.merge(away_games, left_on="fbref_match_id_home", right_on="fbref_match_id_away")

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
        
        return(merged_data)
    
    

    def clean_merged_data(self, merged_data):
        """
        Renames confusing column names, deletes unnecessary columns, and substitutes special characters.
        
        Arguments:
            merged_data (pd.DataFrame): Merged data.
        Returns:
            pd.DataFrame: Cleaned merged data.
        """
        # Rename confusing column names
        merged_data = merged_data.rename(columns={"gf": "gf_home",
                                                  "ga": "gf_away",
                                                  "fbref_squad_id": "fbref_home_id",
                                                  "fbref_opponent_id": "fbref_away_id",
                                                  "xg": "xg_home",
                                                  "xga": "xg_away"})
        # Delete unnecessary columns
        merged_data = merged_data.drop(["venue_home",
                                        "venue_away",
                                        "opponent_home",
                                        "opponent_away"], axis=1)
        
        # clean colnames
        merged_data_clean = self.clean_match_colnames(merged_data)
    
        return merged_data_clean

    def clean_match_colnames(self, df):
        """Lowercase and replace special characters in column names."""

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