# -*- coding: utf-8 -*-
"""
Created on Wed May 10 15:58:07 2023

@author: johan_nii2lon
"""
import pandas as pd

class data_cleaning:
    def clean_raw_data(self, data_raw, fbref_matchdata_rawcleaning):
        """
        Cleans the raw data by dropping columns, removing rows with missing values, and renaming columns.

        Arguments:
            data_raw (pd.DataFrame): Raw data to be cleaned.
            fbref_matchdata_rawcleaning (pd.DataFrame): Dataframe indicating columns to drop.
        Returns:
            pd.DataFrame: Cleaned data.
        """
        # Get the list of columns to drop from data_raw based on fbref_matchdata_rawcleaning['drop_before_merge']
        cols_to_drop = data_raw.columns[fbref_matchdata_rawcleaning['drop_before_merge']].tolist()
        # Drop the columns from data_raw
        data_raw.drop(cols_to_drop, axis=1, inplace=True)

        # Drop rows with missing values in the 'schedule_Result' column
        data_raw.dropna(subset=['schedule_Result'], inplace=True)

        # Rename columns by removing "schedule" in the column names
        new_names = {}
        for col_name in data_raw.columns[4:20]:
            new_names[col_name] = col_name[9:]

        data_raw.rename(columns=new_names, inplace=True)
        data_raw_cleaned = data_raw

        return data_raw_cleaned

    def merge_data(self, data_raw_cleaned):
        """
        Merges the two sets of observations for one game and removes duplicated rows.

        Arguments:
            data_raw_cleaned (pd.DataFrame): Cleaned data.
        Returns:
            pd.DataFrame: Merged data.
        """
        # Filter rows where Venue is 'Home' and where Venue is 'Away'
        home_games = data_raw_cleaned[data_raw_cleaned['Venue'] == 'Home']
        away_games = data_raw_cleaned[data_raw_cleaned['Venue'] == 'Away']

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
        merged_data = merged_data.rename(columns={"GF": "GF_home",
                                                  "GA": "GF_away",
                                                  "fbref_squad_id": "fbref_home_id",
                                                  "fbref_opponent_id": "fbref_away_id",
                                                  "xG": "xG_home",
                                                  "xGA": "xG_away"})
        # Delete unnecessary columns
        merged_data = merged_data.drop(["Venue_home",
                                        "Venue_away",
                                        "Opponent_home",
                                        "Opponent_away"], axis=1)
        # Substitute special characters
        merged_data.rename(columns=lambda x: x.replace("+/-", "_plus_minus"), inplace=True)
        merged_data.rename(columns=lambda x: x.replace('%', '_perc'), inplace=True)
        merged_data.rename(columns=lambda x: x.lower(), inplace=True)
        merged_data.rename(columns=lambda x: x.replace("#", "number_"), inplace=True)
        merged_data.rename(columns=lambda x: x.replace("/", "_per_"), inplace=True)
        merged_data.rename(columns=lambda x: x.replace(":", ""), inplace=True)
        merged_data.rename(columns=lambda x: x.replace("take-ons", "takeons"), inplace=True)
        merged_data.rename(columns=lambda x: x.replace("-", "_minus_"), inplace=True)
        merged_data.rename(columns=lambda x: x.replace("+", "_plus_"), inplace=True)
    
        merged_data_clean = merged_data
    
        return merged_data_clean
    
