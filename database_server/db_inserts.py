import os
import sys

root_path = os.path.abspath(os.path.join('..')) # <- adjust such that root_path always points at the root project dir (i.e. if current file is two folders deep, use '../..'). 
if root_path not in sys.path:
    sys.path.append(root_path)

import database_server.db_utilities as dbu 

from cleaning.data_cleaning import DataCleaning

import numpy as np
import pandas as pd

"""
Note on SQL inserts: 

The efficient way to insert many (potentially) new rows into a table would be to COPY FROM a csv file into a temporary table,
then write a query which copies only new rows into the main table.
But since we are usually only handling a few rows at a time, we use (bulk) INSERT INTO statements to make debugging easier and to have more control over handling data types and NA values.
Duplicate inserts are avoided by using the ON CONFLICT clause along with the table's appropriate constraint (i.e. in most casses the primary key).
"""

def insert_countries(batchsize=1):
    """Read in countries.csv and insert new entries into the db."""
    # read in csv
    csv_path = os.path.join(root_path, 'data', 'scraped', 'fbref', 'countries', 'countries.csv')
    countries_df = pd.read_csv(csv_path)

    # perform inserts
    # (only few rows expected -> no batches)
    n_ins, n_failed = _perform_batch_insert(df=countries_df, 
                                            table_name='countries', 
                                            constraint_name='countries_pkey',
                                            batchsize=batchsize)
    return n_ins, n_failed

def insert_leagues(batchsize=1):
    """Read in leagues.csv and insert new entries into the db."""
    # read in csv
    csv_path = os.path.join(root_path, 'data', 'scraped', 'fbref', 'leagues', 'leagues.csv')
    leagues_df = pd.read_csv(csv_path).drop(columns=['season_str_format'])
    leagues_df['fbref_id'] = leagues_df['fbref_id'].astype(str) # db column fbref_id is string
    
    # perform inserts
    # (only few rows expected -> no batches)
    n_ins, n_failed = _perform_batch_insert(df=leagues_df, 
                                            table_name='leagues', 
                                            constraint_name='leagues_fbref_id_key',
                                            batchsize=batchsize)
    return n_ins, n_failed

def insert_teamwages(batchsize=25):
    """Read in teamwages .csv-files and insert new entries into the db."""
    # read in csv
    wages_path = os.path.join(root_path, 'data', 'scraped', 'fbref', 'wages')
    df = pd.concat([pd.read_csv(os.path.join(wages_path, f), sep=';') for f in os.listdir(wages_path)])

    # clean 
    cleaner = DataCleaning()
    clean_df = cleaner.clean_teamwages_for_db(df)
    # perform inserts
    n_ins, n_failed = _perform_batch_insert(df=clean_df, 
                                            table_name='teamwages', 
                                            constraint_name='teamwages_pkey',
                                            colnames=clean_df.columns,
                                            batchsize=batchsize)
    return n_ins, n_failed

def insert_match_data(include_archive=False, include_new=True):
    """Read in new match data from files in /new and/or /archive.
        If required, inserts are made in the following tables in this order:
            1. teams
            2. matches
            3. matchstats

        Note that leagues and countries tables should already contain all required entries.
    """
    # read in data and concat to one df
    match_path = os.path.join(root_path, 'data', 'scraped', 'fbref', 'match')
    df_archive, df_new = None, None

    if include_archive:
        archive_path = os.path.join(match_path, 'archive')
        df_archive = pd.concat([pd.read_csv(os.path.join(archive_path, f), sep=';') for f in os.listdir(archive_path)])
    if include_new:
        new_path = os.path.join(match_path, 'new')
        df_new = pd.concat([pd.read_csv(os.path.join(new_path, f), sep=';') for f in os.listdir(new_path)])

    raw_df = pd.concat([df for df in [df_archive, df_new] if df is not None])


    # create cleaner instance
    cleaner = DataCleaning()
    # perform inserts in teams, matches and matchstats tables based on raw matches df
    n_ins_teams, n_failed_teams = _insert_teams(raw_df, cleaner=cleaner)
    n_ins_matches, n_failed_matches = _insert_matches(raw_df, cleaner=cleaner)
    n_ins_ms, n_failed_ms = _insert_matchstats(raw_df, cleaner=cleaner)

    # print summary
    print(60*'-')
    print(f"teams: {n_ins_teams} inserted, {n_failed_teams} rejected.")
    print(f"matches: {n_ins_matches} inserted, {n_failed_matches} rejected.")
    print(f"matchstats: {n_ins_ms} inserted, {n_failed_ms} rejected.")


def _insert_teams(df, cleaner=None, batchsize=10):
    """Insert new teams into teams table.
       df: raw matches df"""
    # clean 
    if cleaner is None:
        cleaner = DataCleaning()
    clean_df = cleaner.get_teams_for_db(df)
    
    # perform inserts
    n_ins, n_failed = _perform_batch_insert(df=clean_df, 
                                            table_name='teams', 
                                            batchsize=batchsize, 
                                            constraint_name='teams_fbref_id_key',
                                            colnames=clean_df.columns)
    
    return n_ins, n_failed

def _insert_matches(df, cleaner=None, batchsize=50):
    """Insert new matches into matches table.
       df: raw matches df
    """
    if cleaner is None:
        cleaner = DataCleaning()
    # clean 
    clean_df = cleaner.get_matches_for_db(df)
    
    # perform inserts
    n_ins, n_failed = _perform_batch_insert(df=clean_df, 
                                            table_name='matches', 
                                            batchsize=batchsize, 
                                            constraint_name='matches_fbref_id_key',
                                            colnames=clean_df.columns)
    return n_ins, n_failed  

def _insert_matchstats(df, cleaner=None, batchsize=100):
    """Insert new matchstats into matchstats table.
       df: raw matches df
    """
    # clean 
    if cleaner is None:
        cleaner = DataCleaning()
    clean_df = cleaner.clean_matchstats_for_db(df)

    # perform inserts
    n_ins, n_failed = _perform_batch_insert(df=clean_df, 
                                            table_name='matchstats', 
                                            constraint_name='matchstats_pkey',
                                            batchsize=batchsize, 
                                            colnames=clean_df.columns)
    return n_ins, n_failed
     
def _perform_batch_insert(df, table_name, colnames=None, constraint_name=None, batchsize=1):
    """
    Helper function: Insert df into table in batches of batchsize.
        df: df to insert 
        colnames: list of column names for sql query (retrieved from db if not provided)
        batchsize: number of rows to insert in one batch
    """

    if colnames is None:
        # get column names from db, exclude 'id'-col (serial PK in tables)
        colnames = dbu.select_query(f"SELECT * FROM {table_name} LIMIT 1;").columns
    colnames_str = ', '.join([c for c in colnames if not c in ['id']])

    n_ins = 0 # counter for successfully inserted rows
    n_failed = 0 # counter for failed row inserts
    query_str = ""
    conn = dbu.get_conn(type='DB_client')
    cursor = conn.cursor()
    #df.reset_index(inplace=True, drop=True) # we use df index as counter variable
    for i, row in enumerate(df.itertuples(index=False)):
        # get values string
        values_str = ', '.join([(('\'' + v + '\'' if type(v)==str else str(v)) if not pd.isna(v) else 'NULL') for v in row])
        # build/append query string
        if query_str == "":
            query_str = f"""INSERT INTO {table_name} ({colnames_str}) VALUES ({values_str})"""
        else: query_str += f", ({values_str})"""
        if (i+1) % batchsize == 0:
            # add ON CONFLICT clause with constraint_name if provided
            query_str += f" ON CONFLICT ON CONSTRAINT {constraint_name} DO NOTHING" if constraint_name else ''

            # process insert
            n_ins, n_failed = _process_individual_insert(conn, cursor, table_name, query_str, batchsize, n_ins, n_failed)
            print(f"rows processed: {i+1}/{df.shape[0]}")
            query_str = "" # reset query str

    # insert remaining rows
    n_remaining_rows = len(df) - n_ins - n_failed
    if query_str != "":
        query_str += f" ON CONFLICT ON CONSTRAINT {constraint_name} DO NOTHING" if constraint_name else ''
        n_ins, n_failed = _process_individual_insert(conn, cursor, table_name, query_str, n_remaining_rows, n_ins, n_failed)
        print(f"rows processed: {df.shape[0]}/{df.shape[0]}")
    cursor.close()
    conn.close()
    print(f"{'-'*60}\nInserted {n_ins} new rows into {table_name} table. {n_failed} inserts rejected.")

    return n_ins, n_failed


def _process_individual_insert(conn, cursor, table_name, query_str, batchsize, n_ins, n_failed):
    """Helper function: Process a single (batch) insert query."""
    try:
        cursor.execute(query_str if query_str.endswith(';') else query_str + ';')
        response = cursor.statusmessage
        conn.commit() # commit transaction
        print(response) 
        successful_ins = int(response.split(' ')[-1]) # get number of inserted rows from response
        n_ins += successful_ins
        n_failed += batchsize - successful_ins # number of rejected rows (due to constraint violation)
    except Exception as e:
        print(f"Error with the following insert into {table_name} table: \n{query_str}\n{e}")
        conn.rollback() # rollback transaction
        n_failed += batchsize
    return n_ins, n_failed

     





    
     
     
    