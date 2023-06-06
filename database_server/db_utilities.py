import psycopg2
import pandas as pd

import configparser
import os

def read_config():
    """Returns a ConfigParser object containing the database connection information.""" 
    
    # it is important for this function to work if called from any directory
    path_from_root = r'database_server\config\db_config.ini'
    config = configparser.ConfigParser()
    # get absolute path to this module file
    module_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(module_dir, 'config', 'db_config.ini')
    config.read(config_path)
    return config

def get_conn(type='DB_client', config=None):
    """Returns a connection object to the database specified in the config file. 
        Arguments:
            type:       either 'DB_client' or 'DB_su' for client (default) or superuser connection, respectively
            config: a ConfigParser object containing the database connection information"""
    
    if config is None:
        # read config file if config object was not provided
        config = read_config()
    # instantiate connection object
    conn = psycopg2.connect(
        host=config[type]['host'],
        port=int(config[type]['port']),
        database=config[type]['database'],
        user=config[type]['user'],
        password=config[type]['password']
    )
    return conn


def select_query(query_str, conn=None):
    """Returns the results of the select query as a pandas df. 
        Arguments:
            query:      SQL query to be executed
            conn:       connection object to the database. If provided, the connection will NOT be closed inside this function!
                        If not provided, a connection will be opened and closed in this function."""
    to_be_closed = False # flag to indicate whether connection should be closed inside this function
    if conn is None:
        # get connection if not provided
        conn = get_conn(type='DB_client')
        to_be_closed = True

    # execute query in context of cursor object
    with conn.cursor() as cur:
        cur.execute(query_str if query_str.endswith(';') else query_str + ';')
        result = cur.fetchall() # fetch all rows from cursor object
        colnames = [desc[0] for desc in cur.description] # get column names from cursor object
        result_df = pd.DataFrame(result, columns=colnames) # create pandas dataframe from result

    if to_be_closed:
        conn.close() # close connection if required
    
    return result_df

def manipulating_query(query_str, conn=None):
    """Executes any non-select query (e.g. INSERT, UPDATE, ...) which might manipulate the database.
       Note: Many repeated calls to this function should be avoided, instead rather implement a loop which reuses connection and cursor objects.
        Arguments:
            query:      SQL query to be executed
            conn:       connection object to the database. If provided, the connection will NOT be closed inside this function!
                        If not provided, a connection will be opened and closed in this function.

        Returns:        SQL status message"""
    to_be_closed = False # flag to indicate whether connection should be closed inside this function
    if conn is None:
        # get connection if not provided
        conn = get_conn(type='DB_client')
        to_be_closed = True
    # create cursor and execute query
    with conn.cursor() as cur:
        cur.execute(query_str if query_str.endswith(';') else query_str + ';')
        status_msg = cur.statusmessage # get response
        conn.commit() # commit changes

    if to_be_closed:
        conn.close() # close connection if required
    
    return status_msg
