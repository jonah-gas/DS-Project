import requests
import pandas as pd
from bs4 import BeautifulSoup 
import time
import warnings
import os


class Fbref:
    """ Implements all the functions required to scrape match- (and other) data from fbref.com"""
    # constructor
    def __init__(self, min_request_delay=3.1, output_dir=r'data/scraped/fbref'):
        # This table contains scraping-relevant info for every data type available on fbref. 
        # Storing it here makes it possible to write the scraping procedure as loop over all data types later instead of dealing with them individually.
        self.MATCH_DATA_TYPES = pd.DataFrame({
            'filter_text': ['Scores & Fixtures', 'Shooting', 'Goalkeeping', 'Passing', 'Pass Types', 'Goal and Shot Creation', 'Defensive Actions', 'Possession', 'Miscellaneous Stats'],
            'url_string': ['schedule', 'shooting', 'keeper', 'passing', 'passing_types', 'gca', 'defense', 'possession', 'misc'],
            'n_expected_cols': [18, 25, 36,32, 25, 24, 26, 33, 26], # expected cols in raw table (as read in by pd.read_html())
            'missing_header_replacement': [None, None, None, 'Attacking', 'General', None, 'General', 'General', None], # some tables have missing first-lvl headers for a couple second-level headers
            '10th_col_header_fix': [False, False, False, False, True, False, False, True, False] # some tables have the 'For {squadname}' header in the 10th column which we want to avoid (-> replace)
            }) 
        
        self.min_request_delay = min_request_delay # fbref allows 1 request every 3 seconds (https://www.sports-reference.com/bot-traffic.html)
        self.last_request_time = None
        self.output_dir = output_dir

    def get_squad_match_data_df(self, squad_id, league_id, season_start_year, print_logs=False):
        """Returns a dataframe containing all match data for a given squad, league and season."""

        df_list = [] # list to store dataframes for each data type

        # iterate over data types
        for i, data_type in self.MATCH_DATA_TYPES.iterrows():
        
            # build url 
            season_str = f'{season_start_year}-{season_start_year+1}'
            url = f"https://fbref.com/en/squads/{squad_id}/{season_str}/matchlogs/c{league_id}/{data_type['url_string']}"

            # retrieve data from url
            response = self._make_request(url, print_logs=print_logs)
            
            # read in table from html with pandas
            table_df = pd.read_html(response.text, match=data_type['filter_text'])[0]
            if table_df.shape[1] != data_type['n_expected_cols']: 
                raise ValueError(f"Unexpected number of columns ({table_df.shape[1]}, exp: {data_type['n_expected_cols']}) in scraped table for data type: {data_type['filter_text']}")
            
            # drop redundant columns and rows
            if i != 0: # only if not Scores & Fixtures table
                # drop first 9 columns, last column, and last row
                table_df = table_df.iloc[:-1, 9:-1]
                
            # deal with multiindex problems (see fbref_match_scraping_showcase.ipynb for detailed explanations)
            if table_df.columns.nlevels == 2:
                # replace missing first-level headers and perform 10th column header fix if necessary
                new_colnames = [(l1, l2) if not l1.startswith('Unnamed:') else (data_type['missing_header_replacement'], l2) for l1, l2 in table_df.columns]
                if data_type['10th_col_header_fix']: # we already dropped the first 9 columns so the 10th column is now the first
                    new_colnames[0] = (data_type['missing_header_replacement'], new_colnames[0][1]) # rename first-level header
                table_df.columns = pd.MultiIndex.from_tuples(new_colnames)
                # create new colnames with first level as prefix (lowercase, whitespaces removed)
                new_colnames = [f"{l1.lower()}_{l2}".replace(' ', '') for l1, l2 in table_df.columns]
                # drop first level to get rid of multiindex
                table_df.columns = table_df.columns.droplevel(0)
                # rename columns
                table_df.columns = new_colnames
            
            # add additional columns (fbref ids and season_str) during first iteration
            if i == 0:
                match_ids, opponent_ids = self._get_ids_from_matchlogs_table(response)
                # append columns with data we have as parameters
                table_df['fbref_season'] = season_str # same value each row
                table_df['fbref_league_id'] = league_id # same value each row
                table_df['fbref_squad_id'] = squad_id # same value each row
                # append extracted ids (lengths should match, if not pandas will throw error)
                table_df['fbref_opponent_id'] = opponent_ids
                table_df['fbref_match_id'] = match_ids
            
            # prefix the data type (url string) to all column names
            table_df.columns = [f"{data_type['url_string']}_{col}" for col in table_df.columns]

            # append to result list
            df_list.append(table_df)

        # concatenate retrieved dataframes
        try:
            result_df = pd.concat(df_list, axis=1, ignore_index=False)
        except:
            # print dataframe sizes for debugging
            print('Dataframe sizes:')
            for i, df in enumerate(df_list):
                print(f"{self.MATCH_DATA_TYPES.iloc[i]['filter_text']} df has shape: {df.shape}")
            raise Exception("Error during (vertical) concatenation of dataframes.")
        
        return result_df
    
    def get_league_match_data_df(self, league_id, season_start_year, return_df=True, save_csv=False, print_logs=False):
        """Scrapes all match data for a given league and season. Returns results in a single dataframe and/or saves to csv file. Caution: csv path must exist already!"""
        
        # get squad ids
        squad_id_df = self.get_squad_id_df(league_id, season_start_year, print_logs)
        if print_logs:
            print(f"Retrieved {squad_id_df.shape[0]} squad_ids for {league_id=}, {season_start_year=}.")

        # iterate over squads
        df_list = []
        for i, squad_id in squad_id_df.iterrows():
            if print_logs:
                print(f"Starting data collection for squad {i+1}/{len(squad_id_df)}: {squad_id['squad_name']} (id: {squad_id['squad_id']})")
            data_df = self.get_squad_match_data_df(squad_id['squad_id'], league_id, season_start_year, print_logs=print_logs)
            df_list.append(data_df)

        # before concatenating, perform some checks
        # column check: must match exactly (note: maybe should also check against a final expected column count)
        if not all([df_list[0].columns.equals(df.columns) for df in df_list]):
            # print column counts for debugging
            for i in enumerate(df_list):
                print(f"{df_list[i].shape[1]} columns in data df for {squad_id['squad_name']} (id: {squad_id['squad_id']}).")
            raise Exception("Discrepancy in column count (or names) between squad dfs prevents horizontal concatenation!")
        # row count check: slight differences could be due to league rules -> throw warning 
        if not all([df_list[0].shape[0]==df.shape[0] for df in df_list]):
            # print row counts for debugging
            for i in enumerate(df_list):
                print(f"{df_list[i].shape[0]} rows in data df for {squad_id['squad_name']} (id: {squad_id['squad_id']}).")
            warnings.warn("Discrepancy in match count between squads. Check if error or explained by league rules.")

        # concatenate
        result_df = pd.concat(df_list, axis=0, ignore_index=True)

        # save to csv & return df (if requested)
        try:
            if save_csv:
                filename = f"league{league_id}_ssy{season_start_year}_cols{result_df.shape[1]}_rows{result_df.shape[0]}.csv"
                path = os.path.join(self.output_dir, filename)
                result_df.to_csv(path, sep=';', index=False) # some columns can contain commas -> semicolon as sep
        except:
            raise Exception(f"Error during saving in '{path}'. Check if output dir '{self.output_dir}' exists.")
        finally:
            if return_df:
                return result_df

    def get_squad_id_df(self, league_id, season_start_year, print_logs=False):
        """Returns a df of fbref squad ids for a given league and season."""

        # build url
        season_str = f"{season_start_year}-{season_start_year+1}"  # transforms start_year to needed url-structure
        url = f"https://fbref.com/en/comps/{league_id}/{season_str}"
        
        # make request
        response = self._make_request(url, print_logs=print_logs)
    
        # find relevant table
        soup = BeautifulSoup(response.text, 'html.parser')
        standings_table = soup.select('table.stats_table')[0] # relevant table is always first on the page

        # extract relevant links from table
        links = standings_table.find_all('a')
        links = [l.get("href") for l in links]
        links = [l for l in links if '/squads/' in l] 
        # links should now have the form: "/en/squads/{squad_id}/{season_str}/{squad_name in regular characters and separated by '-'}-Stats"
        
        # extract squad ids and names from links
        squads = [] 
        for link in links:
            parts = link.split('/')
            squad_id = parts[-3] # extract squad id
            team_name = parts[-1] # extract team name (still has '-Stats' at the end)
            team_name = team_name[:-6].replace('-', ' ') # cut off '-Stats' and replace '-'
            squads.append((squad_id, team_name))

        # return as df
        result_df = pd.DataFrame(squads, columns=['squad_id', 'squad_name'])
        return result_df 

    def _implement_delay(self):
        """Implements delay between requests if necessary. Should be called right before a request is made."""
        if self.last_request_time is not None:
            current_time = time.time()
            if current_time - self.last_request_time < self.min_request_delay:
                time.sleep(self.min_request_delay - (current_time - self.last_request_time))
        self.last_request_time = time.time()

    def _make_request(self, url, print_logs=False):
        """Makes a request to a given url and returns the response. Handles delay and printing logs. Should always be used for requests."""
        self._implement_delay()
        if print_logs:
            print(f"Retrieving data from {url}.")
        response = requests.get(url)
        if response.status_code != 200: # 200 -> OK
            raise Exception(f"Got response status code {response.status_code} instead of 200.")
        return response

    def _get_ids_from_matchlogs_table(self, response):
        """Returns lists containing the fbref match ids and opponent squad ids from the table on a matchlogs page"""
        # find table with bs4
        soup = BeautifulSoup(response.text, 'html.parser')
        soup_table = soup.find('table', {'id': 'matchlogs_for'}) # table id is always 'matchlogs_for'

        match_ids, opponent_ids = [], []
        # iterate through table rows 
        for row in soup_table.find_all('tr'):
            # skip avoid non-data rows
            if (row.find('th', {'class': 'poptip'}) is None and # no header row
                row.find('th', {'class': 'over_header'}) is None and # no over header row
                row.find('th', {'class': 'left iz'}) is None): # no bottom summary row
            
                # find opponent column
                td_opp = row.find('td', {'data-stat': 'opponent'})
                # extract link (has form: /en/squads/{squad_id}/...)
                opponent_link = td_opp.find('a')['href']
                # extract opponent squad id from link (has)
                opponent_squad_id = opponent_link.split('/')[3]

                # find match report column
                td_match = row.find('td', {'data-stat': 'match_report'})
                # extract link (has form: /en/matches/{match_id}/..)
                match_report_link = td_match.find('a')['href']
                # extract match id from link
                match_id = match_report_link.split('/')[3]

                match_ids.append(match_id)
                opponent_ids.append(opponent_squad_id)
        return match_ids, opponent_ids