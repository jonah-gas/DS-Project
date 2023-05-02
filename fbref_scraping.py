import requests
import pandas as pd
from bs4 import BeautifulSoup 
import time


class Fbref:
    """ Implements all the functions required to scrape match- (and other) data from fbref.com"""
    # constructor
    def __init__(self, min_request_delay=3):
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

    def get_squad_match_data_df(self, squad_id, league_id, season_start_year, print_logs=False):
        """Returns a dataframe containing all match data for a given squad, league and season."""

        df_list = [] # list to store dataframes for each data type

        # iterate over data types
        for i, data_type in self.MATCH_DATA_TYPES.iterrows():
        
            # build url 
            season_str = f'{season_start_year}-{season_start_year+1}'
            url = f"https://fbref.com/en/squads/{squad_id}/{season_str}/matchlogs/c{league_id}/{data_type['url_string']}"

            # implement delay (if necessary)
            if self.last_request_time is not None:
                self._implement_delay() 

            if print_logs:
                print(f"Retrieving data from {url}")
            self.last_request_time = time.time()

            # retrieve data 
            response = requests.get(url) # 
            if response.status_code != 200: # 200 -> OK
                raise Exception(f"Got response status code {response.status_code} instead of 200.")
            
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
            raise Exception("Error during concatenation of dataframes.")
        
        return result_df
    
    # to do
    def scrape_league_season(self, league_id, season_start_year, return_df=True, save_csv=False, csv_path=None, print_logs=False):
        """Scrapes all match data for a given league and season. Returns a single dataframe and/or saves to csv file."""

    # to do
    def get_squad_ids(self, league_id, season_start_year):
        """Returns a list of fbref squad ids for a given league and season."""
        
        year_range = f"{season_start_year}-{season_start_year+1}"  # transforms start_year to needed url-structure
        base_url = "https://fbref.com/en/comps/"
        url = base_url + str(league_id) + "/" + year_range + "/" + year_range + "/"    # build the complete url to request from
        data = requests.get(url)    # send a GET request for the complete url an store the response
    
        soup = BeautifulSoup(data.text, 'html.parser')
        standings_table = soup.select('table.stats_table')[0]
        # create a BeatifulSoup object to parse the html and find all standings tables in the html
    
        links = standings_table.find_all('a')
        links = [l.get("href") for l in links]
        links = [l for l in links if '/squads/' in l] 
        team_urls = [f"https://fbref.com{l}" for l in links]  
        # filter the links that include the squad data and create a lists with the complete urls
        
        squad_index = {}    # initialize an empty dictionary to store team_name and squad_id

        for team_url in team_urls:
            parts = team_url.split('/')
            squad_id = parts[-3]
            team_name = parts[-1]
            team_name = team_name[:-6]
            squad_index[team_url] = {'team_name': team_name, 'squad_id': squad_id}
            # for every team_url extract team_name and squad_id and add them to the squad_index dictionary

        df = pd.DataFrame.from_dict(squad_index).T
        result_df = df.reset_index(drop=True)
        # create a DataFrame from the squad_index and reset the index

        return result_df # return DataFrame

    # helper function
    def _implement_delay(self):
        """Implements delay between requests if necessary. Should be called right before a request is made."""
        current_time = time.time()
        if current_time - self.last_request_time < self.min_request_delay:
            time.sleep(self.min_request_delay - (current_time - self.last_request_time))
        self.last_request_time = time.time()

    # helper function 
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