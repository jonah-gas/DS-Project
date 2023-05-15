from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from datetime import datetime
import numpy as np

class MarketValues():
    
    def __init__(self, liga, timespan_or_season):
        self.liga = liga
        self.season = timespan_or_season
        
        if self.season > 1991:
            seasons = [self.season]
        elif self.season > 0 & self.season < 100:         
            currentYear = datetime.now().year
            seasons = range(datetime.now().year - self.season, datetime.now().year)
        else:
            print("Please enter either a year in the format '%YYYY' or a timespan (should not go past 1992)")
            
        find_clubs = pd.DataFrame(columns = ['season', 'club', 'club_id', 'link', 'liga', 'team_size'],
                          index=range(20 * len(seasons)))
        lig = liga.replace(" ","").lower()
        self.lig_dict = {'premierleague': 'GB1', 'laliga': 'ES1', 'seriea': 'IT1', 'bundesliga': 'L1', 'ligue1': 'FR1'}

        driver = webdriver.Chrome()
        i = 0
        both = ["odd", "even"]
        for year in seasons:
            url = "https://www.transfermarkt.de/premier-league/startseite/wettbewerb/" + self.lig_dict[lig] + "/plus/?saison_id=" + str(year)
            driver.get(url)
            clubs = driver.find_elements(By.XPATH, "/html/body/div[2]/main/div[2]/div[1]/div[2]/div[2]/div/table/tbody")
            for bt in both:
                club_soup = BeautifulSoup(clubs[0].get_attribute("innerHTML"), 'html.parser').find_all("tr", class_= bt)
                for club in club_soup:
                    td_tags = club.find_all("td")
                    for td in td_tags:
                        find_clubs.loc[i, "season"] = year
                        find_clubs.loc[i, "link"] = td_tags[0].find("a")["href"]
                        find_clubs.loc[i, "club"] = td_tags[0].find("a")["title"].replace(" ","").lower()
                        find_clubs.loc[i, "club_id"] = find_clubs.loc[i, "link"].split("/")[4]
                        find_clubs.loc[i, "team_size"] = td_tags[2].text
                        find_clubs.loc[i, "market_worth_tot"] = td_tags[6].text
                        find_clubs.loc[i, "liga"] = liga
                    i += 1
                time.sleep(2.1)  
        driver.quit()
        self.find_clubs = find_clubs.dropna()
        
        self.player_collections = pd.DataFrame(columns = ["season", "player_name", "player_id", "link", "market_worth"], 
                                                      index = range(0))
        
        
    def get_club_overview(self):
        "Returns basis information on the club in the specified season or timespan in a dataframe"
        return self.find_clubs
    
    def get_club_value(self, club, year):
        "Returns market value of the club in the specified season"
        club = club.replace(" ","").lower()
        value = self.find_clubs["market_worth_tot"].loc[(self.find_clubs.season == year) & (self.find_clubs.club == club)].values[0]
        return self.clean_data(value)
    
    def get_single_player_value(self,name, club, date = None):
        "Returns market value of player at specified date and overview in form of a dictionary"
        name = name.replace(" ","").lower()
        club = club.replace(" ","").lower()
        self.name = name
        
        if name in self.player_collections.player_name.unique():
            return self.clean_data(self.value_at_date(name, date))
        
        if date != None:
            date_to_year = datetime.strptime(date,'%d.%m.%Y')
            if date_to_year.month < 7:
                year = date_to_year.year - 1
            else:
                year = date_to_year.year
        
        driver = webdriver.Chrome()
        player_season = pd.DataFrame(columns = ["season", "player_name", "player_id", "link", "market_worth"], 
                                     index = range(1))
        club_url = self.find_clubs["link"].loc[(self.find_clubs.season == year) & (self.find_clubs.club == club)].values[0]

        url = "https://www.transfermarkt.de" + club_url + "/plus/1"
        driver.get(url)
        player = driver.find_element(By.CLASS_NAME, "items")
        player_soup = BeautifulSoup(player.get_attribute("innerHTML"), 'html.parser')
        
        odd = player_soup.find_all("td", attrs = {'class':"hauptlink"})[0::2]
        for o in range(len(odd)): 
            if odd[o].find("a").text.strip().replace(" ","").lower() != name:
                pass
            else:
                player_season.season = year
                player_season.player_name = odd[o].find("a").text.strip().replace(" ","").lower() #names
                player_season.link = odd[o].find("a")["href"] #links
                player_season.player_id = odd[o].find("a")["href"].split("/")[4]
    
        time.sleep(2.1)
        player_url = "https://www.transfermarkt.de" + player_season.link.values[0]
        driver.get(player_url)
    
        market_val = driver.execute_script('return window.Highcharts.charts[0]'
                                 '.series[0].options.data')
        player_mw_diclist = [item for item in market_val]
        player_mw_dict = {}
        for dict_entry in player_mw_diclist:
            player_mw_dict[dict_entry["datum_mw"]] = dict_entry["mw"]
            
        player_season.market_worth = [[player_mw_dict]]

        driver.quit()
        #self.player_season = player_season
        self.player_collections = pd.concat([self.player_collections, player_season])
        self.player_collections = self.player_collections.reset_index(drop = True)
        
        if date == None:
            date = datetime.now().date()
            return self.clean_data(self.value_at_date(name, date))
        else:
            return self.clean_data(self.value_at_date(name, date))
            
    def get_exact_team_value(self, date, team, club):
        team = list(map(lambda x: x.replace(" ","").lower(), team))
        club = club.replace(" ","").lower()
        
        no_info = np.setdiff1d(team, list(self.player_collections.player_name))
        #print(no_info)
        
        if date != None:
            date_to_year = datetime.strptime(date,'%d.%m.%Y')
            if date_to_year.month < 7:
                year = date_to_year.year - 1
            else:
                year = date_to_year.year
                
        driver = webdriver.Chrome()
        player_season = pd.DataFrame(columns = ["season", "player_name", "player_id", "link", "market_worth"], 
                                     index = range(len(no_info)))

        club_url = self.find_clubs["link"].loc[(self.find_clubs.season == year) & (self.find_clubs.club == club)].values[0]
        
        url = "https://www.transfermarkt.de" + club_url + "/plus/1"
        #print(url)
        driver.get(url)
        player = driver.find_element(By.CLASS_NAME, "items")
        player_soup = BeautifulSoup(player.get_attribute("innerHTML"), 'html.parser')
        i = 0
        odd = player_soup.find_all("td", attrs = {'class':"hauptlink"})[0::2]
        for o in range(len(odd)):
            #print(odd[o].find("a").text.strip().replace(" ","").lower())
            if odd[o].find("a").text.strip().replace(" ","").lower() in no_info:
                player_season.season[i] = year
                player_season.player_name[i] = odd[o].find("a").text.strip().replace(" ","").lower() #names
                player_season.link[i] = odd[o].find("a")["href"] #links
                player_season.player_id[i] = odd[o].find("a")["href"].split("/")[4]
                i += 1
            else:
                pass
        time.sleep(2.1)
        #print(player_season)
        
        for index, row in player_season.iterrows():
            #print(row.link)
            #print(type(row.link))
            url = "https://www.transfermarkt.de" + row.link
            driver.get(url)
            market_val = driver.execute_script('return window.Highcharts.charts[0]'
                                     '.series[0].options.data')
            player_mw_diclist = [item for item in market_val]
            player_mw_dict = {}
            for dict_entry in player_mw_diclist:
                player_mw_dict[dict_entry["datum_mw"]] = dict_entry["mw"]
            player_season.loc[index, 'market_worth'] = [player_mw_dict]
            time.sleep(2.1)
        
        player_season = player_season.dropna()
        self.player_collections = pd.concat([self.player_collections, player_season])
        self.player_collections = self.player_collections.reset_index(drop = True)
        
        #print(self.player_collections)
        actual_market_val = 0
        for player in team:
            actual_market_val += self.clean_data(self.value_at_date(player, date))

        driver.quit()
        
        return actual_market_val
        
        
        
        
    def value_at_date(self, name, date):
        #print(name)
        #print(self.player_collections[self.player_collections.player_name == name].index)
        index, = self.player_collections[self.player_collections.player_name == name].index
        #print(self.player_collections.market_worth[index])
        dates = list(self.player_collections.market_worth[index][0].keys())
        items = [datetime.strptime(x,'%d.%m.%Y') for x in dates]
        pivot = datetime.strptime(date,'%d.%m.%Y')
        nearest = min(items, key=lambda x: abs(x - pivot)).strftime('%d.%m.%Y')
        return self.player_collections.market_worth[index][0][nearest]
    
    def clean_data(self, data):
        splitted = data.replace(",",".").split()
        if 'Mio.' in splitted:
            data = float(splitted[0])*1e6
        elif 'Tsd.' in splitted:
            data = float(splitted[0])*1e3
        else:
            print('Something went wrong')
        return data
    
    def collected_player(self):
        return self.player_collections