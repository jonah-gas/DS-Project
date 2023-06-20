# -*- coding: utf-8 -*-
import os
import requests
from bs4 import BeautifulSoup
import csv
from urllib.parse import urljoin

class FootballDataCoUkScrapping:
    """A class for scrapping CSV data from the football-data.co.uk website."""

    def download_csv_data(self, country, league):
        """Downloads the CSV data for the specified country and league from the football-data.co.uk website. """

        # URL of the webpage containing the table
        url = f"https://www.football-data.co.uk/{country.lower()}m.php"

        # Send a GET request to the URL
        response = requests.get(url)

        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all the hyperlinks with the CSV file based on the label
        csv_links = soup.find_all('a', string=league)

        # Check if CSV links were found
        if csv_links:
            # Create a subfolder to save the CSV files
            subfolder = "football_data_uk"
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)

            # Assign season names to the CSV links
            season_names = ["2022-2023", "2021-2022", "2020-2021", "2019-2020", "2018-2019"]

            # Iterate over the first five CSV links
            for csv_link, season in zip(csv_links[:5], season_names):
                # Get the relative URL of the CSV file
                csv_relative_url = csv_link['href']

                # Construct the complete URL of the CSV file
                csv_url = urljoin(url, csv_relative_url)

                # Send a GET request to the CSV URL
                csv_response = requests.get(csv_url)

                # Check if the request was successful (status code 200)
                if csv_response.status_code == 200:
                    # Parse the CSV content as text and split it into lines
                    lines = csv_response.text.splitlines()

                    # Create a CSV writer object
                    csv_filename = f"{league}_{season}.csv"
                    csv_path = os.path.join(subfolder, csv_filename)
                    csv_writer = csv.writer(open(csv_path, "w", newline=""))

                    # Iterate over each line and write it to the CSV file
                    for line in lines:
                        csv_writer.writerow(line.split(","))

                    print(f"CSV {csv_filename} downloaded and saved successfully.")
                else:
                    print(f"Failed to download CSV {csv_filename}.")
        else:
            print(f"{league} CSV links not found.")



