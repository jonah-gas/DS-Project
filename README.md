# Top3Bets - Soccer Prediction App

Access the public-facing web app here: [top3bets.streamlit.app](top3bets.streamlit.app)

This repo contains source code for all components of the project. The most important contents/modules are highlighted in the following description.

## Scraping New (or Old) Data

- ``scraping/fbref_scraping.py`` implements the main scraping class, which enables scraping team-level match statistics as well as wage data from [fbref.com](https://fbref.com/en/).
- ``scraping/scraping_notebook.ipynb`` demonstrates scraping data for a specified league and season with a single function call. (Note that scraped data is saved in ``data/scraped/fbref/`` in the appropriate subfolders by default.)

## Data Cleaning

- ``cleaning/data_cleaning.py`` implements cleaning steps to be applied to newly scraped data. Used in ``db_inserts.py`` (see next section).

## Database Server

The project includes a postgres database server which stores an up-to-date clean version of the data serving as basis for feature generation. Several modules, including the streamlit app, rely on being able to query this database. 

- ``database_server/db_setup.sql`` defines the SQL database schema.
- ``database_server/db_setup.ipynb`` can be used to completely wipe (!) and recreate the database schema from a jupyter notebook.
- ``database_server/db_utilities.py`` defines utility functions for connecting to and querying the database. All database connections/queries in this project occur via this module.
- ``database_server/db_inserts.py`` defines functions to first load (from ``.csv``-files in the ``data/`` directory), then clean and finally insert newly scraped data into the appropriate tables in the database.

## Feature Generation & Modelling

Feature generation and model training is implemented separately for two approaches: 

#### 'Traditional' ML models such as XGBoost, Random Forest and Logistic Regression

- ``models/trad_ml/feature_generation.py`` implements a flexible feature generation class which constructs feature sets from our base data set. Accepts a dictionary of variables in order to be able to conveniently optimize the feature generation parameters (see ``trad_ml/hyperparameter_optimization_<model type>.ipynb``). Notably, the main function of this class, ``generate_features()``, is able to produce either a full train/test split of features, or (when provided a home and away team ID) only a single feature row for predictions.
- ``models/trad_ml/saved_models/`` is where the final six trained models are stored.
- 
#### A neural net with Long Short-Term Memory architecture (LSTM)

-``models/neural_net/`` contains modules to train and predict using the LSTM model. Also contains the final trained model.

## Web App (hosted via [streamlit](https://streamlit.io/))

- ``streamlit_app/Welcome.py`` is the entry page (-> to run the app locally: from the root directory, run ``streamlit run streamlit_app/Welcome.py``).
- ``streamlit_app/pages/`` contains ``.py`` files for the other 6 pages of the web app.
- ``app_functions.py`` contains most of the functionality of the app, functions in here are called from the individual page files.

## Secrets Management

To prevent exposure of sensitive information (database server IP, passwords etc.), the ``db_utilities`` module relies on a config file being present in the following location: ``database_server/config/config.ini``. This ``config`` folder is not part of the repository! For access, please reach out to us directly.

For the web app we had to adjust to streamlit's secret management system, which requires entering the sensitive data directly on streamlit's platform. To run the app locally, a ``.streamlit/secrets.toml`` file is required. This file is also omitted from the repo. 


