import os
import sys

root_path = os.path.abspath(os.path.join('')) # <- adjust such that root_path always points at the root project dir
if root_path not in sys.path:
    sys.path.append(root_path)

import streamlit as st
import database_server.db_utilities as dbu
from models.trad_ml.feature_generation import FeatureGen

import plotly.express as px
import pandas as pd
import pickle as pkl



####################
# cached functions #
####################

# get db connection object
@st.cache_resource(ttl=60*60*24*7, max_entries=1, show_spinner="Creating database connection...")
def get_db_conn():
    return dbu.get_conn(config=st.secrets)

# load teams data (e.g. for prediction selection)
@st.cache_data(ttl=60*60*24*7, max_entries=1, show_spinner="Retrieving teams data...")
def get_teams_data():
    # load teams relevant for prediction (-> at least one row in matchstats table)
    query_str = """
        SELECT t.id, t.name, t.country FROM teams as t
        WHERE EXISTS(SELECT 1 FROM matchstats as ms WHERE ms.team_id = t.id)
        ORDER BY country, name;
        """
    teams_df = dbu.select_query(query_str, conn=st.session_state['conn'])
    # create matching dicts
    teams_name2id = {row[0]: row[1] for row in teams_df[['name', 'id']].itertuples(index=False)}
    teams_id2name = {row[1]: row[0] for row in teams_df[['name', 'id']].itertuples(index=False)}

    return teams_df, teams_name2id, teams_id2name

# create feature gen instance and load full db dataset 
@st.cache_resource(ttl=60*60*24*7, max_entries=1, show_spinner="Retrieving match data for feature generation...")
def get_feature_gen_instance():
    # create feature gen instance
    fg = FeatureGen(conn=st.session_state['conn'], dpo_path=os.path.join(root_path, 'models', 'trad_ml', 'saved_data_prep_objects') ) # note: no params dict provided yet -> need to call set_params() later
    # load full dataset
    fg.load_data()
    return fg

# get available season strings for selection boxes
@st.cache_data(ttl=60*60*24*7, max_entries=1, show_spinner="Retrieving season data...")
def get_season_str_options():
    query_str = """
        SELECT DISTINCT ms.season_str 
        FROM matchstats as ms
        ORDER BY season_str DESC;
        """
    seasons_list = dbu.select_query(query_str, conn=st.session_state['conn']).iloc[:,0].tolist()
    return seasons_list

# get available league ids (and names) for selection boxes
@st.cache_data(ttl=60*60*24*7, max_entries=1, show_spinner="Retrieving season data...")
def get_league_options():
    query_str = """
        SELECT id, name 
        FROM leagues
        ORDER BY id ASC;
        """
    df = dbu.select_query(query_str, conn=st.session_state['conn'])
    league_ids = df['id'].to_list()
    league_names = df['name'].to_list()
    return league_ids, league_names

@st.cache_resource(ttl=60*60*24*7, max_entries=10, show_spinner="Retrieving aggregated data...")
def get_aggregated_stats(agg_type, season_str_selection, league_id_selection):
    # columns to include in aggregation (db colname, app display name, table alias)
    to_include = [  ('gf', 'Goals scored', 'ms'), 
                    ('xg', 'XG scored', 'ms'),                  
                    ('ga', 'Goals conceded', 'ms'),
                    ('xga', 'XG conceded', 'ms'),
                    ('possession_general_poss', 'Possession %', 'ms'),
                    ('keeper_performance_save_perc', 'Keeper save %', 'ms'),
                    ('passing_total_cmp', 'Passes completed', 'ms'),
                    ('misc_performance_crdy', 'Yellow cards', 'ms'),
                    ('misc_performance_crdr', 'Red cards', 'ms'),
                    ('misc_performance_fls', 'Fouls committed', 'ms')
                    ]
    # build aggregation string part of selection
    select_str = ', '.join([f'to_char({agg_type}({table_alias}.{var}), \'9999999.00\') AS "{alias}"' for var, alias, table_alias in to_include])
    # assemble full query string
    query_str = f"""
            SELECT t.name, {select_str} 
            FROM matchstats ms
            LEFT JOIN matches m ON ms.match_id = m.id
            LEFT JOIN teams t ON ms.team_id = t.id
            LEFT JOIN teamwages w ON ms.team_id = w.team_id
                                    AND ms.season_str = w.season_str
            WHERE ms.season_str = '{season_str_selection}'
            AND ms.league_id = {league_id_selection}
            GROUP BY t.name, ms.season_str, ms.league_id;
                    """
    # execute query
    df = dbu.select_query(query_str, conn=st.session_state['conn'])
    return df

#####################
# non-cached utils  #
#####################

def load_info_dict(model_dict_name):

    info_dict_path = os.path.join(root_path, "models", "trad_ml", "sweep_results")
    d = pkl.load(open(os.path.join(info_dict_path, f"{model_dict_name}.pkl"), "rb"))

    # since info dicts are often optimization sweep results, they might not yet have the data prep object names defined -> define them here via run name
    if d['fg_config']['apply_ohe'] and d['fg_config']['ohe_name'] is None:
        d['fg_config']['ohe_name'] = f"OneHotEncoder_{d['run_name']}"

    if d['fg_config']['apply_scaler'] and d['fg_config']['scaler_name'] is None:
        d['fg_config']['scaler_name'] = f"StandardScaler_{d['run_name']}"

    if d['fg_config']['apply_pca'] and d['fg_config']['pca_name'] is None:
        d['fg_config']['pca_name'] = f"PCA_{d['run_name']}"

    return d

def load_model(model_name):
    model_path = os.path.join(root_path, "models", "trad_ml", "saved_models", f"{model_name}.pkl")
    return pkl.load(open(model_path, "rb"))

def load_lstm_model(state_dict_name):
    state_dict_path = os.path.join(root_path, "models", "neural_net", "models", f"{state_dict_name}.pt")
    #model = TheModelClass(*args, **kwargs)
    #model.load_state_dict(torch.load(models_path))
    #model.eval()
    #return model
    pass

# load trad ml models in session state
def load_trad_ml_models():
    if 'trad_ml_models' not in st.session_state:
        # define model dict (contains loaded models as well as model-specific params / info)
        st.session_state['trad_ml_models'] = {
            'XGBoost**':    {'info': load_info_dict('xgb_all_train'),
                                'model': load_model('MultiOutputClassifier_xgb_all_train')},

            'RF**':         {'info': load_info_dict('rf_all_train'),
                                'model': load_model('MultiOutputClassifier_rf_all_train')},

            'LogReg**':     {'info': load_info_dict('logreg_all_train'),
                                'model': load_model('MultiOutputClassifier_logreg_all_train')},

            'XGBoost':      {'info': load_info_dict('xgb_one_season_test'), # <- info dict file name (without .pkl)
                                'model': load_model('MultiOutputClassifier_xgb_one_season_test')}, # <- model file name (without .pkl)

            'RF':           {'info': load_info_dict('rf_one_season_test'), # <- info dict file name (without .pkl)
                                'model': load_model('MultiOutputClassifier_rf_one_season_test')}, # <- model file name (without .pkl)

            'LogReg':       {'info': load_info_dict('logreg_one_season_test'),
                                'model': load_model('MultiOutputClassifier_logreg_one_season_test')}
        }

# load lstm models in session state
def load_lstm_models():
    """
    if 'lstm_models' not in st.session_state:
        st.session_state['lstm_models'] = {
            'LSTM**':   {'model': appf.load_lstm_model('asdf')},
            'LSTM':     {'model': appf.load_lstm_model('jklö')}
        }
    """
    pass

def update_session_state_tradml_selections(home_id, away_id):
    st.session_state['trad_ml_home_team_select_id'] = home_id
    st.session_state['trad_ml_away_team_select_id'] = away_id

# reset flag to skip submit button when (re-)loading for the specified prediction page
def reset_skip_pred_button(for_page='trad_ml'):
    if for_page == 'trad_ml':
        st.session_state['trad_ml_skip_pred_button'] = True
    elif for_page == 'lstm':
        st.session_state['lstm_skip_pred_button'] = True
    else:
        raise ValueError(f"Unknown page '{for_page}'")

# initialize / update session state variables if required -> to be called at beginning of each page
def init_session_state(reset_trad_ml_skip_pred_button=True, reset_lstm_skip_pred_button=True):
    # db connection object
    if 'conn' not in st.session_state:
        st.session_state['conn'] = get_db_conn()
    # teams data
    if 'teams_df' not in st.session_state:
        st.session_state['teams_df'], st.session_state['teams_name2id'], st.session_state['teams_id2name'] = get_teams_data() # cached
    # tradml preds: selected team ids
    if 'trad_ml_home_team_select_id' not in st.session_state:
        update_session_state_tradml_selections(home_id=135, away_id=122) # default: Dortmund vs. Bayern
    if reset_trad_ml_skip_pred_button:
        
        reset_skip_pred_button(for_page='trad_ml')
    if reset_lstm_skip_pred_button:
        # reset flag to skip submit button when (re-)loading the (LSTM) prediction page
        reset_skip_pred_button(for_page='lstm')


#####################
# plots and tables  #
#####################

"""
# get fractional odds from probability (e.g. probability of 0.5 -> "1/1")
def get_fractional_odds(prob, rounding=1):
    pass
"""
# get moneyline odds from probability (e.g. probability of 0.5 -> "+100")
def get_moneyline_odds(prob, rounding=2):
    prob = round(prob, rounding)

    if prob == 1:
        return "-10000"
    if prob == 0:
        return "+10000"
    
    if prob <= 0.5:
        odds = (100 / (prob / 1)) - 100
        return f"+{round(odds)}"
    else:
        odds = prob*100 / (1 - prob)
        return f"-{round(odds)}"

def get_outcome_prob_plot(ypred, label_type='probability', height=None):
    # ypred is expected to be a dataframe with columns 'home_win_prob', 'draw_prob', 'away_win_prob' and a single row with the predicted values

    # prepare dataframe
    ypred = ypred.rename(columns={'home_winning_prob': 'Home Win', 'draw_prob': 'Draw', 'away_winning_prob': 'Away Win'})
    ypred = ypred.T.reset_index().rename(columns={'index': 'outcome', 0: 'probability'})
        # bar labels
    if label_type == 'percentage':
        ypred[label_type] = ypred['probability'].apply(lambda x: f"{round(x*100, 2):.2f} %")
    elif label_type == 'decimal odds':
        ypred[label_type] = ypred['probability'].apply(lambda x: f"{round(1/x, 2):.2f}")
    #elif label_type == 'fractional odds': # e.g. probability of 0.5 -> "1/1"
    #    ypred[label_type] = ypred['probability'].apply(lambda x: get_fractional_odds(prob=x, rounding=2))
    elif label_type == 'moneyline odds': # e.g. probability of 0.5 -> "+100"
        ypred[label_type] = ypred['probability'].apply(lambda x: get_moneyline_odds(prob=x, rounding=2))
    else:
        raise ValueError(f"Unknown bar label type '{label_type}'")
    
    # create plot
    plot = px.bar(ypred, x='outcome', y='probability', text=ypred[label_type])

    # plot formatting
    plot.update_layout( showlegend=False, 
                        plot_bgcolor='rgba(0, 0, 0, 0)',
                        paper_bgcolor='rgba(0, 0, 0, 0)',
                        xaxis={'showgrid': False}, 
                        yaxis={'showticklabels': False, 'zeroline': True, 'showgrid': False, 'zerolinecolor': 'gold', 'zerolinewidth': 1},
                        font={'family': 'sans-serif'},
                        coloraxis_showscale=False,
                        margin={'l': 0, 'r': 45, 't': 25, 'b': 0},
                        height=height,
                        clickmode='none', hovermode=False, dragmode=False # disable unwanted interaction
                       )

    # format x and y axis
    plot.update_xaxes(title_text='', tickfont={'size': 14})
    plot.update_yaxes(title_text='')

    # bar / label styling
    home_color = f"rgba(255,208,112, {0.55})" # green?
    draw_color = f"rgba(0,0,0, {0})" # gold w/ v. low alpha?
    away_color = f"rgba(139,0,139, {0.7})" # red?
    plot.update_traces(textposition='outside',#['outside' if v < 15 else 'inside' for v in ypred['probability']], # bar labels outside if bar is too small
                       textfont={'size': 12, 'color':'gold'},
                       cliponaxis=False, 
                       marker={#'color': [win_color if ypred['probability'].iloc[0] > ypred['probability'].iloc[2] else loss_color, # home win bar
                               #          draw_color, # draw bar
                               #          win_color if ypred['probability'].iloc[2] > ypred['probability'].iloc[0] else loss_color], # away win bar
                                'color': [home_color, draw_color, away_color], 
                                'line':{'color': 'gold', 'width': 1}},
                        width=0.75 # bar width
                        )
    return plot

def get_goals_prob_plot(goals_home_pred, goals_away_pred, home_name, away_name, height=None):
    ### prepare dataframe

    # join into one df
    df = pd.concat([goals_home_pred, goals_away_pred], axis=0, ignore_index=True)
    df = df.fillna(0) # replace NaNs with 0s (occurs only if preds have different column counts)
    # transpose and rename columns
    home_col_name = f"{home_name} (home)"
    away_col_name = f"{away_name} (away)"
    df = df.T.reset_index().rename(columns={'index': 'n_goals', 0: home_col_name, 1: away_col_name})
    
    # plot
    plot = px.bar(df, x='n_goals', y=[home_col_name, away_col_name], 
                  barmode='group', 
                  #color_discrete_sequence=['rgba(252, 181, 204, 0.5)', 'rgba(185, 219, 184, 0.5)']
                  color_discrete_sequence=['rgba(255,208,112, 0.55)', 'rgba(139,0,139, 0.75)']

                  )

    plot.update_layout( showlegend=True, 
                        plot_bgcolor='rgba(0, 0, 0, 0)',
                        paper_bgcolor='rgba(0, 0, 0, 0)',
                        xaxis={'showgrid': False, 'dtick': 1, 'title': 'number of goals'}, 
                        yaxis={'showticklabels': False, 'showgrid': False, 
                               'zeroline': True, 'zerolinecolor': 'gold', 'zerolinewidth': 1,
                               'title': 'probability'},
                        hoverlabel={'font_color': 'gold'},
                        font={'family': 'sans-serif'},
                        legend={'xanchor': 'right', 'yanchor': 'top', 'title': None},
                        margin={'l': 0, 'r': 45, 't': 25, 'b': 0},
                        height=height,
                        clickmode='none', hovermode='closest', dragmode=False # disable unwanted interaction
                        )
    plot.update_traces(marker={'line':{'color': 'gold', 'width': 1}},
                       hovertemplate='%{y:.2f}<extra></extra>')
    
    return plot



#####################
# styling utilities #
#####################

def show_app_logo_sidebar(vertical_pos='bottom'):
    st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: url(https://i.imgur.com/C8zF3Bn.png);
                background-repeat: no-repeat;
                background-size: 150px; /* Adjust the size of the image */
                padding-top: 0px; /* Adjust the padding from the top */
                padding-left: 0px; /* Adjust the padding from the left */
                background-position: center top; /* Position the image to the left top */
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def show_app_logo(width=150, use_column_width=False):
    app_logo_path = 'streamlit_app/app_logo.png'
    st.image(app_logo_path, width=width, use_column_width=True) # use_column_width takes precedence over width


# show team logo
def show_team_logo(team_id, width=None):
    logos_path = 'streamlit_app/team_logos/' # hardcoding necessary, st.image() doesn't seem to work with os.path.join() outputs...
    if os.path.isfile(os.path.join(root_path, "streamlit_app", "team_logos", f"{team_id}.png")):
        st.image(f"{logos_path}{team_id}.png", width=width)#), width=150)
    else:
        st.image(f"{logos_path}placeholder_transparent.png", width=width)#, width=150)   

# styling options for normal text (incl. font size)
def aligned_text(text, header_lvl=None, align='center', bold=False, color=None, font_size=None):

    if bold:
        text = f"<b>{text}</b>"
    if color:
        text = f"<span style='color:{color};'>{text}</span>"
    size = f"font-size:{font_size}px;" if font_size is not None else ""

    div = f'<div style="text-align: {align}; {size}">{text}</div>'
    return st.write(div, unsafe_allow_html=True)

# styling options for headers
def header_txt(text, lvl=1, align='left', color=None):
    # add span for color
    html = text
    if color:
        html = f'<span style="color:{color};">{html}</span>'
    # header tag
    html = f'<h{lvl}>{html}</h{lvl}>'
    # div for alignment
    if align:
        html = f'<div style="text-align: {align};">{html}</div>'
    return st.markdown(html, unsafe_allow_html=True)



# hide fullscreen option for images (-> should be called before plots are rendered)
def hide_image_fullscreen_option():
    css = '''
        <style>
        button[title="View fullscreen"]{
            visibility: hidden;}
        </style>
        '''
    st.markdown(css, unsafe_allow_html=True)

# keep sidebar extended
def keep_sidebar_extended():
    st.markdown("""
        <style>div[data-testid='stSidebarNav'] ul {max-height:none}</style>
        """, unsafe_allow_html=True)



