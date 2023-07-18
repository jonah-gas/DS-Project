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

#####################
# plots and tables  #
#####################

def get_outcome_prob_plot(ypred, height=None):
    # ypred is expected to be a dataframe with columns 'home_win_prob', 'draw_prob', 'away_win_prob' and a single row with the predicted values

    # prepare dataframe
    ypred = ypred.rename(columns={'home_winning_prob': 'Home Win', 'draw_prob': 'Draw', 'away_winning_prob': 'Away Win'})
    ypred = ypred.T.reset_index().rename(columns={'index': 'outcome', 0: 'probability'})
    ypred['probability'] = ypred['probability'].apply(lambda x: round(x*100, 2))

    plot = px.bar(ypred, x='outcome', y='probability')

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

    # bar styling and labels
    bar_alpha = 0.75
    win_color = f"rgba(127,255,0, {bar_alpha})" # green?
    draw_color = f"rgba(255,208,112, {0.1})" # gold w/ v. low alpha?
    loss_color = f"rgba(220,20,60, {bar_alpha})" # red?
    plot.update_traces(texttemplate='%{y:.2f}%', 
                       textposition='outside',#['outside' if v < 15 else 'inside' for v in ypred['probability']], # bar labels outside if bar is too small
                       textfont={'size': 12, 'color':'gold'},
                       cliponaxis=False, 
                       marker={'color': [win_color if ypred['probability'].iloc[0] > ypred['probability'].iloc[2] else loss_color, # home win bar
                                         draw_color, # draw bar
                                         win_color if ypred['probability'].iloc[2] > ypred['probability'].iloc[0] else loss_color], # away win bar
                                'line':{'color': 'gold', 'width': 1}},
                        width=0.75 # bar width
                        )
    return plot

def get_goals_prob_plot(goals_home_pred, goals_away_pred, height=None):
    ### prepare dataframe

    # join into one df
    df = pd.concat([goals_home_pred, goals_away_pred], axis=0, ignore_index=True)
    df = df.fillna(0) # replace NaNs with 0s (occurs only if preds have different column counts)
    # transpose and rename columns
    df = df.T.reset_index().rename(columns={'index': 'n_goals', 0: 'Home', 1: 'Away'})
    
    # plot
    plot = px.bar(df, x='n_goals', y=['Home', 'Away'], 
                  barmode='group', color_discrete_sequence=['lightblue', 'purple'])

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

# show logo
def show_logo(team_id, width=None):
    logos_path = 'streamlit_app/team_logos/' # hardcoding necessary, st.image() doesn't seem to work with os.path.join() outputs...
    if os.path.isfile(os.path.join(root_path, "streamlit_app", "team_logos", f"{team_id}.png")):
        st.image(f"{logos_path}{team_id}.png", width=width)#), width=150)
    else:
        st.image(f"{logos_path}placeholder_transparent.png", width=width)#, width=150)   

# aligned text
def aligned_text(text, align='center', bold=False, color=None, font_size=None):
    if bold:
        text = f"<b>{text}</b>"
    if color:
        text = f"<span style='color:{color};'>{text}</span>"
    if font_size:
        text = f"<span style='font-size:{font_size};'>{text}</span>"
    return st.markdown(f'<div style="text-align: {align};">{text}</div>', unsafe_allow_html=True)

# hide fullscreen option for images (-> should be called before plots are rendered)
def hide_image_fullscreen_option():
    css = '''
        <style>
        button[title="View fullscreen"]{
            visibility: hidden;}
        </style>
        '''
    st.markdown(css, unsafe_allow_html=True)






