import os
import sys

root_path = os.path.abspath(os.path.join('')) # <- adjust such that root_path always points at the root project dir (i.e. if current file is two folders deep, use '../..'). 
if root_path not in sys.path:
    sys.path.append(root_path)

import streamlit as st
import streamlit_app.app_functions as appf # <- contains functions used in our app

# import lstm modules
import models.neural_net.LSTM_prediction as lstm_pred
from models.neural_net.gru_models import Sport_pred_1GRU_3, Sport_pred_2GRU_1

### page setup (visual) ###
st.set_page_config(initial_sidebar_state='expanded')
appf.hide_image_fullscreen_option()
appf.show_app_logo_sidebar(vertical_pos='top')

### session state updates ###
appf.init_session_state(reset_lstm_skip_pred_button=False)

### load model(s) in session state (if not already loaded)
appf.load_lstm_models()

### get required objects for LSTM ###
#clubs, rearrange_list, scale_df, result_dict = appf.call_lstm_setup
clubs, rearrange_list, scale_df, result_dict, venue_dict = appf.load_lstm_setup_files()

### sidebar ###
bar_label_type_options = ["percentage", "decimal odds", "moneyline odds"]
with st.sidebar.form(key="lstm_sidebar_form", clear_on_submit=False):
    st.session_state['bar_label_type'] = st.radio(key=f"lstm_radio_bar_label_type", 
                                                  label="W/D/L prediction format:", 
                                                  horizontal=False, 
                                                  options=bar_label_type_options, 
                                                  index=bar_label_type_options.index(st.session_state['bar_label_type']))
    submitted = st.form_submit_button("Change", use_container_width=True)
    if submitted:
        st.session_state['lstm_skip_pred_button'] = True # immediately update predictions (without requiring button click)
    

### header & text above selection ###
appf.header_txt("Predictions (LSTM)", lvl=1, align="center", color=None)
st.write('') # spacing

st.warning("⚠️ Caution: LSTM models are still in development, the current version might yield questionable predictions. Please refer to the traditional ML models for now.")


### team selection 
with st.form(key="lstm_team_selection", clear_on_submit=False):

    # define selection box options / formatting
    team_select_options = st.session_state['teams_df'].sort_values(['country', 'name'])['id'].tolist()
    team_select_format_func = lambda id: f"({st.session_state['teams_df'].query(f'id=={id}')['country'].values[0]}) {st.session_state['teams_id2name'][id]}"

    c1, c2, c3 = st.columns(3)
    with c1:
        home_team_id = st.selectbox(
            label="Home team:",
            options=team_select_options,
            format_func=team_select_format_func,
            index=team_select_options.index(st.session_state['trad_ml_home_team_select_id'])
        )

    with c3:
        away_team_id = st.selectbox(
            label="Away team:",
            options=team_select_options,
            format_func=team_select_format_func,
            index=team_select_options.index(st.session_state['trad_ml_away_team_select_id'])
        )

    # disallow same team as home and away
    if home_team_id == away_team_id:
        st.error("Please choose two different teams.")

    # submit button
    submitted = st.form_submit_button("Predict Matchup", use_container_width=True)
    if 'lstm_skip_pred_button' in st.session_state and st.session_state['lstm_skip_pred_button']: 
        submitted = True
        st.session_state['lstm_skip_pred_button'] = False

    if submitted and home_team_id != away_team_id:
        # enter selected team ids into session state
        appf.update_session_state_tradml_selections(home_team_id, away_team_id)

        # display names & logos of selected teams
        with st.container():
            c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(spec=[1, 1, 0.5, 0.5, 1, 0.5, 0.5, 1, 1]) # columns for image positioning
            with c2:
                appf.show_team_logo(home_team_id)
            with c5:
                st.write("") # empty column for spacing
                appf.aligned_text("vs.", align="center", bold=True)#, color="#FFD700")            
            with c8:
                appf.show_team_logo(away_team_id)

        # create model tabs
        model_names = list(st.session_state['lstm_models'].keys())
        tabs = st.tabs(model_names)

        # define result lists
        outcome_preds = []
        for i, model_name in enumerate(model_names):

            ### prediction ### 

            pred_df = lstm_pred.sequence_models(model=st.session_state['lstm_models'][model_name]['model'],
                                                team1=home_team_id,
                                                team2=away_team_id,
                                                clubs=clubs,
                                                rearrange_list=rearrange_list,
                                                scale_df=scale_df,
                                                result_dict=result_dict,
                                                venue_dict=venue_dict)
            outcome_preds.append(pred_df)
            ### display prediction results ###
            with tabs[i]:
                with st.container():
                    # home win / draw / away win probabilities plot
                    st.plotly_chart(appf.get_outcome_prob_plot(outcome_preds[i], label_type=st.session_state['bar_label_type'], height=350), use_container_width=True, config={'displayModeBar': False})

            

### text below selection ###

#appf.aligned_text(text="**:", align="left", color="#FFD700")
#st.markdown("""Indicates models which were re-fitted on all available data after hyperparameter tuning. We expect these variants to perform better than their unmarked counterparts, 
#               but this claim is not verifiable! The unmarked model variants were trained with the most recent season's data omitted, which enabled us to evaluate their performance.""")
#st.divider()


### end of loading cycle - sidebar & other stuff ###

appf.keep_sidebar_extended()
appf.hide_image_fullscreen_option()

