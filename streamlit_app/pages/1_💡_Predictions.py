import os
import sys

root_path = os.path.abspath(os.path.join('')) # <- adjust such that root_path always points at the root project dir (i.e. if current file is two folders deep, use '../..'). 
if root_path not in sys.path:
    sys.path.append(root_path)

import streamlit as st
import streamlit_app.app_functions as appf # <- contains functions used in our app

import models.trad_ml.training_prediction_evaluation as tpe


### page setup (visual) ###
st.set_page_config(initial_sidebar_state='expanded')
appf.hide_image_fullscreen_option()
appf.show_app_logo_sidebar(vertical_pos='top')

### session state updates ###
appf.init_session_state(reset_trad_ml_skip_pred_button=False)

# load models in session state (if not already loaded)
appf.load_trad_ml_models()

### instantiate fg object ###
fg = appf.get_feature_gen_instance()

### sidebar ###
with st.sidebar.form(key="sidebar_form", clear_on_submit=False):
    st.session_state['bar_label_type'] = st.radio(key=f"radio_bar_label_type", 
                                                  label="W/D/L prediction format:", 
                                                  horizontal=False, 
                                                  options=["percentage", 
                                                           "decimal odds", 
                                                           #"fractional odds", # todo: implement in app_functions.py
                                                           "moneyline odds"], 
                                                  index=0)
    submitted = st.form_submit_button("Change", use_container_width=True)
    if submitted:
        st.session_state['trad_ml_skip_pred_button'] = True # immediately update predictions (without requiring button click)
        
    

### header & text above selection ###
appf.header_txt("Predictions", lvl=1, align="center", color=None)
st.write('') # spacing

### team selection 
with st.form(key="team_selection", clear_on_submit=False):

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
    if 'trad_ml_skip_pred_button' in st.session_state and st.session_state['trad_ml_skip_pred_button']: 
        submitted = True
        st.session_state['trad_ml_skip_pred_button'] = False

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
        model_names = list(st.session_state['trad_ml_models'].keys())
        tabs = st.tabs(model_names)

        # define result lists
        outcome_preds, goals_home_preds, goals_away_preds = [], [], []
        for i, model_name in enumerate(model_names):

            ### prediction ### 
            # set fg params and generate features for prediction
            fg.set_params(st.session_state['trad_ml_models'][model_name]['info']['fg_config'])
            X_pred = fg.generate_features(incl_non_feature_cols=False, home_team_id=home_team_id, away_team_id=away_team_id, print_logs=False)
            predictor = tpe.ModelPrediction() # instantiate predictor object
            out, gh, ga = predictor.predict_prob(X_pred, st.session_state['trad_ml_models'][model_name]['model'], dif=False, goal_prob=True)
            outcome_preds.append(out)
            goals_home_preds.append(gh)
            goals_away_preds.append(ga)

            ### display prediction results ###
            with tabs[i]:
                with st.container():
                    subtab1, subtab2 = st.tabs(["Match Outcome", "Number of Goals"])
                    with subtab1:
                        # home win / draw / away win probabilities plot
                        st.plotly_chart(appf.get_outcome_prob_plot(outcome_preds[i], label_type=st.session_state['bar_label_type'], height=350), use_container_width=True, config={'displayModeBar': False})
                    with subtab2:
                        # n_goal distribution plot
                        st.plotly_chart(appf.get_goals_prob_plot(goals_home_preds[i], 
                                                                 goals_away_preds[i], 
                                                                 home_name=st.session_state['teams_id2name'][home_team_id], 
                                                                 away_name=st.session_state['teams_id2name'][away_team_id],
                                                                 height=350), 
                                        use_container_width=True, config={'displayModeBar': False})


### text below selection ###
appf.aligned_text(text="**:", align="left", color="#FFD700")
st.markdown("""Indicates models which were re-fitted on all available data after hyperparameter tuning. We expect these variants to perform better than their unmarked counterparts, 
               but this claim is not verifiable! The unmarked model variants were trained with the most recent season's data omitted, which enabled us to evaluate their performance.""")
st.divider()


### end of loading cycle - sidebar & other stuff ###
appf.show_app_logo_sidebar(vertical_pos='top') # sidebar logo
appf.keep_sidebar_extended()
appf.hide_image_fullscreen_option()