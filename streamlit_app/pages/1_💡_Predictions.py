import os
import sys

root_path = os.path.abspath(os.path.join('')) # <- adjust such that root_path always points at the root project dir (i.e. if current file is two folders deep, use '../..'). 
if root_path not in sys.path:
    sys.path.append(root_path)

import streamlit as st
import streamlit_app.app_functions as appf # <- contains functions used in our app

import models.trad_ml.training_prediction_evaluation as tpe
import pickle as pkl


# hide 'view fullscreen' option for images
appf.hide_image_fullscreen_option()

# load models in session state (if not already loaded)
if 'trad_ml_models' not in st.session_state:
    # define model dict (contains loaded models as well as model-specific params / info)
    st.session_state['trad_ml_models'] = {
        'XGBoost (1 test season)':  {'info': appf.load_info_dict('xgb_one_season_test'), # <- info dict file name (without .pkl)
                                     'model': appf.load_model('MultiOutputClassifier_xgb_one_season_test')}, # <- model file name (without .pkl)
        
        'XGBoost (no split)':       {'info': appf.load_info_dict('xgb_all_train'),
                                     'model': appf.load_model('MultiOutputClassifier_xgb_all_train')},
        
        'LogReg (1 test season)':   {'info': appf.load_info_dict('logreg_one_season_test'),
                                     'model': appf.load_model('MultiOutputClassifier_logreg_one_season_test')},
        
        'LogReg (no split)':        {'info': appf.load_info_dict('logreg_all_train'),
                                     'model': appf.load_model('MultiOutputClassifier_logreg_all_train')}
        

    }

### definitions ###
team_select_options = st.session_state['teams_df'].sort_values(['country', 'name'])['id'].tolist()
team_select_format_func = lambda id: f"({st.session_state['teams_df'].query(f'id=={id}')['country'].values[0]}) {st.session_state['teams_id2name'][id]}"

### instantiate fg object ###
fg = appf.get_feature_gen_instance()

### team selection 
# get team display strings for selectboxes
team_display_strings = st.session_state['teams_df']['name'].tolist()
st.write("# Header - Predictions using traditional ML models")

with st.form(key="team_selection", clear_on_submit=False):
    #appf.aligned_text("Select teams for prediction:")
    c1, c2, c3 = st.columns(3)
    with c1:
        home_team_id = st.selectbox(
            label="Home team:",
            options=team_select_options,
            format_func=team_select_format_func,
            index=7
        )
    with c3:
        away_team_id = st.selectbox(
            label="Away team:",
            options=team_select_options,
            format_func=team_select_format_func,
            index=16
        )

    # disallow same team as home and away
    if home_team_id == away_team_id:
        st.error("Please choose two different teams.")

    # submit button
    submitted = st.form_submit_button("Predict Matchup", use_container_width=True)

    if submitted and home_team_id != away_team_id:
        
        # display name & logo of selected teams
        with st.container():
            c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(spec=[1, 1, 0.5, 0.5, 1, 0.5, 0.5, 1, 1]) # columns for image positioning
            with c2:
                # home team logo (if available)
                
                appf.show_logo(home_team_id)
            with c5:
                st.write("") # empty column for spacing
                appf.aligned_text("vs.", align="center", bold=True)#, color="#FFD700")            
            with c8:
                appf.show_logo(away_team_id)

        model_names = list(st.session_state['trad_ml_models'].keys())
        tabs = st.tabs(model_names)
        for i, model_name in enumerate(model_names):

            ### prediction ### 
            # set fg params and generate features for prediction
            fg.set_params(st.session_state['trad_ml_models'][model_name]['info']['fg_config'])
            X_pred = fg.generate_features(incl_non_feature_cols=False, home_team_id=home_team_id, away_team_id=away_team_id, print_logs=False)
            predictor = tpe.ModelPrediction() # instantiate predictor object
            outcome_pred, goals_home_pred, goals_away_pred = predictor.predict_prob(X_pred, st.session_state['trad_ml_models'][model_name]['model'], dif=False, goal_prob=True)

            ### display prediction results ###
            with tabs[i]:
                with st.container():
                    subtab1, subtab2 = st.tabs(["Match Outcome", "Number of Goals"])
                    with subtab1:
                        # home win / draw / away win probabilities plot
                        st.plotly_chart(appf.get_outcome_prob_plot(outcome_pred, height=300), use_container_width=True, config={'displayModeBar': False})
                    with subtab2:
                        # n_goal distribution plot
                        st.plotly_chart(appf.get_goals_prob_plot(goals_home_pred, goals_away_pred, height=300), use_container_width=True, config={'displayModeBar': False})




