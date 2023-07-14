import os
import sys

root_path = os.path.abspath(os.path.join('..')) # <- adjust such that root_path always points at the root project dir (i.e. if current file is two folders deep, use '../..'). 
if root_path not in sys.path:
    sys.path.append(root_path)

import streamlit as st
import streamlit_app.app_functions as appf # <- contains functions used in our app

import models.trad_ml.training_prediction_evaluation as tpe
import pickle as pkl


# debug
st.write(f"root_path: {root_path}")

### definitions ###
team_select_options = st.session_state['teams_df'].sort_values(['country', 'name'])['id'].tolist()
team_select_format_func = lambda id: f"({st.session_state['teams_df'].query(f'id=={id}')['country'].values[0]}) {st.session_state['teams_id2name'][id]}"

### instantiate fg object ###
fg = appf.get_feature_gen_instance()
# set params (loads data prep objects
results_dict = pkl.load(open(os.path.join(root_path, "models", "trad_ml", "sweep_results", "logreg_test_for_app_1_dodsdf.pkl"), "rb"))
fg_params = results_dict['fg_config']
# adjust data prep object names
fg_params['scaler_name'] = 'StandardScaler_dodsdf'
fg_params['pca_name'] = 'PCA_dodsdf'

st.write(f"fg dpo path: {fg.data_prep_objects_path}") # DEBUG
fg.set_params(fg_params)



### load model ###
model = pkl.load(open(os.path.join(root_path, "models", "trad_ml", "saved_models", "MultiOutputClassifier_dodsdf.pkl"), "rb"))

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
            index=1
        )
    with c3:
        away_team_id = st.selectbox(
            label="Away team:",
            options=team_select_options,
            format_func=team_select_format_func,
            index=2
        )

    # disallow same team as home and away
    if home_team_id == away_team_id:
        st.error("Home team and away team cannot be the same.")

    # submit button
    submitted = st.form_submit_button("Predict Matchup", use_container_width=True)

    if submitted and home_team_id != away_team_id:
        ### prediction ###
        # generate prediction features
        X_pred = fg.generate_features(incl_non_feature_cols=False, home_team_id=home_team_id, away_team_id=away_team_id, print_logs=False)
        # predict

        # instantiate prediction instance
        predictor = tpe.ModelPrediction()
        y_pred = predictor.predict_prob(X_pred, model, dif=False)
        
        # display name & logo of selected teams
        with st.container():
            c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(9) # columns for image positioning
            with c2:
                # home team logo (if available)
                if os.path.isfile(os.path.join(root_path, "streamlit_app", "team_logos", f"{home_team_id}.png")):
                    st.image(f'team_logos/{home_team_id}.png')#), width=150)
                else:
                    st.image(f'team_logos/placeholder_transparent.png')#, width=150)    
            with c5:
                st.write("") # empty column for spacing
                appf.aligned_text("vs.", align="center", bold=True, color="#FFD700")            
            with c8:
                # away team logo (if available)
                if os.path.isfile(os.path.join(root_path, "streamlit_app", "team_logos", f"{away_team_id}.png")):
                    st.image(f'team_logos/{away_team_id}.png')#, width=150)
                else:
                    st.image(f'team_logos/placeholder_transparent.png')#, width=150)
                
        st.divider()
        # display prediction results
        with st.container():
            c1, c2, c3 = st.columns(3)
            with c1:
                appf.aligned_text(f"Home win: {round(y_pred['home_winning_prob'].values[0]*100, 2)}%", align="center")
            with c2:
                appf.aligned_text(f"Draw: {round(y_pred['draw_prob'].values[0]*100, 2)}%", align="center")
            with c3:
                appf.aligned_text(f"Away win: {round(y_pred['away_winning_prob'].values[0]*100, 2)}%", align="center")
        

