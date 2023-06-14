# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:29:27 2023

@author: johan_nii2lon
"""
import os
os.chdir("C:/Users/johan_nii2lon/OneDrive/Jonah/Master/2_Semester/Data_Science_Project")
import numpy as np
import pandas as pd
from data_cleaning import DataCleaning
import glob
import time

'''LOAD IN ALL DATA'''

fbref_matchdata_rawcleaning = pd.read_excel("fbref_matchdata_rawcleaning.xlsx")
# Specify the directory path where your CSV files are located
directory_path = 'data/scraped/fbref/match'
# Use glob to get a list of all CSV files in the directory
csv_files = glob.glob(directory_path + '/*.csv')






'''CLEAN DATA AND CONCAT'''

# Create an empty list to store the cleaned data
clean_data = []
# Iterate over the CSV files
for file in csv_files:
    # Read the CSV file into a DataFrame
    data_raw = pd.read_csv(file, delimiter=";")

    # Clean the raw data
    cleaned_data = DataCleaning().clean_raw_data(data_raw, fbref_matchdata_rawcleaning)

    # Reset the index of the cleaned data DataFrame
    cleaned_data.reset_index(drop=True, inplace=True)

    # Append the cleaned data to the list
    clean_data.append(cleaned_data)
# Concatenate all the cleaned data into a single DataFrame
clean_data = pd.concat(clean_data, ignore_index=True)





'''CREATE POINT VARIABLE'''

# create New Column
clean_data["points"] = pd.np.where(clean_data["Result"] == "W", "3",pd.np.where(clean_data["Result"]== "D","1", "0"))
clean_data["points"] = clean_data["points"].astype(float)
print(clean_data.dtypes)




'''CREATE MOVING AVERAGES FEATURES'''

# Sort the dataframe by 'fbref_squad_id' and 'schedule_Date'
df3 = clean_data.sort_values(['fbref_squad_id', 'schedule_Date'])

# Store the column names that are of type float64
float_columns = df3.select_dtypes(include=['float64']).columns

# Specify the window size and decay factor
window_size = 7
decay_factor = 0.65

# Calculate the moving averages excluding the current game for each group
moving_averages_replacements = df3.groupby('fbref_squad_id')[float_columns].apply(lambda x: x.iloc[:-1].ewm(alpha=1 - decay_factor, min_periods=window_size).mean())

# Reset the index of the moving averages dataframe
moving_averages_replacements = moving_averages_replacements.reset_index(level=0, drop=True)

# Merge the moving averages back to the original dataframe and drop original columns
df3 = df3.drop(columns=float_columns).join(moving_averages_replacements.add_suffix('_avg'))


# Do the same for goals but do not drop the current date, since they are needed for prediction
df3["GF"] = df3["GF"].astype(float)
df3["GA"] = df3["GA"].astype(float)


# Calculate the moving averages excluding the current game for each group
moving_averages_to_keep = df3.groupby('fbref_squad_id')["GF", "GA"].apply(lambda x: x.iloc[:-1].ewm(alpha=1 - decay_factor, min_periods=window_size).mean())

# Reset the index of the moving averages dataframe
moving_averages_to_keep = moving_averages_to_keep.reset_index(level=0, drop=True)

# Merge the moving averages back to the original dataframe
df3 = pd.concat([df3, moving_averages_to_keep.add_suffix('_avg')], axis=1)

# Drop the first seven observations for each 'fbref_season' within each team
df3 = df3.groupby(['fbref_squad_id', 'fbref_season']).apply(lambda x: x.iloc[7:]).reset_index(drop=True)

print(df3.columns)

import re
matching_columns = [col for col in df3.columns if re.search("GF", col)]
print(matching_columns)






'''MERGE ROWS OF THE SAME GAME'''

# Sort the dataframe again by 'fbref_squad_id', 'schedule_Date', and 'Referee' need for merge_data function
df3 = df3.sort_values(['fbref_squad_id', 'schedule_Date', 'Referee'], ascending=[True, True, True])
merged_df3 = DataCleaning().merge_data(df3)
clean_df3 = DataCleaning().clean_merged_data(merged_df3)





'''ENCODE DUMMY VARIABLES'''

#import Encoder
from sklearn.preprocessing import OneHotEncoder

# Select the non-numeric columns using select_dtypes
non_numeric_columns = clean_df3.select_dtypes(exclude='number').columns
# Of non-numeric (print to view) only encode schedule_time, fbref_home_id, fbref_away_id, schedule_round
desired_values = ['schedule_round', 'schedule_day', 'fbref_home_id', 'fbref_away_id']

# Make a subset of the index based on the desired values
non_numeric_columns = non_numeric_columns[non_numeric_columns.isin(desired_values)]

encoder = OneHotEncoder()

# Fit and transform the non-numeric columns using OneHotEncoder
encoded_features = encoder.fit_transform(clean_df3[non_numeric_columns])

# Convert the encoded features to a DataFrame
encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names(non_numeric_columns))

# Reset the index of encoded_df
encoded_df.reset_index(drop=True, inplace=True)

# Concatenate the original DataFrame with the encoded DataFrame
clean_df3 = pd.concat([clean_df3, encoded_df], axis=1)

# Delete all NA
clean_df3 = clean_df3.dropna()






'''Check the distributions of results'''

# Count total number of results
home_wins = clean_df3['result_home'].value_counts()['W']
draw = clean_df3['result_home'].value_counts()['D']
home_loss = clean_df3['result_home'].value_counts()['L']

#Compute percentages
homewin_perc = home_wins/len(clean_df3)
draw_perc = draw/len(clean_df3)
homeloss_perc = home_loss/len(clean_df3)






'''Split data into train and test respectivly X and Y'''

from sklearn.model_selection import train_test_split

# Sort data by date for splitting into train and test data.
clean_df3.sort_values(by='schedule_date', inplace=True)
clean_df3.reset_index(drop=True, inplace=True)
clean_df3.set_index('schedule_date', inplace=True)

# Create feature matrix
columns_to_drop = ['gf_home', 'gf_away']
X = clean_df3.drop(columns_to_drop, axis="columns")
numeric_columns = X.select_dtypes(include='number').columns
X = X[numeric_columns]

# Create target variables matrix
y = clean_df3[["gf_home", "gf_away"]]
y = y.astype(int)

# Specify the split date
split_date = "2022-05-23"

# Split the data into training and test sets based on the split date
X_train = X[X.index < split_date]
y_train = y[X.index < split_date]

X_test = X[X.index >= split_date]
y_test = y[y.index >= split_date]




'''Just For Fun Random Forest Regressor'''

from sklearn.ensemble import RandomForestRegressor

# Create an instance of the Random Forest model
rf_model = RandomForestRegressor()

# Train the model 
rf_model.fit(X_train, y_train)

# Make the predictions
y_pred = rf_model.predict(X_test)

# compute real result ("W", "D", or "L")
y_test['result_home_perspective'] = pd.np.where(y_test['gf_home'] - y_test['gf_away'] > 0.5, 'W',
                                                pd.np.where(y_test['gf_away'] - y_test['gf_home'] > 0.5, 'L', 'D'))
# compute predicted result ("W", "D", or "L") basedon predicted goals
y_pred = pd.DataFrame(y_pred, columns=['gf_home', "gf_away"])
y_pred['result_home_perspective'] = pd.np.where(y_pred['gf_home'] - y_pred['gf_away'] > 0.2, 'W',
                                                pd.np.where(y_pred['gf_away'] - y_pred['gf_home'] > 0.2, 'L', 'D'))
#compare real result to predicted result
y_pred.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
y_pred['real_result'] = y_test['result_home_perspective']
count = (y_test["result_home_perspective"].values == y_pred["result_home_perspective"].values).sum()

#compute and print accuracy
accuracy = count/len(y_pred)
print(accuracy)

# get feature importance and stor them into a data frame.
feature_importances = rf_model.feature_importances_
feature_names = X.columns
df_feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# sort the DataFrame by importance values in descending order
df_feature_importances = df_feature_importances.sort_values(by='Importance', ascending=False)
print(df_feature_importances)






'''Random Forest Classifier'''

from sklearn.ensemble import RandomForestClassifier
# Create an instance of the random forest regressor
rf_model_prob = RandomForestClassifier()

# Fit the model to the training data
rf_model_prob.fit(X_train, y_train)
y_pred_prob = rf_model_prob.predict_proba(X_test)

# Concatenate the two lists along the columns axis
combined_prob = np.concatenate(y_pred_prob, axis=1)

# Create a DataFrame from the combined probabilities
df_pred_prob = pd.DataFrame(combined_prob, columns=['Column{}'.format(i+1) for i in range(combined_prob.shape[1])])



# Devide date into two dataframes
goal_prob_home = df_pred_prob.iloc[:, :10]
goal_prob_away = df_pred_prob.iloc[:, 10:]

# Change column names
new_columns = ['prob_{}_goal_home'.format(i+1) for i in range(goal_prob_home.shape[1])]
goal_prob_home.columns = new_columns

new_columns = ['prob_{}_goal_away'.format(i+1) for i in range(goal_prob_away.shape[1])]
goal_prob_away.columns = new_columns

#COMPUTE THE PROBABILITES OF HOMEWIN, AWAYWIN AND DRAW AND ADD TO Y_TEST DF
# Get the number of goals columns for home and away teams
num_goals_home = goal_prob_home.shape[1]
num_goals_away = goal_prob_away.shape[1]

# Create a new data frame to store the results
winning_prob_df = pd.DataFrame(index=goal_prob_home.index)

# Create three arrays for the propabilitis
home_winning_prob = pd.Series(0.0, index=goal_prob_home.index)

away_winning_prob = pd.Series(0.0, index=goal_prob_home.index)

draw_prob = pd.Series(0.0, index=goal_prob_home.index)

# Iterate over all possible goal combinations
for home_goals in range(num_goals_home):
    for away_goals in range(num_goals_away):
        # Home team wins if home goals > away goals
        if home_goals > away_goals:
            home_winning_prob += goal_prob_home.iloc[:, home_goals] * goal_prob_away.iloc[:, away_goals]
        # Away team wins if home goals < away goals
        elif home_goals < away_goals:
            away_winning_prob += goal_prob_home.iloc[:, home_goals] * goal_prob_away.iloc[:, away_goals]
        # Match ends in a draw if home goals = away goals
        else:
            draw_prob += goal_prob_home.iloc[:, home_goals] * goal_prob_away.iloc[:, away_goals]


# Add the probabilities to the y_test
y_test['home_winning_prob'] = home_winning_prob
y_test['draw_prob'] = draw_prob
y_test['away_winning_prob'] = away_winning_prob



'''XgBoost Classifier'''


from xgboost import XGBClassifier

     
# Filter out instances with class value 9, to get rid of outlier (necessary for this algorithm)
filtered_indices = y_train.iloc[:, 1] != 9
X_train_filtered = X_train[filtered_indices]
y_train_filtered = y_train.iloc[:, 1][filtered_indices]

# Train the XGBoost classifier with the filtered data
xgb_model_prob_away = XGBClassifier(objective='binary:logistic')
xgb_model_prob_away.fit(X_train_filtered, y_train_filtered)

# Train the XGBoost Classifier for home team goal probabilities
xgb_model_prob_home = XGBClassifier(objective='binary:logistic')
xgb_model_prob_home.fit(X_train, y_train.iloc[:, 0])


# Predict class probabilities with XGBoost for home and away teams
y_pred_prob_home_xgb = xgb_model_prob_home.predict_proba(X_test)
y_pred_prob_away_xgb = xgb_model_prob_away.predict_proba(X_test)

xgb_goal_prob_home = pd.DataFrame(y_pred_prob_home_xgb)
xgb_goal_prob_away = pd.DataFrame(y_pred_prob_away_xgb)

# Change column names
new_columns = ['prob_{}_goal_home'.format(i) for i in range(xgb_goal_prob_home.shape[1])]
xgb_goal_prob_home.columns = new_columns

new_columns = ['prob_{}_goal_away'.format(i) for i in range(xgb_goal_prob_away.shape[1])]
xgb_goal_prob_away.columns = new_columns

#COMPUTE THE PROBABILITES OF HOMEWIN, AWAYWIN AND DRAW AND ADD TO Y_TEST DF
# Get the number of goals columns for home and away teams
xgb_num_goals_home = xgb_goal_prob_home.shape[1]
xgb_num_goals_away = xgb_goal_prob_away.shape[1]

# Create a new DataFrame to store the winning probabilities
xgb_winning_prob_df = pd.DataFrame(index=xgb_goal_prob_home.index)

# Create three arrays for the propabilitis
xgb_home_winning_prob = pd.Series(0.0, index=xgb_goal_prob_home.index)

xgb_away_winning_prob = pd.Series(0.0, index=xgb_goal_prob_home.index)

xgb_draw_prob = pd.Series(0.0, index=xgb_goal_prob_home.index)




xgb_home_winning_prob = 0
xgb_away_winning_prob = 0
xgb_draw_prob = 0


# Iterate over all possible goal combinations
for xgb_home_goals in range(xgb_num_goals_home):
    for xgb_away_goals in range(xgb_num_goals_away):
        # Home team wins if home goals > away goals
        if xgb_home_goals > xgb_away_goals:
            xgb_home_winning_prob += xgb_goal_prob_home.iloc[:, xgb_home_goals] * xgb_goal_prob_away.iloc[:, xgb_away_goals]
        # Away team wins if home goals < away goals
        elif xgb_home_goals < xgb_away_goals:
            xgb_away_winning_prob += xgb_goal_prob_home.iloc[:, xgb_home_goals] * xgb_goal_prob_away.iloc[:, xgb_away_goals]
        # Match ends in a draw if home goals = away goals
        else:
            xgb_draw_prob += xgb_goal_prob_home.iloc[:, xgb_home_goals] * xgb_goal_prob_away.iloc[:, xgb_away_goals]

# Add the probabilities to the y_test DataFrame
y_test['xgb_home_winning_prob'] = xgb_home_winning_prob
y_test['xgb_draw_prob'] = xgb_draw_prob
y_test['xgb_away_winning_prob'] = xgb_away_winning_prob

print(xgb_goal_prob_home)
print(xgb_goal_prob_away)



# Retrieve feature importances
xgb_importances_home = xgb_model_prob_home.feature_importances_
xgb_importances_away =xgb_model_prob_away .feature_importances_

# Create a DataFrame for feature importances
feature_importances_xgb_home = pd.DataFrame({'Feature': X_train.columns, 'Importance': xgb_importances_home})
feature_importances_xgb_away = pd.DataFrame({'Feature': X_train.columns, 'Importance': xgb_importances_away})





'''Evaluation of the Algorithms with Counterprobs'''

#COMPUTE "ACCURACY" OF OWN PROBABILITIES
#Random Forest
def evaluate_accuracy(row):
    if row['result_home_perspective'] == 'W':
        return 1 - row['home_winning_prob']
    elif row['result_home_perspective'] == 'D': 
        return 1 - row['draw_prob']
    elif row['result_home_perspective'] == 'L':
        return 1 - row['away_winning_prob']
    else:
        return None

y_test['Accuracy'] = y_test.apply(evaluate_accuracy, axis=1)
accuracy_rate = 1-y_test['Accuracy'].mean()


#xgb 
def evaluate_accuracy(row):
    if row['result_home_perspective'] == 'W':
        return 1 - row['xgb_home_winning_prob']
    elif row['result_home_perspective'] == 'D': 
        return 1 - row['xgb_draw_prob']
    elif row['result_home_perspective'] == 'L':
        return 1 - row['xgb_away_winning_prob']
    else:
        return None

# Applying the evaluate_accuracy function to each row
y_test['xgb_Accuracy'] = y_test.apply(evaluate_accuracy, axis=1)

# Calculating accuracy rate
accuracy_rate_xgb = 1-y_test['xgb_Accuracy'].mean()



'''Evaltuation of Modells with Log loss function'''

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

#get true values
true_labels = y_test["result_home_perspective"]
# Encode the true labels using LabelEncoder
label_encoder = LabelEncoder()
true_labels_encoded = label_encoder.fit_transform(true_labels)
#Convert true labels to numpy arrays
true_labels_encoded = np.array(true_labels_encoded)

# get rf predictions for rf
predicted_probs_df_rf = y_test[['draw_prob', 'away_winning_prob', 'home_winning_prob']]
# Convert the true labels and predicted probabilities to numpy arrays
predicted_probs_rf = np.array(predicted_probs_df_rf)
# Calculate the log loss for rf
loss_rf = log_loss(true_labels_encoded, predicted_probs_rf)

#get xgb predicitions
predicted_probs_df_xgb = y_test[['xgb_draw_prob', 'xgb_away_winning_prob', 'xgb_home_winning_prob']]
# Encode the true labels using LabelEncoder
predicted_probs_xgb = np.array(predicted_probs_df_xgb)
# Calculate the log loss for xgb
loss_xgb = log_loss(true_labels_encoded, predicted_probs_xgb)





