# -*- coding: utf-8 -*-
"""
Created on Tue May 23 17:13:43 2023

@author: johan_nii2lon
"""
import os 
import pandas as pd
import numpy as np
os.chdir("C:/Users/johan_nii2lon/OneDrive/Jonah/Master/2_Semester/Data_Science_Project/")

data = pd.read_csv("E0.csv")

# compute real percentages
data["B365H_prob_real"] = 100/data["B365H"] 
data["B365D_prob_real"] = 100/data["B365D"] 
data["B365A_prob_real"] = 100/data["B365A"] 

#get factor from real to fair odds
data["factor"] = (data["B365H_prob_real"]+data["B365D_prob_real"]+data["B365A_prob_real"])

#compute fair percentages
data["B365H_prob_fair"] =  (1/data["factor"])* data["B365H_prob_real"]
data["B365D_prob_fair"] =  (1/data["factor"])* data["B365D_prob_real"]
data["B365A_prob_fair"] =  (1/data["factor"])* data["B365A_prob_real"]

#plausability check wether no perc add up to one
print(data["B365H_prob_fair"]+data["B365D_prob_fair"]+data["B365A_prob_fair"])


#loss function: if they are right they get the counter prob of odd. So high odds on the right result give you a low value and high odds but the wrong result leed to big loss
def evaluate_accuracy(row):
    if row['FTR'] == 'H':
        return 1 - row['B365H_prob_fair']
    elif row['FTR'] == 'D': 
        return 1 - row['B365D_prob_fair']
    elif row['FTR'] == 'A':
        return 1 - row['B365A_prob_fair']
    else:
        return None

# Applying the evaluate_accuracy function to each row
data['Accuracy'] = data.apply(evaluate_accuracy, axis=1)

#since for our "accuracy" a low value is good we need compute minus one
# Calculating accuracy rate
accuracy_rate_bookmakers = 1-data['Accuracy'].mean()




from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import numpy as np

# Example true labels and predicted probabilities
true_labels = data["FTR"]
predicted_probs_df = data[['B365A_prob_fair', 'B365D_prob_fair', 'B365H_prob_fair']]
# Encode the true labels using LabelEncoder
label_encoder = LabelEncoder()
true_labels_encoded = label_encoder.fit_transform(true_labels)

# Convert the true labels and predicted probabilities to numpy arrays
true_labels_encoded = np.array(true_labels_encoded)
predicted_probs = np.array(predicted_probs_df)

# Calculate the log loss
loss_book = log_loss(true_labels_encoded, predicted_probs)

print(label_encoder.classes_)
print("Log Loss:", loss_book)
