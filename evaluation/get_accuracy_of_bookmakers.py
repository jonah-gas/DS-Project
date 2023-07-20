# -*- coding: utf-8 -*-
import os 
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

root_path = os.path.abspath(os.path.join('..')) # <- adjust such that root_path always points at the root project dir (i.e. if current file is two folders deep, use '../..'). 
if root_path not in sys.path:
    sys.path.append(root_path)


data = pd.read_csv(root_path, "/data/scraped/football_data_uk/betting_odds/bookmaker_odds_2022_2023.csv")


bookmakers = ["B365", "BW", "IW", "PS",]

# Create an empty dataframe to store the accuracies and log loss
accuracy_bookmakers_df = pd.DataFrame(columns=['Bookmaker', 'Accuracy', 'Log Loss'])

for bookmaker in bookmakers:
    data[bookmaker + "H_prob_real"] = 100 / data[bookmaker + "H"]
    data[bookmaker + "D_prob_real"] = 100 / data[bookmaker + "D"]
    data[bookmaker + "A_prob_real"] = 100 / data[bookmaker + "A"]

    # Get factor from real to fair odds
    data["factor"] = data[bookmaker + "H_prob_real"] + data[bookmaker + "D_prob_real"] + data[bookmaker + "A_prob_real"]

    # Compute fair percentages
    data[bookmaker + "H_prob_fair"] = (1 / data["factor"]) * data[bookmaker + "H_prob_real"]
    data[bookmaker + "D_prob_fair"] = (1 / data["factor"]) * data[bookmaker + "D_prob_real"]
    data[bookmaker + "A_prob_fair"] = (1 / data["factor"]) * data[bookmaker + "A_prob_real"]

    # Plausibility check whether no percentage adds up to one
    print(data[bookmaker + "H_prob_fair"] + data[bookmaker + "D_prob_fair"] + data[bookmaker + "A_prob_fair"])

    # Loss function: if they are right, they get the counter probability of the odd.
    # So high odds on the right result give you a low value, and high odds but the wrong result lead to a big loss.
    def evaluate_accuracy(row):
        max_prob = max(row[bookmaker + 'H_prob_fair'], row[bookmaker + 'D_prob_fair'], row[bookmaker + 'A_prob_fair'])

        if row['FTR'] == 'H' and max_prob == row[bookmaker + 'H_prob_fair']:
            return 1
        elif row['FTR'] == 'D' and max_prob == row[bookmaker + 'D_prob_fair']:
            return 1
        elif row['FTR'] == 'A' and max_prob == row[bookmaker + 'A_prob_fair']:
            return 1
        else:
            return 0

    # Applying the evaluate_accuracy function to each row
    data['Accuracy'] = data.apply(evaluate_accuracy, axis=1)

    # Calculate accuracy rate
    accuracy_rate_bookmaker = data['Accuracy'].mean()

    # Example true labels and predicted probabilities
    true_labels = data["FTR"]
    predicted_probs_df = data[[bookmaker + "A_prob_fair", bookmaker + "D_prob_fair", bookmaker + "H_prob_fair"]]

    # Encode the true labels using LabelEncoder
    label_encoder = LabelEncoder()
    true_labels_encoded = label_encoder.fit_transform(true_labels)

    # Convert the true labels and predicted probabilities to numpy arrays
    true_labels_encoded = np.array(true_labels_encoded)
    predicted_probs = np.array(predicted_probs_df)

    # Calculate the log loss
    loss_book = log_loss(true_labels_encoded, predicted_probs)

    # Append the accuracy and log loss to the dataframe
    accuracy_bookmakers_df = accuracy_df.append({'Bookmaker': bookmaker, 'Accuracy': accuracy_rate_bookmaker, 'Log Loss': loss_book}, ignore_index=True)