# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:08:45 2023

@author: johan_nii2lon
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Define the ModelTrainer class
class ModelTrainer:
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.logreg_model = None
        
    # Train a Random Forest model
    def train_rf(self, X_train, y_train, dif=True, save_model=False, n_estimators=100, max_depth=None, min_samples_split=2):
        """
        Trains a Random Forest model on the given training data.

        Arguments:
        - X_train: The input features for training.
        - y_train: The target variables for training.
        - dif (optional): Indicates whether to perform difference-based training. Default is True.
        - save_model (optional): Indicates whether to save the trained model. Default is False.
        - n_estimators (optional): The number of trees in the random forest. Default is 100.
        - max_depth (optional): The maximum depth of the trees. Default is None.
        - min_samples_split (optional): The minimum number of samples required to split an internal node. Default is 2.

        Returns:
        - The trained Random Forest model.
        """

        if not dif:
            # Create a Random Forest classifier with specified parameters
            rf_classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)

            # Create a MultiOutputClassifier to handle multi-output classification
            self.rf_model = MultiOutputClassifier(rf_classifier)

            # Fit the model on the training features with home and away goals as target variables
            self.rf_model.fit(X_train, y_train)
        else:
            # Compute the difference between the home and the away goals
            y_dif_train = y_train.iloc[:, 1] - y_train.iloc[:, 0]

            # Create a Random Forest classifier with specified parameters
            self.rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)

            # Fit the model on the training features with difference in goals as target variable
            self.rf_model.fit(X_train, y_dif_train)

        # Save the model if save_model is True
        if save_model:
            with open('rf_model.pkl', 'wb') as f:
                pickle.dump(self.rf_model, f)

        # Return model
        return self.rf_model





    def train_xgb(self, X_train, y_train, dif=True, save_model=False, learning_rate=0.1, max_depth=3, n_estimators=100):
        """
        Trains an XGBoost classifier on the given training data.
    
        Arguments:
        - X_train: The input features for training.
        - y_train: The target variables for training.
        - dif (optional): Indicates whether to perform difference-based training. Default is True.
        - save_model (optional): Indicates whether to save the trained model. Default is False.
        - learning_rate (optional): The learning rate of the XGBoost classifier. Default is 0.1.
        - max_depth (optional): The maximum depth of each tree in the XGBoost classifier. Default is 3.
        - n_estimators (optional): The number of trees in the XGBoost classifier. Default is 100.
    
        Returns:
        - The trained XGBoost model.
        """
        
       # Filter the rows where the target variable is equal to 9 to get rid of outlier
        filtered_indices = np.any(y_train == 9, axis=1)
        X_train_filtered = X_train[~filtered_indices]
        y_train_filtered = y_train[~filtered_indices]
   
        if not dif:
            # Create an XGBoost classifier with specified parameters
            xgb_classifier = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
           
            # Create a MultiOutputClassifier to handle multi-output classification
            self.xgb_model = MultiOutputClassifier(xgb_classifier)
            
            # Fit the model on the training features with home and away goals as target variables
            self.xgb_model.fit(X_train_filtered, y_train_filtered)
        else:
             # Compute the difference between the home and the away goals
             y_dif_train = y_train_filtered.iloc[:, 1] - y_train_filtered.iloc[:, 0]
             
             # Map the training data to get rid of negativ classes
             y_dif_train = y_dif_train.astype(int)
             y_dif_train = y_dif_train + abs(np.min(y_dif_train))
             
             # Create an XGBoost classifier with specified parameters
             self.xgb_model = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)
             
             # Fit the model on the training features with difference in goals as target variable
             self.xgb_model.fit(X_train_filtered, y_dif_train)
             
          # Save the model if save_model is True
        if save_model:
              with open('xgb_model.pkl', 'wb') as f:
                  pickle.dump(self.xgb_model, f)
  
        return self.xgb_model

    



    def train_logreg(self, X_train, y_train, dif=True, save_model=False, C=1.0, solver='lbfgs', max_iter=100):
        """
        Trains a Logistic Regression classifier on the given training data.
         
        Arguments:
         - X_train: The input features for training.
         - y_train: The target variables for training.
         - dif (optional): Indicates whether to perform difference-based training. Default is True.
         - save_model (optional): Indicates whether to save the trained model. Default is False.
         - C (optional): Inverse of regularization strength for Logistic Regression. Default is 1.0.
         - solver (optional): Solver algorithm for Logistic Regression. Default is 'lbfgs'.
         - max_iter (optional): Maximum number of iterations for Logistic Regression. Default is 100.
         
        Returns:
         - The trained Logistic Regression model.
         """
        if dif is not True:
            # Create an Logistic Regression classifier with specified parameters
            logreg_classifier = LogisticRegression(C=C, solver=solver, max_iter=max_iter)
            
            # Create a MultiOutputClassifier to handle multi-output classification 
            self.logreg_model = MultiOutputClassifier(logreg_classifier)
            
            # Fit the model on the training features with home and away goals as target variables
            self.logreg_model.fit(X_train, y_train)
        else:
            # Compute the difference between the home and the away goals
            y_dif_train = y_train.iloc[:, 1] - y_train.iloc[:, 0]
            
            # Create an Logistic Regression classifier with specified 
            self.logreg_model = LogisticRegression(C=C, solver=solver, max_iter=max_iter)
            
            # Fit the model on the training features with difference in goals as target variable
            self.logreg_model.fit(X_train, y_dif_train)
         
        # Save the model if save_model is True
        if save_model:
            with open('logreg_model.pkl', 'wb') as f:
                pickle.dump(self.logreg_model, f)
            
        return self.logreg_model

    
    
    


class ModelPrediction:
    def __init__(self):
        pass

    def predict_prob(self, X_test, model, xgb = True, dif=True):
        """
        Predicts the probabilities of home win, draw, and away win for given test data.
        
        Arguments:
        - X_test: The test data to make predictions on.
        - model: The trained model to use for predictions.
        - xgb (optional): Indicates whether the model is an XGBoost classifier. Default is True.
        - dif (optional): Indicates whether to perform difference-based predictions. Default is True.
        
        Returns:
        - The predicted probabilities of home win, draw, and away win.
        """

        if dif is not True :
            # Predict the probabilities using the model
            y_pred_prob = model.predict_proba(X_test)
            goal_prob_home = pd.DataFrame(y_pred_prob[0])  # Probability of home win
            goal_prob_away = pd.DataFrame(y_pred_prob[1])  # Probability of away win
            num_goals_home = goal_prob_home.shape[1]  # Number of possible home goals for later iteration
            num_goals_away = goal_prob_away.shape[1]  # Number of possible away goals for later iteration
            
            # Empty dataframe to save series
            df_result_prob = pd.DataFrame(index=goal_prob_home.index)
            
            # Empty series for probilities of possible results
            home_winning_prob = pd.Series(0.0, index=goal_prob_home.index)
            away_winning_prob = pd.Series(0.0, index=goal_prob_home.index)
            draw_prob = pd.Series(0.0, index=goal_prob_home.index)
            
            # Compute probabilities for geneal results (homewin, draw, awaywin) while iterating over all possible combination
            # and adding the probalities of specific results depending on the general result they indicate
            for home_goals in range(num_goals_home):
                for away_goals in range(num_goals_away):
                    if home_goals > away_goals:
                        home_winning_prob += goal_prob_home.iloc[:, home_goals] * goal_prob_away.iloc[:, away_goals]
                    elif home_goals < away_goals:
                        away_winning_prob += goal_prob_home.iloc[:, home_goals] * goal_prob_away.iloc[:, away_goals]
                    else:
                        draw_prob += goal_prob_home.iloc[:, home_goals] * goal_prob_away.iloc[:, away_goals]
            
            # Save probabilities in dataframe 
            df_result_prob['home_winning_prob'] = home_winning_prob
            df_result_prob['draw_prob'] = draw_prob
            df_result_prob['away_winning_prob'] = away_winning_prob

        else:
            if xgb is True:
                # Predict probabilities using the model
                y_pred_prob_dif = model.predict_proba(X_test)
                
                # Save probabilities in dataframe and rename columns
                goal_prob_dif = pd.DataFrame(y_pred_prob_dif)
                goal_prob_dif.columns = goal_prob_dif.columns.astype(int) - 8
            else:
                # Predict probabilities using the model
                y_pred_prob_dif = model.predict_proba(X_test)
                
                # rename colums to class names 
                classes = model.classes_
                goal_prob_dif = pd.DataFrame(y_pred_prob_dif)
                new_columns = ['{}'.format(i) for i in classes]
                goal_prob_dif.columns = new_columns
    
            # Add up all single probabilities for the differences in goals according to the sign of their class represented in the colnames
            negative_columns = [col for col in goal_prob_dif.columns if float(col) < 0]
            home_winning_prob_dif = goal_prob_dif[negative_columns].sum(axis=1)
            positive_columns = [col for col in goal_prob_dif.columns if float(col) > 0]
            away_winning_prob_dif = goal_prob_dif[positive_columns].sum(axis=1)
            draw_columns = [col for col in goal_prob_dif.columns if float(col) == 0]
            draw_prob_dif = goal_prob_dif[draw_columns].sum(axis=1)

            # Save probabilities in dataframe 
            df_result_prob = pd.DataFrame()
            df_result_prob['home_winning_prob_dif'] = home_winning_prob_dif
            df_result_prob['draw_prob_dif'] = draw_prob_dif
            df_result_prob['away_winning_prob_dif'] = away_winning_prob_dif

        return df_result_prob
