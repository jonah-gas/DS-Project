# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:41:41 2023

@author: johan_nii2lon
"""
import pandas as pd

def drop_identical_cols(df):
    # get all pairs of identical columns
    duplicated_cols = []
    for i, col in enumerate(df.columns[:-1]):
        for j in range(i+1, len(df.columns)):
            if df[col].equals(df.iloc[:, j]):
                duplicated_cols.append((col, df.columns[j]))

    # drop one column from each pair of identical columns
    for col_pair in duplicated_cols:
        df = df.drop(col_pair[1], axis=1)

    # save the modified dataframe with a new name
    data_clean = df.copy()

    # return the modified dataframe
    return data_clean

