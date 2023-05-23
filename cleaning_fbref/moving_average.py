import pandas as pd
import numpy as np 

def moving_average(df, window_size = 10, discount = 0.5):
    "compute the moving average with recency bias"
    weights = np.array([discount**x for x in reversed(range(window_size))]) # discount y_t-windowsize with discount**windowsize
    sum_weights = np.sum(weights)
    mov_avg = df.rolling(window = window_size).apply(lambda x: np.sum(weights*x) / sum_weights, raw=False) # compute moving average of window_size 
    return mov_avg