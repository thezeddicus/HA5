#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression



class Builder:
    def __init__(self, x_train, y_train):
        self.X_train = x_train
        self.y_train = y_train
        self.random_seed = 0

    def set_random_seed(self, seed):
        self.random_seed = seed

    def get_subsample(self, df_share):
        x = self.x_train.copy()
        y = self.y_train.copy()
        x, y = shuffle(x, y, random_state=self.random_seed)
        indexes = int((df_share / 100) * len(y))
        return x[:indexes], y[:indexes]

def main():
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    x = iris_df.drop(labels='sepal length (cm)', axis=1)
    y = iris_df['sepal length (cm)']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.42, random_state=107)

    pattern_item = Builder(x_train, y_train)
    pattern_item.set_random_seed(61)
    for df_share in range(10, 107, 10):
        curr_x_train, curr_y_train = pattern_item.get_subsample(df_share)
        lr = LinearRegression()
        lr.fit(curr_x_train, curr_y_train)

        lr.predict(x_test)
        pred = lr.predict(x_test)

        print(f'MSE at {df_share}:', mean_squared_error(y_test, pred))
        
if __name__ == "__main__":
    main()

