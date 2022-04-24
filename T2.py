#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import LinearSVC

class Facade:
    def __init__(self, models) -> None:
        self.models = models

    def fit(self, x, y):
        for model in self.models:
            model.fit(x, y)

    def predict(self, x):
        predictions = []

        for model in self.models:
            predictions.append(model.predict(x))

        predictions = np.vstack(predictions)
        result = np.zeros(predictions.shape[1])
        for i in range(predictions.shape[1]):
            result[i] = np.bincount(predictions[:, i]).argmax()

        return result
    
    def main():

        data = load_iris()
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.41, shuffle=True)

    models = [LogisticRegression(), GaussianProcessClassifier(), LinearSVC(max_iter=1000)]

    ensamble = Facade(models)
    ensamble.fit(x_train, y_train)
    predict = ensamble.predict(x_test)
    print(predict)
    
if __name__ == "__main__":
    main()

