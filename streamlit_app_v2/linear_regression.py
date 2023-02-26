import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score


class Linear_Regression:

    def __init__(self, data, test_size):
        self.data = data
        self.test_size = test_size

    # def clean_data(self):
    #     self.data['Date'] = pd.to_datetime(self.data['Date']).dt.date
    #     return data

    def create_train_test_data(self):
        # data = self.clean_data()
        train_data_len = math.ceil(len(self.data) * (1 - self.test_size))
        train_data = self.data[:train_data_len]
        test_data = self.data[train_data_len:]

        return train_data, test_data

    def create_model(self):
        train_data = self.create_train_test_data()[0]
        test_data = self.create_train_test_data()[1]
        x_train = train_data.drop(columns=['Date', 'Close'], axis=1)
        x_test = test_data.drop(columns=['Date', 'Close'], axis=1)
        y_train = train_data['Close']
        y_test = test_data['Close']

        # First Create the LinearRegression object and then fit it into the model

        model = LinearRegression()
        model.fit(x_train, y_train)

        # Making the Predictions
        prediction = model.predict(x_test)

        return prediction
