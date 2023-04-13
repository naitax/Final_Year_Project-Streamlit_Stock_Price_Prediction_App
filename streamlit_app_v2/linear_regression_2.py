#Linear Regression libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly import graph_objects as go
import plotly.express as px

class Linear_Regression:

    def __init__(self, data, test_size, feature):
        self.data = data
        self.test_size = test_size
        self.feature = feature

    def sub_data(self, feature):
        df = pd.DataFrame(self.data, columns=[feature])
        # Reset index column so that we have integers to represent time for later analysis
        df = df.reset_index()
        return df

    def linear_regression_model(self, feature):
        df = self.sub_data(feature)
        train, test = train_test_split(df, test_size=self.test_size)
        # Reshape index column to 2D array for .fit() method
        X_train = np.array(train.index).reshape(-1, 1)
        y_train = train[feature]
        # Create LinearRegression Object
        model = LinearRegression()
        # Fit linear model using the train data set
        model.fit(X_train, y_train)

        # Create test arrays
        X_test = np.array(test.index).reshape(-1, 1)
        y_test = test[feature]
        # Generate array with predicted values
        y_pred = model.predict(X_test)

        return model, X_test, y_test, y_pred

    def model_evaluation(self, model, y_test, X_test):

        y_pred = model.predict(X_test)
        #slope = np.asscalar(np.squeeze(model.coef_))
        intercept = model.intercept_
        mea = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        r2 = metrics.r2_score(y_test, y_pred)
        return  intercept, mea, mse, rmse, r2

    def show_model_performance(self, model, X_test, y_test):
        plt.figure(1, figsize=(16, 10))
        plt.title('Linear Regression | Price vs Time')
        plt.plot(X_test, model.predict(X_test), color='r', label='Predicted Price')
        plt.scatter(X_test, y_test, edgecolor='w', label='Actual Price')

        plt.xlabel('Integer Date')
        plt.ylabel('Stock Price in $')

        return plt

    def show_predicted_vs_actual(self, pred_data, feature):

        #line plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pred_data['Date'], y=pred_data[feature], mode='lines', name=feature))
        fig.add_trace(go.Scatter(x=pred_data['Date'], y=pred_data['Prediction'], mode='lines', name='Predicted'))
        fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), height=550, width=800,
                          autosize=False, margin=dict(l=25, r=75, b=100, t=0))

        return fig

    def show_scatter_predicted_vs_actual(self, y_test, y_pred):
        # need:
        # y_test
        # y_pred

        fig = px.scatter(x=y_test, y=y_pred, title='Predicted vs Actual Price', labels={
            'x': 'Actual Prices',
            'y': 'Predicted Prices'
        })

        return fig

    def prediction_data(self, model, feature):
        df = self.sub_data(feature)
        df['Prediction'] = model.predict(np.array(df.index).reshape(-1, 1))
        return df
