import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Flatten, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers.convolutional import Conv1D, MaxPooling1D

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn import metrics
from collections import deque

import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from datetime import date
import datetime
from stock import *

np.random.seed(123)
tf.random.set_seed(123)
random.seed(123)

def load_data(ticker, start, end, n_steps, lookup_step, test_size, scale=True,
              feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):
    stock = Stock(ticker)
    df = stock.load_data(start, end)
    df['ticker'] = ticker
    df.set_index('Date', inplace=True)
    df['date'] = df.index
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'adjclose',
                            'Volume': 'volume'})
    # this will contain all the elements we want to return from this function
    # print(f'Yahoo finance data: {df.head()}')
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()
    column_scaler = {}
    # scale the data (prices) from 0 to 1
    for column in feature_columns:
        scaler = preprocessing.MinMaxScaler()
        df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
        column_scaler[column] = scaler
    # add the MinMaxScaler instances to the result returned
    result["column_scaler"] = column_scaler
    # print(f'Result: {result}')
    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['adjclose'].shift(-lookup_step)
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    # drop NaNs
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices that are not available in the dataset
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # add to result
    result['last_sequence'] = last_sequence
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # split the dataset randomly
    result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                                test_size=test_size)
    # get the list of test set dates
    dates = list(result["X_test"][:, -1, -1])
    dates = [date_obj.strftime('%Y-%m-%d') for date_obj in dates]

    # retrieve test features from the original dataframe
    result["test_df"] = result["df"].loc[dates]
    # remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
    return result


def create_model(sequence_length, n_features, n_layers, dropout,
                 loss, optimizer, model_type, activation, units):
    model = Sequential()

    if model_type == 'Bidirectional LSTM':
        for i in range(n_layers):
            if i == 0:
                # first layer
                model.add(Bidirectional(LSTM(units, return_sequences=True),
                                        batch_input_shape=(None, sequence_length, n_features)))

            elif i == n_layers - 1:
                # last layer
                model.add(Bidirectional(LSTM(units, return_sequences=False)))

            else:
                # hidden layers
                model.add(Bidirectional(LSTM(units, return_sequences=True)))
            # add dropout after each layer
            model.add(Dropout(dropout))
        model.add(Dense(1, activation=activation))

    elif model_type == 'Vanilla LSTM':

        # single hidden layer of LSTM units, and an output layer used to make a prediction
        model.add(LSTM(units, batch_input_shape=(None, sequence_length, n_features)))  #
        model.add(Dense(1))

    elif model_type == 'Stacked LSTM':

        # Multiple hidden LSTM layers stacked one on top of another
        model.add(LSTM(units, activation=activation, return_sequences=True,
                       batch_input_shape=(None, sequence_length, n_features)))
        model.add(LSTM(units, activation=activation))
        model.add(Dense(1))

    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model



def plot_graph(test_df, LOOKUP_STEP, stock):
    plt.figure(figsize=(30, 20))
    plt.plot(test_df[f'true_adjclose_{LOOKUP_STEP}'], c='green')
    plt.plot(test_df[f'predicted adjclose_{LOOKUP_STEP}'], c='purple')
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Price", fontsize=20)
    plt.legend(["Actual Price", "Predicted Price"], prop={'size': 25})
    # plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.title(f'{stock} stock price prediction', fontsize=25)
    plt.xticks(size=15)
    plt.yticks(size=15)

    return plt

def get_final_df(model, data, LOOKUP_STEP):
    """
    This function takes the `model` and `data` dict to
    construct a final dataframe that includes the features along
    with true and predicted prices of the testing dataset
    """
    SCALE = True
    X_test = data["X_test"]
    y_test = data["y_test"]
    # perform prediction and get prices
    y_pred = model.predict(X_test)
    # scale
    y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    test_df = data["test_df"]
    # add predicted future prices to the dataframe
    test_df[f"predicted adjclose_{LOOKUP_STEP}"] = y_pred
    # add true future prices to the dataframe
    test_df[f"true_adjclose_{LOOKUP_STEP}"] = y_test
    # sort the dataframe by date
    test_df.sort_index(inplace=True)
    final_df = test_df


    return final_df


def predict(model, data, N_STEPS):
    # retrieve the last sequence from data


    SCALE = True
    predicted_price = []
    for n in range(0, N_STEPS):

        last_sequence = data["last_sequence"][-n:]
        # expand dimension
        last_sequence = np.expand_dims(last_sequence, axis=0)
        # get the prediction (scaled from 0 to 1)
        prediction = model.predict(last_sequence)
        # get the price (by inverting the scaling)
        if SCALE:
            predicted = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
        else:
            predicted = prediction[0][0]
        predicted_price.append(predicted)



    return predicted_price


def evaluate(model, y_true, y_pred):

  MAE = mean_absolute_error(y_true, y_pred)
  MSE = mean_squared_error(y_true, y_pred)
  RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
  R2 = r2_score(y_true, y_pred)
  RMSE = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
  MAPE = mean_absolute_percentage_error(y_true, y_pred) *100

  return MAE, MSE, RMSE, R2, RMSE, MAPE



def train_model(ticker, N_STEPS, SCALE, LOOKUP_STEP, TEST_SIZE, FEATURE_COLUMNS, LOSS, OPTIMIZER, DROPOUT, BATCH_SIZE, EPOCHS, START, END):

    FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
    data = load_data(ticker, START, END, N_STEPS, scale=SCALE, split_by_date=False,
                     shuffle=True, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE)

    model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell='LSTM', n_layers=2,
                         dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=True)

    history = model.fit(data["X_train"], data["y_train"],
                        batch_si6ze=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(data["X_test"], data["y_test"]),
                        verbose=1)



