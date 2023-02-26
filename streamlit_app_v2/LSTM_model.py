import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import deque

import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from datetime import date
import datetime
from stock import *

'''
random seed value while creating training and test data set. 
The goal is to make sure we get the same training and validation 
data set while we use different hyperparameters or 
machine learning algorithms in order to assess the 
performance of different models
'''

np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)

'''
Preparing the dataset

This function accepts several arguments to be as flexible as possible:

The ticker argument is the ticker we want to load. For instance, 
we can use TSLA for the Tesla stock market, AAPL for Apple, and so on. 
It can also be a pandas Dataframe with the condition it includes the columns in feature_columns and date as an index.

n_steps integer indicates the historical sequence length we want to use (some people call it the window size), 
choosing 50 means that we will use 50 days of stock prices to predict the next lookup time step.

scale is a boolean variable that indicates whether to scale prices from 0 to 1; 
we will set this to True as scaling high values from 0 to 1 will help the neural network to learn much faster and more effectively.

lookup_step is the future lookup step to predict, 
the default is set to 1 (e.g., next day). 15 means the next 15 days, and so on.

split_by_date is a boolean that indicates whether we split our 
training and testing sets by date. Setting it to False means we 
randomly split the data into training and testing using sklearn's 
train_test_split() function. If it's True (the default), we split the data in date order.

We will use all the features available in this dataset: open, high, low, volume, and adjusted close.

The function does the following:

First, it loads the dataset using stock_info.get_data() 
function in yahoo_fin module. It adds the "date" column 
from the index if it doesn't exist, this will help later 
to get the features of the testing set. If the scale argument 
is passed as True, it will scale all the prices from 0 to 1 
(including the volume) using sklearn's MinMaxScaler class. 
Note that each column has its own scaler.
It then adds the future column, which indicates the target 
values (the labels to predict, or the y's) by shifting the
adjusted close column by lookup_step.After that, it shuffles 
and splits the data into training and testing sets and 
finally returns the result.
'''

def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)


def load_data(ticker, start, end, n_steps, lookup_step, test_size, scale=True, shuffle=True, split_by_date=True,
              feature_columns=['adjclose', 'volume', 'open', 'high', 'low']):
    """
    Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
    Params:
        ticker (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
        n_steps (int): the historical sequence length (i.e window size) used to predict, default is 50
        scale (bool): whether to scale prices from 0 to 1, default is True
        shuffle (bool): whether to shuffle the dataset (both training & testing), default is True
        lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
        split_by_date (bool): whether we split the dataset into training/testing by date, setting it
            to False will split datasets in a random way
        test_size (float): ratio for test data, default is 0.2 (20% testing data)
        feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
    """
    # see if ticker is already a loaded stock from yahoo finance
    #     if isinstance(ticker, str):
    #         # load it from yahoo_fin library
    #         df = si.get_data(ticker)
    #     elif isinstance(ticker, pd.DataFrame):
    #         # already loaded, use it directly
    #         df = ticker
    #     else:
    #         raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")
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
    # make sure that the passed feature_columns exist in the dataframe
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    # add date as a column
    if "date" not in df.columns:
        df["date"] = df.index
    if scale:
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
    if split_by_date:
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"] = X[train_samples:]
        result["y_test"] = y[train_samples:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:
        # split the dataset randomly
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                                    test_size=test_size,
                                                                                                    shuffle=shuffle)
    # get the list of test set dates
    dates = list(result["X_test"][:, -1, -1])
    dates = [date_obj.strftime('%Y-%m-%d') for date_obj in dates]
    # print(dates)
    # df['ConvertedDate']=pd.to_datetime(df['DateTypeCol'].astype(str), format='%Y/%m/%d')
    # print(dates)

    # retrieve test features from the original dataframe

    result["test_df"] = result["df"].loc[dates]
    # print(result)
    # remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
    return result


def create_model(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                 loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    model = Sequential()
    # for i in range(n_layers):
    #     if i == 0:
    #         # first layer
    #         if bidirectional:
    #             model.add(Bidirectional(cell(units, return_sequences=True),
    #                                     batch_input_shape=(None, sequence_length, n_features)))
    #         else:
    #             model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
    #     elif i == n_layers - 1:
    #         # last layer
    #         if bidirectional:
    #             model.add(Bidirectional(cell(units, return_sequences=False)))
    #         else:
    #             model.add(cell(units, return_sequences=False))
    #     else:
    #         # hidden layers
    #         if bidirectional:
    #             model.add(Bidirectional(cell(units, return_sequences=True)))
    #         else:
    #             model.add(cell(units, return_sequences=True))
    #     # add dropout after each layer
    #     model.add(Dropout(dropout))

    #model = Sequential()
    model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
    model.add(LSTM(units=50, return_sequences=False))

    model.add(Dense(25))
    model.add(Dense(1))

    #model.add(Dense(1, activation="linear"))
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
    # if predicted future price is higher than the current,
    # then calculate the true future price minus the current price, to get the buy profit
    buy_profit  = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
    # if the predicted future price is lower than the current price,
    # then subtract the true future price from the current price
    sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0
    X_test = data["X_test"]
    y_test = data["y_test"]
    # perform prediction and get prices
    y_pred = model.predict(X_test)
    if SCALE:
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
    # add the buy profit column
    final_df["buy_profit"] = list(map(buy_profit,
                                    final_df["adjclose"],
                                    final_df[f"predicted adjclose_{LOOKUP_STEP}"],
                                    final_df[f"true_adjclose_{LOOKUP_STEP}"])
                                    # since we don't have profit for last sequence, add 0's
                                    )
    # add the sell profit column
    final_df["sell_profit"] = list(map(sell_profit,
                                    final_df["adjclose"],
                                    final_df[f"predicted adjclose_{LOOKUP_STEP}"],
                                    final_df[f"true_adjclose_{LOOKUP_STEP}"])
                                    # since we don't have profit for last sequence, add 0's
                                    )
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






def train_model(ticker, N_STEPS, SCALE, LOOKUP_STEP, TEST_SIZE, FEATURE_COLUMNS, LOSS, OPTIMIZER, DROPOUT, BATCH_SIZE, EPOCHS, START, END):

    FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
    data = load_data(ticker, START, END, N_STEPS, scale=SCALE, split_by_date=False,
                     shuffle=True, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE)

    model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell='LSTM', n_layers=2,
                         dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=True)

    # some tensorflow callbacks
    # checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True,
    #                                save_best_only=True, verbose=1)
    # tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name))

    # train the model and save the weights whenever we see
    # a new optimal model using ModelCheckpoint
    history = model.fit(data["X_train"], data["y_train"],
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(data["X_test"], data["y_test"]),
                        verbose=1)


