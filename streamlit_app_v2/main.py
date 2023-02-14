import streamlit as st
#from stock import Stock
import yfinance
from plotly import graph_objects as go
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from get_all_tickers import get_tickers as gt #get all tickers
from stock import Stock
import pandas as pd
import math

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow import keras


def get_today():
    return dt.datetime.now().strftime("%Y-%m-%d")

def nearest_business_day(DATE):
    """
    Takes a date and transform it to the nearest business day
    """
    if DATE.weekday() == 5:
        DATE = DATE - dt.timedelta(days=1)

    if DATE.weekday() == 6:
        DATE = DATE + dt.timedelta(days=1)
    return DATE

def get_yesterday():
    yesterday = dt.date.today() - dt.timedelta(days=1)
    yesterday = nearest_business_day(yesterday)
    return yesterday

def show_data_table(data, start_date):
    for i in range(0, len(data)):
        if start_date <= pd.to_datetime(data['Date'][i]):
            start_row = i
            break
    # data = data.set_index(pd.DatetimeIndex(data['Date'].values))
    st.write(data.iloc[start_row:, :])

# LSTM
def test_train_LSTM(stock_data, test_size, epochs, batch_size, optimizer, loss):

    stock_data_open = stock_data.filter(['Close'])
    dataset = stock_data_open.values

    # Training Data
    training_data_len = math.ceil(len(dataset) * test_size)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0: training_data_len, :]

    x_train_data, y_train_data = [], []

    for i in range(60, len(train_data)):
        x_train_data.append(train_data[i - 60:i, 0])
        y_train_data.append(train_data[i, 0])

    x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)

    x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

    # Testing Data
    test_data = scaled_data[training_data_len - 60:, :]

    x_test_data = []
    y_test_data = dataset[training_data_len:, :]

    for j in range(60, len(test_data)):
        x_test_data.append(test_data[j - 60:j, 0])

    x_test_data = np.array(x_test_data)

    x_test_data = np.reshape(x_test_data, (x_test_data.shape[0], x_test_data.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))

    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer=optimizer, loss=loss)

    model.fit(x_train_data, y_train_data, batch_size=int(batch_size), epochs=int(epochs))
    st.success("Your Model is Trained Succesfully!")
    st.markdown('')
    st.write("Predicted vs Actual Results for LSTM")
    st.write("Stock Prediction on Test Data for - ", ticker_name)

    predictions = model.predict(x_test_data)
    predictions = scaler.inverse_transform(predictions)

    train = stock_data_open[:training_data_len]
    valid = stock_data_open[training_data_len:]
    valid['Predictions'] = predictions

    new_valid = valid.reset_index()
    new_valid.drop('index', inplace=True, axis=1)
    st.dataframe(new_valid)
    st.markdown('')
    st.write("Plotting Actual vs Predicted ")

    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.figure(figsize=(14, 8))
    plt.title('Actual Close prices vs Predicted Using LSTM Model', fontsize=20)
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Actual', 'Predictions'], loc='upper left', prop={"size": 20})
    st.pyplot()
# --------------------- Main - Layour and Title ---------------------

st.set_page_config(layout='wide', initial_sidebar_state='expanded')
st.title('Stock Analysis')
#arguments: label, min, default, max



# --------------------- SIDEBAR ---------------------
sidebar = st.sidebar.container() # create an empty container in the sidebar
sidebar.markdown("# Settings") # add a title to the sidebar container
sidebar.subheader('Retrieve Data')
sub_columns = sidebar.columns(2) #S plit the container into two columns for start and end date

# Time window selection
TODAY = get_today()
YESTERDAY = get_yesterday()
DEFAULT_START=YESTERDAY - dt.timedelta(days=700)
DEFAULT_START = nearest_business_day(DEFAULT_START)
START = sub_columns[0].date_input("From")
END = sub_columns[1].date_input("To")

if START == '':
    START = DEFAULT_START
if END == '':
    END = YESTERDAY
# Stock Symbol Selection
STOCKS = np.array(['AAPL', 'GOOGL', 'INTC', 'TSLA'])
SYMBOL = sidebar.selectbox('Select Stock', STOCKS)


#  ------------------------Tabs--------------------
plot, stock_data, stock_info = st.tabs(['Plot', 'Stock Data', 'Stock Information'])

with plot:

    #  ------------------------Plot stock linechart--------------------
    chart_width = st.expander(label='chart width').slider('', 1000, 2800, 1400)
    fig=go.Figure()
    stock = Stock(symbol=SYMBOL)
    data = stock.load_data(START, END, inplace=True)
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    features_selected = sidebar.multiselect('Features to Plot', numeric_cols, default=['Open'])
    fig = stock.plot_multiple(data, choice=features_selected)
    # #fig2 = stock.plot_high_low(fig)
    # #styling for plotly
    fig.update_layout(
                width=chart_width,
                margin=dict(l=0, r=0, t=0, b=0, pad=0),
                legend=dict(
                    x=0,
                    y=0.99,
                    traceorder="normal",
                    font=dict(size=12),
                ),
                autosize=False,
        template="plotly_dark",
    )

    st.write(fig)

with stock_data:
    show_data_table(data, START)

with stock_info:
    history_1mo= stock.stock_information()[0]
    splits = stock.stock_information()[1]
    st.write('Company History in the past 1 month', history_1mo)
    st.write('Splits', splits)


# Preediction method selection

sidebar.markdown('## Stock Prediction')
prediction_method = sidebar.selectbox(
    'Select Prediction Method',
    ('Long Short Term Memory', 'Linear Regression'))

if prediction_method == 'Long Short Term Memory':

    sidebar.write('Select Parameters')
    # ------------- Default Parameters -------------
    # if "TEST_SIZE" not in sidebar.session_state:
    #     # set the initial default value of test size
    #     sidebar.session_state.TEST_SIZE = 0.2
    #
    # if "BATCH_SIZE" not in sidebar.session_state:
    #     # set the initial default value of the training length widget
    #     sidebar.session_state.BATCH_SIZE = 64
    #
    # if "EPOCHS" not in sidebar.session_state:
    #     # set the initial default value of the training length widget
    #     sidebar.session_state.EPOCHS = 50
    #
    # if "DROPOUT" not in sidebar.session_state:
    #     # set the initial default value of horizon length widget
    #     sidebar.session_state.DROPOUT = 0.4


    # ------------- Parameters - Choice -------------
    DROPOUT = sidebar.number_input(
        "Dropout (This regularization can help the model not overfit our training data)", min_value=0.1, max_value=0.5,
        key="DROPOUT"
    )
    TEST_SIZE = sidebar.number_input(
        "The testing set rate, e.g  0.2 means 20% of the total dataset",
        min_value=0.1,
        max_value=0.5,
        key="TEST_SIZE",
    )

    EPOCHS = sidebar.slider("Number of epochs", 0, 300, step=1)

    BATCH_SIZE = sidebar.slider('Batch size', 0, 1024, step=1)
    # Optimizer Selection
    optimizer = sidebar.selectbox(
        'Select Optimizer',
        ('adam', 'sgd', 'adamw', 'adadelta'))

    # Loss Selection
    loss = sidebar.selectbox(
        'Select Loss',
        ('mean squared error', 'huber loss'))

    submit = sidebar.button('Train Model')
    if submit:
        st.write('**Dataset_ _for_ Training**')
        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date']).dt.date
        st.write(data[['Date', 'Close']])
        #prepare train test
        test_train_LSTM(data, TEST_SIZE, EPOCHS, BATCH_SIZE, optimizer, loss)