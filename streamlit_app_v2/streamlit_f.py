import streamlit as st
#from stock import Stock
from plotly import graph_objects as go
import datetime as dt
from datetime import timedelta, date
from stock import Stock
import math
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import io
from LSTM_model import *
from linear_regression import *
from methods import *
from linear_regression_2 import *
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils.vis_utils import plot_model

from PIL import Image
def show_data_table(data, start_date):
    for i in range(0, len(data)):
        if start_date <= pd.to_datetime(data['Date'][i]):
            start_row = i
            break
    # data = data.set_index(pd.DatetimeIndex(data['Date'].values))
    st.write(data.iloc[start_row:, :])




# LSTM2
def train_model(ticker, N_STEPS, SCALE, LOOKUP_STEP, TEST_SIZE, FEATURE_COLUMNS, LOSS, OPTIMIZER, DROPOUT, BATCH_SIZE, EPOCHS, START, END, MODEL_TYPE, N_LAYERS, ACTIVATION, UNITS):


    data = load_data(ticker, START, END, N_STEPS, scale=SCALE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE)

    model = create_model(N_STEPS, len(FEATURE_COLUMNS), N_LAYERS, DROPOUT, LOSS,
                         OPTIMIZER, MODEL_TYPE, ACTIVATION, UNITS)

    history = model.fit(data["X_train"], data["y_train"],
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(data["X_test"], data["y_test"]),
                        verbose=1)

    model.save(
        f'streamlit_app_v2/prediction_models/model_type={MODEL_TYPE},batch_size={BATCH_SIZE},epochs={EPOCHS},dropout={DROPOUT},seq_length={LOOKUP_STEP},optimizer={OPTIMIZER},loss={LOSS}.h5')

    # final dataframe
    final_df = get_final_df(model, data, LOOKUP_STEP)
    final_df = final_df.drop('date', axis=1)
    # evaluation metrics
    y_true = final_df[f'true_adjclose_{LOOKUP_STEP}'].values.tolist()
    y_pred = final_df[f'predicted adjclose_{LOOKUP_STEP}'].values.tolist()
    MAE, MSE, RMSE, R2, RMSE, MAPE = evaluate(model, y_true, y_pred)
    # future prices for N_STEPS days
    future_price = predict(model, data, N_STEPS)
    predicted_price = pd.DataFrame(future_price)
    tomorrow = END + timedelta(days=1)
    predicted_price.rename(columns={0: 'Future Price'})
    predicted_price['Date'] = pd.date_range(start=tomorrow, periods=len(predicted_price), freq='D')
    predicted_price['Date'] = pd.to_datetime(predicted_price['Date']).dt.date
    predicted_price.rename(columns={0: 'Future Adj Close Price'}, inplace=True)
    # plot for future prediction
    fig_predicted = go.Figure()
    fig_predicted.add_trace(go.Scatter(x=predicted_price['Date'], y=predicted_price['Future Adj Close Price'], mode='lines', name='Future Adj Close Price'))
    fig_predicted.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), height=550, width=1000,
                      autosize=False, margin=dict(l=25, r=75, b=100, t=0))

    st.success("Your Model is Trained Succesfully!")
    st.markdown('')
    st.write('Final dataframe with true and predicted prices')
    st.dataframe(final_df)
    st.write(f'Prices for the next {N_STEPS} days')
    st.dataframe(predicted_price)
    # st.plotly_chart(fig_predicted)

    # plot = plot_graph(final_df, LOOKUP_STEP, SYMBOL)
    #st.pyplot(plot)
    st.write(f'True vs Predicted Adj Close')
    plotly_figure3 = px.line(data_frame=final_df, y=[f'true_adjclose_{LOOKUP_STEP}', f'predicted adjclose_{LOOKUP_STEP}'],
                             title=f'')

    plotly_figure3.update_layout(
        width=1000,
        margin=dict(l=0, r=0, t=0, b=0, pad=0),
        legend=dict(
            x=0,
            y=0.99,
            traceorder="normal",
            font=dict(size=12),
        ),
        autosize=False,
        template="seaborn",
    )

    st.write(plotly_figure3)

    st.subheader(f'{MODEL_TYPE} Model Evaluation')
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    #
    # image = Image.open('model_plot.png')
    # st.write(image)
    st.write('Mean Absolute Percentage Error: ', MAPE)
    st.write('MAE: ', MAE)
    st.write('MSE: ', MSE)
    st.write('RMSE: ', RMSE)
    st.write('R^2: ', R2)

    save_prediction_models(MODEL_TYPE, ticker, START, END, FEATURE_COLUMNS, TEST_SIZE, BATCH_SIZE, EPOCHS, DROPOUT, LOOKUP_STEP, OPTIMIZER, LOSS,
                           '-', MAE, MSE, RMSE, R2)



# Linear Regression
def prediction_plot(prediction_data, test_data, symbol):


    test_data['Predicted'] = 0
    test_data['Predicted'] = prediction_data

    # Resetting the index
    test_data.reset_index(inplace=True, drop=True)
    st.success("Your Model is Trained Succesfully!")
    st.markdown('')
    st.write(f"Predicted Price vs Actual Close Price Results for {symbol}")
    st.write(test_data[['Date', 'Close', 'Predicted']])
    st.write(f"Plotting Close Price vs Predicted Price for - {symbol}")

    # Plotting the Graph
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=test_data['Date'], y=test_data['Predicted'], mode='lines', name='Predicted'))
    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), height=550, width=800,
                      autosize=False, margin=dict(l=25, r=75, b=100, t=0))

    st.plotly_chart(fig)
#
# # LSTM
# def test_train_LSTM(stock_data, test_size, epochs, batch_size, optimizer, loss, symbol):
#
#     stock_data_open = stock_data.filter(['Close'])
#     dataset = stock_data_open.values
#
#     # Training Data
#     training_data_len = math.ceil(len(dataset) * test_size)
#
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(dataset)
#
#     train_data = scaled_data[0: training_data_len, :]
#
#     x_train_data, y_train_data = [], []
#
#     for i in range(60, len(train_data)):
#         x_train_data.append(train_data[i - 60:i, 0])
#         y_train_data.append(train_data[i, 0])
#
#     x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
#
#     x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))
#
#     # Testing Data
#     test_data = scaled_data[training_data_len - 60:, :]
#
#     x_test_data = []
#     y_test_data = dataset[training_data_len:, :]
#
#     for j in range(60, len(test_data)):
#         x_test_data.append(test_data[j - 60:j, 0])
#
#     x_test_data = np.array(x_test_data)
#
#     x_test_data = np.reshape(x_test_data, (x_test_data.shape[0], x_test_data.shape[1], 1))
#
#     model = Sequential()
#     model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
#     model.add(LSTM(units=50, return_sequences=False))
#
#     model.add(Dense(25))
#     model.add(Dense(1))
#
#     model.compile(optimizer=optimizer, loss=loss)
#
#     model.fit(x_train_data, y_train_data, batch_size=int(batch_size), epochs=int(epochs))
#     st.success("Your Model is Trained Succesfully!")
#     st.markdown('')
#     # stringlist = []
#     # model.summary(print_fn=lambda x: stringlist.append(x))
#     # short_model_summary = "\n".join(stringlist)
#     # st.write(short_model_summary)
#     st.write('Model Summary')
#     model_summary_string = get_model_summary(model)
#     st.write(model_summary_string)
#     st.write("Predicted vs Actual Results for LSTM")
#     st.write(f"Stock Prediction for {symbol}")
#
#     predictions = model.predict(x_test_data)
#     predictions = scaler.inverse_transform(predictions)
#
#     train = stock_data_open[:training_data_len]
#     valid = stock_data_open[training_data_len:]
#     valid['Predictions'] = predictions
#
#     new_valid = valid.reset_index()
#     new_valid.drop('index', inplace=True, axis=1)
#     st.dataframe(new_valid)
#     st.markdown('')
#     st.write("Plotting Actual vs Predicted ")
#     st.write(f'Used parameters: \n'
#               f'Test Size: {test_size}, \n'
#               f'Number of Epochs: {epochs}, \n'
#               f'Number of batches: {batch_size}, \n'
#               f'Optimizer: {optimizer}, \n'
#               f'Loss: {loss}')
#     # st.set_option('deprecation.showPyplotGlobalUse', False)
#     # plt.figure(figsize=(14, 8))
#     # plt.title(f'Actual Close prices vs Predicted Using LSTM Model', fontsize=20)
#     # plt.plot(valid[['Close', 'Predictions']])
#     # plt.legend(['Actual', 'Predictions'], loc='upper left', prop={"size": 20})
#     # st.pyplot()
#     leng= len(valid['Close'])
#     plotly_figure2 = px.line(data_frame=valid, y=['Close', 'Predictions'],
#                             title=f'')
#     plotly_figure2.update_layout(
#         width=chart_width,
#         margin=dict(l=0, r=0, t=0, b=0, pad=0),
#         legend=dict(
#             x=0,
#             y=0.99,
#             traceorder="normal",
#             font=dict(size=12),
#         ),
#         autosize=False,
#         template="seaborn",
#     )
#
#     st.write(plotly_figure2)


# --------------------- Main - Layour and Title ---------------------

def linear_regression_tab(data, test_size, symbol, feature, start, end):


    lr = Linear_Regression(data, test_size, feature)
    model, X_test, y_test, y_pred = lr.linear_regression_model(feature)
    intercept, mea, mse, rmse, r2 = lr.model_evaluation(model, y_test, X_test)
    st.success("Your Model is Trained Succesfully!")
    st.markdown('')
    st.write(f"Predicted Price vs Actual Close Price Data for {symbol}")
    st.write(lr.prediction_data(model, feature))
    st.write(f"Model Performance for {symbol}")
    # fig1 = lr.show_model_performance(model, X_test, y_test)
    # st.write(fig1)
    # st.write(f"Predicted Price vs Actual Close Price for {symbol}")
    # fig2 = lr.show_predicted_vs_actual(model, X_test, y_test)
    # st.write(fig2)

    pred_data = lr.prediction_data(model, feature)
    # Plotting the Graph
    st.write('Linear Regression | Actual vs Predicted Price')
    line_fig = lr.show_predicted_vs_actual(pred_data, feature)
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=pred_data['Date'], y=pred_data['Close'], mode='lines', name='Close'))
    # fig.add_trace(go.Scatter(x=pred_data['Date'], y=pred_data['Prediction'], mode='lines', name='Predicted'))
    # fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), height=550, width=800,
    #                   autosize=False, margin=dict(l=25, r=75, b=100, t=0))

    st.plotly_chart(line_fig)

    scatter_fig = lr.show_scatter_predicted_vs_actual(y_test, y_pred)
    st.plotly_chart(scatter_fig)
    st.markdown('')
    st.write('Model evaluation')
    #st.write('Slope: ', slope)
    st.write('Intercept: ', intercept)
    st.write('Mean Absolute Error: ', mea)
    st.write('Mean Squared Error: ', mse)
    st.write('Root Mean Squared Error: ', rmse)
    st.write('R2:', r2)

    save_prediction_models('Linear Regression', symbol, start, end, feature, test_size, intercept, mea, mse, rmse, r2)



def streamlit_app():

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
    START = sub_columns[0].date_input('From', min_value=datetime.datetime(2010, 1, 1))
    END = sub_columns[1].date_input('To', max_value=TODAY)

    if START == '':
        START = DEFAULT_START
    if END == '':
        END = YESTERDAY
    # Stock Symbol Selection
    STOCKS = np.array(get_all_ticker_names())
    SYMBOL = sidebar.selectbox('Select Stock', STOCKS)


    #  ------------------------Tabs--------------------
    plot, stock_data, stock_info, stock_analysis = st.tabs(['Plot', 'Stock Data', 'Stock Information', 'Stock Analysis'])

    if check_if_today_greater(END, TODAY) is True or check_if_today_greater(START, END) is True:
        st.warning('Please select a date range with end date not greater than today and start date not greater than end date')

    else:
        with plot:

            #  ------------------------Plot stock linechart--------------------
            chart_width = st.expander(label='chart width').slider('', 1000, 2800, 1400)
            fig=go.Figure()
            stock = Stock(symbol=SYMBOL)
            data = stock.load_data(START, END, inplace=True)
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            features_selected = sidebar.multiselect('Features to Plot', numeric_cols, default=['Open'])
            try:
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
            except ValueError:
                st.error('Please select a date range')
                #st.markdown(':red[Please select a **date range**]')

        with stock_data:
            try:
                show_data_table(data, START)
                st.markdown(f'**{SYMBOL} Summary**')
                st.write(data.describe())
            except:
                st.write('')


        with stock_info:
            history_1mo= stock.stock_information()[0]
            splits = stock.stock_information()[1]
            ticker = stock.ticker()
            major_holders = ticker.major_holders
            major_holders.rename(columns={0: 'Percentage', 1: 'Holders'}, inplace=True)
            st.write('Company History in the past 1 month', history_1mo)
            st.markdown('**Holders**')
            st.write(f'Major Holders' ,major_holders)
            st.write(f'Institutional Holders' ,ticker.institutional_holders)
            st.write(f'Mutual Holders' ,ticker.mutualfund_holders)
            st.markdown('**Earnings**')
            st.write(f'Earnings Dates' ,ticker.earnings_dates)

        with stock_analysis:

            try:

                st.subheader(f'{SYMBOL} Stock Correlation')

                st.write(stock.visualise_correlation(data))

                st.subheader(f'{SYMBOL} Stock Volatility')
                volatility = stock.calculate_volatility(data)
                volatility_fig = stock.visualise_volatility(data, volatility, SYMBOL)
                st.plotly_chart(volatility_fig, use_container_width=True, height=800)
                #st.write(volatility_fig)

                st.subheader(f'Decomposition of time series')
                for feature in features_selected:
                    multiplicative, additive = stock.decompose_time_series(data, feature)
                    additive_fig = stock.plot_seasonal_decompose(additive, data, data['Date'], 'Additive', feature)
                    st.write(f'{feature} Additive Seasonal Decomposition')
                    st.plotly_chart(additive_fig, use_container_width=True)
                    #st.write(additive_fig)
                    additive_table = stock.additive_decomposing_table(additive)
                    st.write(f'Additive {SYMBOL} Table Decomposition')
                    st.write(additive_table)
                    multiplicative_fig = stock.plot_seasonal_decompose(multiplicative, data, data['Date'], 'Multiplicative', feature)
                    st.write(f'{feature} Multiplicative Seasonal Decomposition')
                    st.plotly_chart(multiplicative_fig, use_container_width=True)
                    #st.write(multiplicative_fig)
            except ValueError:
                st.error('Please select a date range')
                #st.markdown(':red[Please select a **date range**]')




        # Preediction method selection

        sidebar.markdown('## Stock Prediction')
        prediction_method = sidebar.selectbox(
            'Select Prediction Method',
            ('Long Short Term Memory', 'Linear Regression'))

        if prediction_method == 'Long Short Term Memory':

            sidebar.write('Select Parameters')

            # ------------- Parameters - Choice -------------

            MODEL_TYPE = sidebar.selectbox(
                "Model Type",
                ('Bidirectional LSTM', 'Vanilla LSTM', 'Stacked LSTM'),
                key="MODEL_TYPE"
            )
            if MODEL_TYPE == 'Bidirectional LSTM':
                N_LAYERS = sidebar.number_input(
                    "Number of layers",
                    min_value=2,
                    max_value=10,
                    key="N_LAYERS",
                )
            else:
                N_LAYERS = 1

            DROPOUT = sidebar.number_input(
                "Dropout (This regularization can help the model not overfit our training data)", min_value=0.1, max_value=0.5,
                key="DROPOUT"
            )
            TEST_SIZE = sidebar.number_input(
                "The testing set rate, e.g  0.2 means 20% of the total dataset",
                min_value=0.1,
                max_value=0.9,
                key="TEST_SIZE",
            )

            EPOCHS = sidebar.number_input(
                "Number of epochs",
                min_value=1,
                key="EPOCHS",
            )
            BATCH_SIZE = sidebar.selectbox(
                "Batch size",
                (1, 8, 16, 32, 64, 128, 256, 512),
                key="BATCH_SIZE"
            )
            SEQ_LENGTH = sidebar.number_input(
                "Sequence length",
                min_value=1,
                max_value=150,
                key="SEQ_LENGTH",
            )

            UNITS = sidebar.number_input(
                "Number of units",
                min_value=50,
                max_value=500,
                key="UNITS",
            )
            FUTURE_DAYS = sidebar.number_input(
                "Future Days",
                min_value=1,
                key="FUTURE_DAYS",
            )
            # Optimizer Selection
            optimizer = sidebar.selectbox(
                'Select Optimizer',
                ('adam', 'RMSProp', 'sgd', 'adadelta'))

            # Loss Selection
            loss = sidebar.selectbox(
                'Select Loss',
                ('mean_squared_error', 'huber_loss'))

            # Optimizer Activation
            ACTIVATION = sidebar.selectbox(
                'Select activation',
                ('linear', 'relu'))



            submit = sidebar.button('Train Model')
            if submit:
                st.write(f'**Dataset for Training ({START} - {END}**)')
                data = data.reset_index()
                data['Date'] = pd.to_datetime(data['Date']).dt.date

                st.write(data[['Date', 'Close', 'Open', 'Low', 'High', 'Volume', 'Adj Close']])
                FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
                train_model(SYMBOL, FUTURE_DAYS, True, SEQ_LENGTH, TEST_SIZE, FEATURE_COLUMNS, loss, optimizer, DROPOUT, BATCH_SIZE, EPOCHS, START, END, MODEL_TYPE, int(N_LAYERS), ACTIVATION, UNITS)
                    #prepare train test
                    #test_train_LSTM(data, TEST_SIZE, EPOCHS, BATCH_SIZE, optimizer, loss, SYMBOL)

        elif prediction_method == 'Linear Regression':

            sidebar.write('Select Parameters')
            TEST_SIZE = sidebar.number_input(
                "The testing set rate, e.g  0.2 means 20% of the total dataset",
                min_value=0.1,
                max_value=0.9,
                key="TEST_SIZE",
            )

            # Feature Selection
            feature = sidebar.selectbox(
                'Select Feature to train model on',
                ('Close', 'Open', 'Low', 'High', 'Volume', 'Adj Close'))
            submit = sidebar.button('Train Model')
            if submit:
                st.write(f'**Dataset for Training ({START} - {END}**)')
                data = data.reset_index()
                data['Date'] = pd.to_datetime(data['Date']).dt.date
                st.write(data[['Date', 'Close', 'Open', 'Low', 'High', 'Volume', 'Adj Close']])
                data = data.set_index('Date')

                linear_regression_tab(data, TEST_SIZE, SYMBOL, feature, START, END)
                # liner_regression_model = Linear_Regression(data, TEST_SIZE)
                # prediction_data = liner_regression_model.create_model()
                # test_data = liner_regression_model.create_train_test_data()[1]
                # prediction_plot(prediction_data, test_data, SYMBOL)



