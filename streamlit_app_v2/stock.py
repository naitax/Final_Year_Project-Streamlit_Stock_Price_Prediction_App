import yfinance as yf
import datetime
import plotly.express as px

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#to visualise decomposition of time series
from plotly.subplots import make_subplots
import pandas as pd
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose
from plotly import graph_objects as go

# Add validation
# end date cannot be later than today
# start date cannot be earlier than 2010
class Stock:
    """
    This class enables data loading, plotting and statistical analysis of a given stock,
     upon initialization load a sample of data to check if stock exists.

    """

    def __init__(self, symbol="GOOGL"):

        self.end = datetime.datetime.today()
        self.start = self.end - datetime.timedelta(days=4)
        self.symbol = symbol
        self.data = self.load_data(self.start, self.end)

    def load_data(self, start, end, inplace=False):
        """
        takes a start and end dates, download data do some processing and returns dataframe
        """

        data = yf.download(self.symbol, start, end)
        # Check if there is data
        try:
            assert len(data) > 0
        except AssertionError:
            print("Cannot fetch data, check spelling or time window")
        data.reset_index(inplace=True)
        #         data.rename(columns={"Date": "datetime"}, inplace=True)
        #         data["date"] = data.apply(lambda raw: raw["datetime"].date(), axis=1)

        #         data = data[["date", 'Close']]
        return data

    def ticker(self) -> str:

        df_ticker = yf.Ticker(self.symbol)
        return df_ticker

    def stock_information(self) -> list:

        ticker = self.ticker()
        history_1mo = ticker.history(period='1mo')
        splits = ticker.splits
        return history_1mo, splits

    def calculate_volatility(self, data) -> float:

        # create a column called Log returns with the daily log return of the Close price.
        data['Log returns'] = np.log(data['Close'] / data['Close'].shift())

        # get standard deviation
        std = data['Log returns'].std()

        # calculate volatility
        volatility = std * 252 ** .5

        return volatility

    # def visualise_volatility(self, data, volatility, symbol):
    #
    #     str_vol = str(round(volatility, 4) * 100)
    #     fig, ax = plt.subplots()
    #     data['Log returns'].hist(ax=ax, bins=50, alpha=0.6, color='b')
    #     ax.set_xlabel('Log return')
    #     ax.set_ylabel('Freq of log return')
    #     ax.set_title(f'{symbol} volatility: {str_vol} %')
    def visualise_volatility(self, data, volatility, symbol):

        str_vol = str(round(volatility, 4) * 100)
        fig = px.histogram(data, x=data['Log returns'],
                           title=f'{symbol} volatility: {str_vol} %')

        fig.update_layout(
            xaxis_title='Log return',
            yaxis_title='Freq of log return'
        )
        return fig

    #correlation
    def visualise_correlation(self, data):

        fig, ax = plt.subplots()
        corr = data.corr()
        sns.heatmap(corr, ax=ax, annot=True)
        return fig

    #decomposition of time series
    def decompose_time_series(self, data, feature) -> list:
        """
        A function that returns the trend, seasonality and residual captured by applying both multiplicative and
        additive model.
        df -> DataFrame
        column_name -> column_name for which trend, seasonality is to be captured
        """

        data = data.set_index('Date')
        result_multiplicative = seasonal_decompose(data[feature], model='multiplicative', extrapolate_trend='freq',
                                                   period=30)
        result_additive = seasonal_decompose(data[feature], model='additive', extrapolate_trend='freq', period=30)

        return result_multiplicative, result_additive

    def plot_seasonal_decompose(self, result, data, x_values, model, feature):

        """
        result: result_multiplicative or result_additive
        data: data passes
        x_values: date
        model: Additive or Multiplicative, used in the title
        feature: column_name (open, close, high..)
        """

        return (
            make_subplots(
                rows=2,
                cols=2,
                subplot_titles=["Observed Series", "Trend", "Seasonality", "Residuals"],
            )
                .add_trace(
                go.Scatter(x=x_values, y=result.observed, mode="lines", name='Observed'),
                row=1,
                col=1,
            )
                .add_trace(
                go.Scatter(x=x_values, y=result.trend, mode="lines", name='Trend'),
                row=1,
                col=2,
            )
                .add_trace(
                go.Scatter(x=x_values, y=result.seasonal, mode="lines", name='Seasonal'),
                row=2,
                col=1,
            )
                .add_trace(
                go.Scatter(x=x_values, y=result.resid, mode="lines", name='Residual'),
                row=2,
                col=2,
            )
                .update_layout(
                height=900, title=f'', title_x=0.5,
                showlegend=False
            )
        )

    def additive_decomposing_table(self, result_additive):

        # table can be only created with additive
        df_reconstructed = pd.concat(
            [result_additive.seasonal, result_additive.trend, result_additive.resid, result_additive.observed], axis=1)
        df_reconstructed.columns = ['Seasonality', 'Trend', 'Residuals', 'Actual Values']
        # create a list of columns and remove the ones that shouldnt be added together
        temp_list = list(df_reconstructed)
        temp_list.remove('Actual Values')
        # add a new column with sum of seas, trend and resid
        df_reconstructed['Sum of seas, trend and resid'] = df_reconstructed[temp_list].sum(axis=1)

        return df_reconstructed

# e.g
# symbol = 'AAPL'
# stock = Stock(symbol)
# data = stock.load_data('2015-03-01', '2022-01-01')
# volatility = stock.calculate_volatility(data)
# stock.visualise_volatility(data, volatility, symbol)


    def plot_row_data(self, fig, start, end):
        """
        Plot time-serie line chart of closing price on a given plotly.graph_objects.Figure object
        """
        data = self.load_data(start, end)
        figure = px.line(self.data, x=data['Date'], y=data['Close'],
                         title='')
        figure.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        #         fig = fig.add_trace(
        #             go.Scatter(
        #                 x=self.data.date,
        #                 y=self.data['Close'],
        #                 mode="lines",
        #                 name=self.symbol,
        #             )
        #         )
        return figure

    def plot_multiple(self, data, choice='Open'):

        df_ticker = self.ticker()
        numeric_df = data.select_dtypes(['float', 'int'])
        cust_data = data[choice]


        plotly_figure = px.line(data_frame=cust_data, x=data['Date'], y=choice,
                                title=f'')

        # plotly_figure.update_xaxes(
        #     rangeselector=dict(
        #         buttons=list([
        #             dict(count=1, label="1m", step="month", stepmode="backward"),
        #             dict(count=6, label="6m", step="month", stepmode="backward"),
        #             dict(count=3, label="3m", step="month", stepmode="backward"),
        #             dict(count=1, label="1y", step="year", stepmode="backward"),
        #             dict(step="all")
        #         ])
        #     )
        # )

        """
        Add date range slider
        """
        plotly_figure.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=3,
                             label="3m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )
        return plotly_figure