import yfinance as yf
import datetime
import plotly.express as px


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

    def ticker(self):

        df_ticker = yf.Ticker(self.symbol)
        return df_ticker

    def stock_information(self):

        ticker = self.ticker()
        history_1mo = ticker.history(period='1mo')
        splits = ticker.splits
        return history_1mo, splits

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

        plotly_figure.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        return plotly_figure