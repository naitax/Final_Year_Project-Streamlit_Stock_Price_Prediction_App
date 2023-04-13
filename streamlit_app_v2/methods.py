import datetime as dt
from datetime import date
from stocksymbol import StockSymbol
#creating/updating csv file
import os.path
import csv
from csv import writer
import pandas as pd


def check_if_today_greater(date, today):
    if date > today:
        return True

def get_today():
    #dt.datetime.now().strftime("%Y-%m-%d")
    return date.today()

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

def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

def get_all_ticker_names():
  api_key = '55af380e-652d-407a-8d08-bea00df715eb'

  ss = StockSymbol(api_key)

  # get symbol list based on market
  symbol_list_us = ss.get_symbol_list(market="US") # "us" or "america" will also work
  all_ticker = []
  for i in range(0, len(symbol_list_us)):
    all_ticker.append(symbol_list_us[i]['symbol'])

  all_ticker = sorted(all_ticker)
  return all_ticker


def save_prediction_models(prediction_model, symbol, start, end, feature, test_size, intercept, mea, mse, rmse, r2):
    path_to_file = 'lr_models.csv'
    data_to_append = [prediction_model, symbol, f'{start}-{end}', feature, test_size, intercept, mea, mse, rmse, r2]
    header_list = ['Prediction Model', 'Stock', 'Date Range', 'Feature', 'Test Size', 'intercept', 'mea', 'mse', 'rmse',
                   'r2']
    if os.path.exists(path_to_file) is False:
        with open(path_to_file, 'w', newline='') as file:
            writer = csv.writer(file)
            dw = csv.DictWriter(file, delimiter=',', fieldnames=header_list)
            dw.writeheader()
            writer.writerow(data_to_append)
    else:
        with open(path_to_file, 'a') as file:
            writer = csv.writer(file)
            # writer_object = writer(file)
            # writer_object.writerow(data_to_append)
            writer.writerow(data_to_append)

    df = pd.read_csv(path_to_file)
    file.close()
    return df
