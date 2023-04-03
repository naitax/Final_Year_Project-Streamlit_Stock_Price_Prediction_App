import datetime as dt
from stocksymbol import StockSymbol


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

