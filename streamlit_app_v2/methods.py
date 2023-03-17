import datetime as dt

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