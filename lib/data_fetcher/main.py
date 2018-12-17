import quandl
import csv
import datetime
import pandas as pd

"""
Connects to the Quandl Api and fetches the stock data
*@ param: none
*@ return: Data Frame(df)
"""
def get_stock_data_quandl():
    quandl.ApiConfig.api_key = "swTGxurFd5MX4_sGjmj9"
    df = quandl.get('WIKI/GOOGL')
    return df

def clean_data_quandl(data_frame):
    target = data_frame['Adj. Open'][2:]
    target = list(target)

    data_frame = data_frame.ix[:-2]
    data_frame['Target'] = target

    del data_frame['Open']
    del data_frame['High']
    del data_frame['Low']
    del data_frame['Close']
    del data_frame['Volume']
    del data_frame['Split Ratio']
    del data_frame['Ex-Dividend']

    data_frame = data_frame.reset_index()

    date = data_frame['Date']

    del data_frame['Date']


    return [data_frame, target, date]


"""
Fetching to the historical data from google finance
*@ param: symbol (company code)
*@ return: Data Frame(df)
"""
def get_historical_data(symbol, start_date, end_date):
    ''' Daily quotes from Google. Date format='yyyy-mm-dd' '''
    symbol = symbol.upper()
    start = datetime.date(int(start_date[0:4]), int(start_date[5:7]), int(start_date[8:10]))
    end = datetime.date(int(end_date[0:4]), int(end_date[5:7]), int(end_date[8:10]))
    url_string = "http://www.google.com/finance/historical?q={0}".format(symbol)
    url_string += "&startdate={0}&enddate={1}&num={0}&ei=KKltWZHCBNWPuQS9147YBw&output=csv".format(start.strftime('%b%d,%Y'), end.strftime('%b%d,%Y'),4000)
    col_names = ['Date','Open','High','Low','Close','Volume']
    stocks = pd.read_csv(url_string, header=0, names=col_names) 

    df = pd.DataFrame(stocks)
    return df