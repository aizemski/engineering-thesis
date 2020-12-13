import warnings
from tickers import wig_20_stocks_tickers
from get_stock_data import save_ticker
from lstm import evaluate_lstm
from data import *
from arima import evaluate_arima
from var import evaluate_var
from lasso import evaluate_lasso
   
data=input('Download data (Y/N): ')
if data == 'Y':
    for ticker in wig_20_stocks_tickers:
        save_ticker(ticker)

data=input('Train all tickers (Y/N): ')
if data == 'Y':
    for ticker in wig_20_stocks_tickers:
        if ticker !='ALE':
            train(ticker,seq_len=4,epochs=21)

for ticker in wig_20_stocks_tickers:
    if ticker !='ALE':
        evaluate_lstm(ticker,4,21)
        evaluate_arima(ticker,1)
        evaluate_var(ticker,1)
        evaluate_lasso(ticker,1)

