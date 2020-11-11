from tickers import wig_20_stocks_tickers
from get_stock_data import save_ticker


data=input('Download data (Y/N): ')
if data == 'Y':
    for ticker in wig_20_stocks_tickers:
        save_ticker(ticker)