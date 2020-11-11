import requests 
import os
from pathlib import Path
DIRECTORY = './../data/stocks'

def save_ticker(ticker, path=DIRECTORY):
    Path(path).mkdir(parents=True, exist_ok=True)  # creating dir if not exist
    data = get_data(ticker)

    with open(path+'/'+ticker+'.csv', 'wb+') as file:
        file.write(data)


def get_data(ticker):
    url = 'https://stooq.pl/q/d/l/?s='+ticker.replace('\n', '')+'&i=d'
    result = requests.get(url)
    return result.content




            