from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt

from data import *
from trade import wheter_to_buy


def var_predict(history,seq_len):    
    model = VAR(history)
    history = history.values
    model_fit = model.fit(1)
    forecast = model_fit.forecast(y=history,steps=1)
    return forecast[0][0]


def evaluate_var(ticker,seq_len,test_case=200):
    data = load_data(ticker)[-(test_case+seq_len*5):]
    
    predictions =[]
    for i in range(5*seq_len,test_case-seq_len-1):
        current = data[:seq_len+i+1] 
        result= var_predict(current,seq_len)
        predictions.append(result)
    raw_data = load_raw_data(ticker)[-test_case+6*seq_len:]
    fund_return=0
    old_fund = fund = 100 # percents
    efficiency = 0
    transactions =0
    fund_status=[]
    ticker_price =[]
    
    for i in range(len(predictions)):

        ticker_price.append(data['Zamkniecie'][i-1+6*seq_len]/data['Zamkniecie'][6*seq_len-1])
        fund*= wheter_to_buy(raw_data[i],[predictions[i]])
        transactions+=1

        if (fund/old_fund>1):
            efficiency+=1    
        fund_status.append(fund/100)       
        old_fund = fund
    
    fund_return+=ticker_price[-1]/ticker_price[0]

    plt.plot(fund_status, label=ticker+' var fund') 
    plt.plot(ticker_price, label='price change')
    plt.legend()
    plt.show()

    return fund_return*100,fund

