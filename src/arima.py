from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt

from data import *
from trade import wheter_to_buy




def arima_predict(history):
    history = [x for x in history]
    model = ARIMA(history,order=(1,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()[0]
    return output

def evaluate_arima(ticker,seq_len,test_case=200,commission=0.3,display_plots=0):
    data = load_data(ticker)[-(test_case+seq_len*5):]
    predictions =[]
    
    for i in range(5*seq_len,test_case-seq_len-1):
        current = data['zwrot'][:seq_len+i+1]     
        result= arima_predict(current)
        predictions.append(result)
    
    raw_data = load_raw_data(ticker)[-test_case+6*seq_len:]
    fund_return=0
    old_fund = fund = 100 # percents
    efficiency = 0
    transactions =0
    fund_status=[]
    ticker_price =[]
    
    for i in range(len(predictions)):
        current_prediction = data['Zamkniecie'][i-1+6*seq_len]*(1+predictions[i][0])
        ticker_price.append(data['Zamkniecie'][i-1+6*seq_len]/data['Zamkniecie'][6*seq_len-1])
        fund*= wheter_to_buy(raw_data[i],[current_prediction],commission)
        transactions+=1

        if (fund/old_fund>1):
            efficiency+=1    
        fund_status.append(fund/100)       
        old_fund = fund
    
    fund_return+=ticker_price[-1]/ticker_price[0]
    if display_plots:
        plt.plot(fund_status, label=ticker+' arima') 
        plt.plot(ticker_price, label='Zmiana ceny')
        plt.legend()
        plt.show()

    return fund_return*100,fund,fund_status