from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from data import *
from trade import wheter_to_buy


def arima_predict(history):
    history = [x for x in history]
    model = ARIMA(history,order=(0,0,4))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()[0]
    return output


def evaluate_arima(ticker,seq_len,test_case=200,commission=0.3,display_plots=0):
    data = load_data(ticker,test_case=test_case+seq_len*5)
    
    # przygotowanie przewidywan
    predictions =[]
    for i in range(5*seq_len-2,test_case+4*seq_len-1):
        current = data['zwrot'][:seq_len+i+1]     
        result= arima_predict(current)
        predictions.append(result)
    
    raw_data = load_raw_data(ticker,test_case=test_case+seq_len*5)[-test_case+seq_len:]
    fund_return=0
    fund = 100 # procenty

    # symulacja handlu na gieldzie  
    fund_status=[]
    ticker_price =[]
    for i in range(len(predictions)-seq_len):
        current_prediction = data['Zamkniecie'][i]*(1+predictions[i][0])
        ticker_price.append(data['Zamkniecie'][i]/data['Zamkniecie'][0])
        fund*= wheter_to_buy(raw_data[i],[current_prediction],commission)
        fund_status.append(fund/100)       

    return fund_status