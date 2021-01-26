from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from data import *
from trade import wheter_to_buy


def arima_predict(history):
    history = [x for x in history]
    model = ARIMA(history,order=(0,1,0))
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

    
    if display_plots:
        plt.plot(fund_status, label=ticker+' arima') 
        plt.plot(ticker_price, label='Zmiana ceny')
        plt.ylabel('Zwrot (%)')
        plt.xlabel('Czas')
        plt.legend()
        plt.savefig('../data/plots/arima_{}_{}%_{}days.pdf'.format(ticker,commission,test_case))
        plt.close()

    return fund_status