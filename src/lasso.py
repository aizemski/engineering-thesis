from sklearn import linear_model
import matplotlib.pyplot as plt

from data import *
from trade import wheter_to_buy


def lasso_predict(history,seq_len):    
    model = linear_model.Lasso(alpha=0.1)
    model_fit = model.fit(X=history,y=history)
    forecast = model_fit.predict(history[-1:])
    return forecast[0][0]

def evaluate_lasso(ticker,seq_len,test_case=200,commission=0.3,display_plots=0):
    data = load_data(ticker,test_case=test_case+seq_len*5)
    
    predictions =[]
    for i in range(5*seq_len,test_case-seq_len-1):
        current = data[i:seq_len+i+1] 
        result =lasso_predict(current,seq_len)
        predictions.append(result)   
    raw_data = load_raw_data(ticker,test_case=test_case)[-test_case+6*seq_len:]
    fund_return=0
    old_fund = fund = 100 # percents
    efficiency = 0
    transactions =0
    fund_status=[]
    ticker_price =[]
    
    for i in range(len(predictions)):

        ticker_price.append(data['Zamkniecie'][i-1+6*seq_len]/data['Zamkniecie'][6*seq_len-1])
        fund*= wheter_to_buy(raw_data[i],[predictions[i]],commission)
        transactions+=1

        if (fund/old_fund>1):
            efficiency+=1    
        fund_status.append(fund/100)       
        old_fund = fund
    
    fund_return+=ticker_price[-1]/ticker_price[0]
    if display_plots:
        plt.plot(fund_status, label=ticker+' lasso') 
        plt.plot(ticker_price, label='Zmiana ceny')
        plt.legend()
        plt.show()

    return fund_return*100,fund,fund_status

