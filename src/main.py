import warnings
import matplotlib.pyplot as plt
from tickers import wig_20_stocks_tickers
from get_stock_data import save_ticker
from lstm import train,load,prepare_data,inverse
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from trade import *
from data import *
from arima import arima_predict



def load_models(ticker,seq_len,epochs,k_flod):
    path = './../data/models/'    
    models = []
    for i in range(k_flod):
        models.append(load(path+'/'+str(ticker)+'/'+str(i)+'_'+str(seq_len)+'_'+str(epochs)+'.h5'))
    return models



def evaluate_arima(ticker,seq_len,test_case=200):
    data = load_data(ticker)[-(test_case+seq_len*5):]
    
    # current = data[:seq_len]
    predictions =[]
    # arima_predict(current,seq_len)
    for i in range(5*seq_len,test_case-seq_len-1):
        current = data['zwrot_log'][:seq_len+i+1]     
        result= arima_predict(current,seq_len)
        predictions.append(result)
    
    fund_return=0
    old_fund = fund = 100 # percents
    efficiency = 0
    transactions =0
    fund_status=[]
    # for i in range(len(predictions)):
    #     fund*= wheter_to_buy(data[i],predictions)
    #     transactions+=1

    #     if (fund/old_fund>1):
    #         efficiency+=1    
    #     print(fund)
    #     fund_status.append(fund/100)       
    #     old_fund = fund
    print(len(predictions),len(current)-seq_len*5)
    plt.plot(predictions, label=ticker+'prediction') 
    plt.plot(current[seq_len*6:], label='price change')
    plt.legend()
    plt.show()

    return fund_return*100,fund

def evaluate_lstm(ticker,seq_len,epochs,test_case=200):
    model = load_models(ticker,seq_len,epochs,5)
    fund_return=0
    fund = 100 # percents 
    
    x_data, y_data = prepare_data(ticker,'./../data/stocks/',seq_len)
    x_data = x_data[-test_case:]
    y_data = y_data[-test_case:]
    y_data = inverse(y_data,ticker)
    raw_data = load_raw_data(ticker)[-test_case:]
    
    result=np.zeros((test_case,))
    buying = []

    for i in range(5):
        y_pred = model[i].predict(x_data)   
        y_pred = np.c_[y_pred,np.zeros(y_pred.shape)]
        y_pred_inverse = inverse(y_pred,ticker)[:,0]
        result += y_pred_inverse
        buying.append(y_pred_inverse)
    
    efficiency = 0
    transactions =0

    fund_status=[]
    ticker_price=[]
    old_fund = fund

    for i in range(1,len(buying[0])):
        predictions=[] 
        for j in range(5):
            predictions.append(buying[j][i])
        ticker_price.append(y_data[i][0]/y_data[0][0])
        # print(fund,end=' ')
      
        fund*= wheter_to_buy(raw_data[i],predictions)

        transactions+=1
        
        if (fund/old_fund>1):
            efficiency+=1
            
        fund_status.append(fund/100)       
        old_fund = fund
    
    fund_return+=ticker_price[-1]/ticker_price[0]
    
    # plt.plot(fund_status, label=ticker+'fund') 
    # plt.plot(ticker_price, label='price change')
    # plt.legend()
    # plt.show()
    return fund_return*100,fund

    
# data=input('Download data (Y/N): ')
# if data == 'Y':
#     for ticker in wig_20_stocks_tickers:
#         save_ticker(ticker)

# data=input('Train all tickers (Y/N): ')
# if data == 'Y':
#     for ticker in wig_20_stocks_tickers:
#         if ticker !='ALE':
#             train(ticker,seq_len=4,epochs=21)
a_t=[]
b_t=[]
for ticker in wig_20_stocks_tickers:
    if ticker !='ALE':
        a,b=evaluate_arima(ticker,1)#(ticker,4,21)
        c=b
        if b>100:
            c= 100 + (b-100)*0.81
        print('{} {} {} {}'.format(ticker,a,b,c))
        a_t.append(a)
        b_t.append(b)

# print(np.mean(a_t))
# print(np.mean(b_t))

