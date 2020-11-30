import matplotlib.pyplot as plt
from tickers import wig_20_stocks_tickers
from get_stock_data import save_ticker
from lstm import train,load,prepare_data,inverse
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from trade import *
from data import *

def load_models(ticker,seq_len,epochs,k_flod):
    path = './../data/models/'    
    models = []
    for i in range(k_flod):
        models.append(load(path+'/'+str(ticker)+'/'+str(i)+'_'+str(seq_len)+'_'+str(epochs)+'.h5'))
    return models





def evaluate(ticker,seq_len,epochs):
    # print(ticker)
    model = load_models(ticker,seq_len,epochs,5)
    test_case =200
    total_return=0
    
    fund_return=0

    fund = 100 # percents 
    fund_without_com=100
    
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
    investing_result=[]
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
            
        investing_result.append(fund_without_com/100)
        fund_status.append(fund/100)       
        old_fund = fund
    
    total_return+=fund
    fund_return+=ticker_price[-1]/ticker_price[0]
    
    # plt.plot(investing_result,label=ticker+"no comm")
    # plt.plot(fund_status, label=ticker+'fund') 
    # plt.plot(ticker_price, label='price change')
    # plt.legend()
    # plt.show()
    return fund_return*100,total_return

    
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
        a,b=evaluate(ticker,4,21)
        c=b
        if b>100:
            c= 100 + (b-100)*0.81
        print('{} {} {} {}'.format(ticker,a,b,c))
        a_t.append(a)
        b_t.append(b)

# print(np.mean(a_t))
# print(np.mean(b_t))

