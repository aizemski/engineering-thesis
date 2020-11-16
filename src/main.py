import matplotlib.pyplot as plt
from tickers import wig_20_stocks_tickers
from get_stock_data import save_ticker
from lstm import train,load,prepare_data,inverse
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_models(ticker,seq_len,epochs,k_flod):
    path = './../data/models/'    
    models = []
    for i in range(k_flod):
        models.append(load(path+'/'+str(ticker)+'/'+str(i)+'_'+str(seq_len)+'_'+str(epochs)+'.h5'))
    return models


def wheter_to_buy(current_price,predictions,commission=0.3):
    how_may_votes = len(predictions)
    to_buy =0 
    
    for i in range(how_may_votes):
        if ((predictions[i]/current_price)-1)*100 > commission*2:
            to_buy+=1
    if to_buy > how_may_votes//2:
        return True
    return False


def buy(current_price,final_price,commision=0.3,min_commision=3,how_many=1,):
    # print(final_price,current_price)
    add = how_many*current_price*commision/100
    minus = how_many*final_price*commision/100
    if add < min_commision:
        add = min_commision
    if minus <min_commision:
        minus = min_commision
 

    return (how_many*final_price-minus)/(how_many*current_price+add)


def evaluate(ticker,seq_len,epochs):
    model = load_models(ticker,seq_len,epochs,5)
    test_case =200
    
    for t in wig_20_stocks_tickers:
        fund = 100 # percents 
        fund_without_com=100
        if t =='ALE':#does not have enough data
            continue
        x_data, y_data = prepare_data(t,'./../data/stocks/',seq_len)
        x_data = x_data[-test_case:]
        y_data = y_data[-test_case:]
        y_data = inverse(y_data,ticker)
        # plt.figure(figsize=(14, 5))
        # plt.plot(y_data[:,0], color='red',label='Real stock price')
        result=np.zeros((test_case,))
        buying = []
    
        for i in range(5):
            y_pred = model[i].predict(x_data)   
            y_pred = np.c_[y_pred,np.zeros(y_pred.shape)]
            y_pred_inverse = inverse(y_pred,ticker)[:,0]

            # print(y_pred_inverse)
            result += y_pred_inverse
            buying.append(y_pred_inverse)
            # plt.plot(inverse(y_pred,ticker)[:,0],label='Predicted stock price')
        
        efficiency = 0
        transactions =0
        investing_result=[]
        fund_status=[]
        ticker_price=[]
        old_fund = fund
        for i in range(len(buying[0])):
            predictions=[]
            for j in range(5):
                predictions.append(buying[j][i])
            ticker_price.append(y_data[i][0]/y_data[0][0])
            if i>0 and wheter_to_buy(y_data[i-1][0],predictions,0):
                
                fund*=buy(y_data[i-1][0],y_data[i][0],0.1,min_commision=0)
                fund_without_com*=buy(y_data[i-1][0],y_data[i][0],0,0,1)
                transactions+=1
                
                if (fund/old_fund>1):
                    efficiency+=1
                # elif fund/old_fund < 0.9: #stop 
                #     break     
            investing_result.append(fund_without_com/100)
            fund_status.append(fund/100)       
            old_fund = fund
        print(efficiency/transactions)
        print(transactions)
        print(fund_without_com)
        print(fund)
        print(fund*0.81)
        plt.plot(investing_result,label=t+"no comm")
        plt.plot(fund_status, label=t+'fund') 
        plt.plot(ticker_price, label='price change')
        plt.legend()
        plt.show()
    # plt.plot(result/5,label='Predicted stock price')
    # print(result/5)
    # print(y_data[-1])
    # plt.title('stock price prediction')
    # plt.xlabel('Time')
    # plt.ylabel('Stock price')
    # plt.legend()
    # plt.show()
    
# data=input('Download data (Y/N): ')
# if data == 'Y':
#     for ticker in wig_20_stocks_tickers:
#         save_ticker(ticker)

# data=input('Train all tickers (Y/N): ')
# if data == 'Y':
#     for ticker in wig_20_stocks_tickers:
#         if ticker !='ALE':
#             train(ticker)

# train('CCC',seq_len=1,epochs=20)
# train('CCC',seq_len=4,epochs=20)
# train('CCC',seq_len=5,epochs=20)
ticker ='CDR'
# train(ticker,seq_len=4,epochs=1002)


evaluate(ticker,4, epochs=1002)


