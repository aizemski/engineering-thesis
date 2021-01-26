import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from keras.models import Sequential,load_model
from keras.layers import LSTM, Dense, Dropout
from pathlib import Path
import matplotlib.pyplot as plt

from data import *
from trade import wheter_to_buy




def save(model,ticker,i,seq_len,epochs):
    path='./../data/models'+ticker
    Path(path).mkdir(parents=True, exist_ok=True)  # creating dir if not exist
    model.save(path+'/'+str(i)+'_'+str(seq_len)+'_'+str(epochs)+'.h5')


def load(path):
    return load_model(path)


def train(ticker,path='./../data/stocks/',k_fold=5,seq_len=20,batch_size=50,epochs=100):
    print('{} start'.format(ticker))
    
    x_data, y_data =  prepare_data(ticker,path,seq_len)

    #K fold
    skf = KFold(n_splits=k_fold, shuffle=True)
    i=0
    for train, test in skf.split(x_data):
        print('{}-{}'.format(ticker,i+1))
        x_train, x_test = x_data[train],x_data[test]
        y_train, y_test = y_data[train],y_data[test]
        
        #create model 
        model = Sequential()
        model.add(LSTM(units=128, activation='relu',return_sequences=True,
                input_shape=(x_train.shape[1],x_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=128,activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(units=1))
        model.compile(optimizer='adam',loss='mean_squared_error')

        #train model
        history = model.fit(x_train,y_train, epochs=epochs,batch_size=batch_size,
                validation_data=(x_test,y_test),verbose=0)

        #save trained model
        save(model,ticker,i,seq_len,epochs)

        i+=1
    print('{} stop'.format(ticker))


def load_models(ticker,seq_len,epochs,k_flod):
    path = './../data/models'    
    models = []
    for i in range(k_flod):
        models.append(load(path+'/'+str(ticker)+'/'+str(i)+'_'+str(seq_len)+'_'+str(epochs)+'.h5'))
    return models


def evaluate_lstm(ticker,seq_len,epochs,test_case=194,commission=0.3,display_plots=0):
    model = load_models(ticker,seq_len,epochs,5)
    fund_return=0
    fund = 100 # percents 
    
    x_data, y_data = prepare_data(ticker,'./../data/stocks/',seq_len,test_case=test_case+seq_len+1)
    x_data = x_data[-test_case:]
    y_data = y_data[-test_case:]
    y_data = inverse(y_data,ticker)
    raw_data = load_raw_data(ticker,test_case=test_case)[-test_case:]
    
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
        fund*= wheter_to_buy(raw_data[i],predictions,commission)
        transactions+=1
        
        if (fund/old_fund>1):
            efficiency+=1
            
        fund_status.append(fund/100)       
        old_fund = fund
    
    fund_return+=ticker_price[-1]/ticker_price[0]
    if display_plots:
    
        plt.plot(fund_status, label=ticker+' lstm') 
        plt.plot(ticker_price, label='Zmiana ceny')
        plt.ylabel('Zwrot (%)')
        plt.xlabel('Czas')
        plt.legend()
        plt.savefig('../data/plots/lstm_{}_{}%_{}days.pdf'.format(ticker,commission,test_case))
        plt.close()
    return fund_return*100,fund,fund_status
