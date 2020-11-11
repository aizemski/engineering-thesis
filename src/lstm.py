import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import LSTM, Dense
from pathlib import Path


def load_data(ticker,path):
    df  = pd.read_csv(path+ticker+'.csv')
    df.set_index('Data', drop=True, inplace=True)
    return df [['Zamkniecie']]
def save_model(model,ticker,i,seq_len,epochs):
    path='./../data/models/'+ticker
    Path(path).mkdir(parents=True, exist_ok=True)  # creating dir if not exist
    model.save(path+'/'+str(i)+'_'+str(seq_len)+'_'+str(epochs)+'.h5')
    
def train(ticker,path='./../data/stocks/',k_fold=5,seq_len=20,batch_size=50,epochs=100):
    
    data = load_data(ticker,path)
    
    #why log returns https://quantivity.wordpress.com/2011/02/21/why-log-returns/
    data['zwrot_log'] =data.Zamkniecie.pct_change()
    data['zwrot_log'] = np.log(1+data['zwrot_log'])
    
    # scale data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    
    #prepare data
    x_data=[]
    y_data=[]
    for i in range(seq_len+1,len(scaled_data)):
        x_data.append(scaled_data[i-seq_len:i,:scaled_data.shape[1]])
        y_data.append(scaled_data[i])
    x_data= np.array(x_data)
    y_data = np.array(y_data)

    #K fold
    skf = KFold(n_splits=k_fold, shuffle=False)
    i=0
    for train, test in skf.split(x_data):
        # print(train,test)
        x_train, x_test = x_data[test],x_data[train]
        y_train, y_test = y_data[test],y_data[train]
        
        #create model 
        model = Sequential()
        model.add(LSTM(units=128, activation='relu',return_sequences=True,
                input_shape=(x_train.shape[1],x_train.shape[2])))
        model.add(LSTM(units=128,activation='relu'))
        model.add(Dense(units=1))
        model.compile(optimizer='adam',loss='mean_squared_error')

        #train model
        history = model.fit(x_train,y_train, epochs=epochs,batch_size=batch_size,
                validation_data=(x_test,y_test),verbose=1)

        #save trained model
        save_model(model,ticker,i,seq_len,epochs)


        i+=1
