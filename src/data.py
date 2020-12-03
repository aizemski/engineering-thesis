import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def inverse(data,ticker):
    X = load_data(ticker)
    scaler = MinMaxScaler(feature_range=(0,1)).fit(X)
    return scaler.inverse_transform(data)

def load_raw_data(ticker,path='./../data/stocks/'):
    df =pd.read_csv(path+ticker+'.csv')
    df.set_index('Data', drop=True, inplace=True)
    return df[['Otwarcie','Najwyzszy','Najnizszy', 'Zamkniecie']][-250:].rename_axis('ID').values

def load_data(ticker,path='./../data/stocks/'):
    df  = pd.read_csv(path+ticker+'.csv')
    df.set_index('Data', drop=True, inplace=True)
    #why log returns https://quantivity.wordpress.com/2011/02/21/why-log-returns/
    df['zwrot_log'] =df.Zamkniecie.pct_change()
    df['zwrot_log'] = np.log(1+df['zwrot_log'])
    return df [['Zamkniecie','zwrot_log']][-250:] # return last 12 months

def prepare_data(ticker,path,seq_len):
    data = load_data(ticker,path)
    
    # scale data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    
    #prepare data
    x_data=[]
    y_data=[]
    for i in range(seq_len+1,len(scaled_data)):
        x_data.append(scaled_data[i-seq_len:i,:scaled_data.shape[1]])
        y_data.append(scaled_data[i])
    return np.array(x_data),np.array(y_data)    



