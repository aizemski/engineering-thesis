#https://www.analyticsvidhya.com/blog/2018/09/multivariate-time-series-guide-forecasting-modeling-python-codes/
#https://towardsdatascience.com/analyzing-electricity-price-time-series-data-using-python-time-series-decomposition-and-price-4cd61924ef49
from statsmodels.tsa.api import VAR
def var_predict(history,seq_len):
    
    
    
    model = VAR(history)
    history = history.values
    model_fit = model.fit(1)
    forecast = model_fit.forecast(y=history,steps=1)
    return forecast[0][0]

