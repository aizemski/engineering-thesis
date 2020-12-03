from statsmodels.tsa.arima_model import ARIMA


def arima_predict(history,seq_len):

    # print('HITORY',history,"ENT STORY")

    history = [x for x in history]
    model = ARIMA(history,order=(seq_len,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()[0]
    return output
