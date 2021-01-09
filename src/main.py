import warnings
import matplotlib.pyplot as plt
from tickers import wig_20_stocks_tickers
from get_stock_data import save_ticker
from lstm import evaluate_lstm
from data import *
from arima import evaluate_arima
from var import evaluate_var
from lasso import evaluate_lasso




def evaluate(commission,display_plots = 0):
    lstm_fund=[]
    arima_fund=[]
    var_fund=[]
    lasso_fund=[]    
    for ticker in wig_20_stocks_tickers:
        
        if ticker !='ALE':
            _,_,fund_status=evaluate_lstm(ticker,4,21,commission=commission,display_plots=display_plots)
            lstm_fund.append(fund_status)

            _,_,fund_status=evaluate_arima(ticker,1,commission=commission,display_plots=display_plots)
            arima_fund.append(fund_status)

            _,_,fund_status=evaluate_var(ticker,1,commission=commission,display_plots=display_plots)
            var_fund.append(fund_status)

            _,_,fund_status=evaluate_lasso(ticker,1,commission=commission,display_plots=display_plots)
            lasso_fund.append(fund_status)

    wig20 = load_data('wig20')[-194:]
    plt.title('Commission {}%'.format(commission))
    plt.ylabel('Return (%)')
    plt.xlabel('Time')
    plt.plot(100*np.mean(lstm_fund,axis=0), label=' LSTM WIG20') 
    plt.plot(100*np.mean(arima_fund,axis=0), label='ARIMA WIG20')
    plt.plot(100*np.mean(var_fund,axis=0), label='VAR WIG20')
    plt.plot(100*np.mean(lasso_fund,axis=0), label='LASSO WIG20')
    plt.plot(100*wig20['Zamkniecie']/wig20['Zamkniecie'][0], label='WIG20')
    plt.xticks(np.arange(0, len(wig20), 45.0))

    plt.legend()
    # plt.savefig('Rezultat_{}%.pdf'.format(commission))
    plt.close()

data=input('Download data (Y/N): ')
if data == 'Y':
    save_ticker('wig20')
    for ticker in wig_20_stocks_tickers:
        save_ticker(ticker)

data=input('Train all tickers (Y/N): ')
if data == 'Y':
    for ticker in wig_20_stocks_tickers:
        if ticker !='ALE':
            train(ticker,seq_len=4,epochs=21)
data=input('Display plots for every ticke (Y/N): ')
display_plots = 0
if data == 'Y':
    displat_plots = 1            
#0% commison
evaluate(0.0,displat_plots)

#0.13% 
evaluate(0.13,displat_plots)

#0.3% commison
evaluate(0.3,displat_plots)

print('Finished')
