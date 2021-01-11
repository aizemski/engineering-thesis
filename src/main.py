import warnings
import matplotlib.pyplot as plt
from tickers import wig_20_stocks_tickers
from get_stock_data import save_ticker
from lstm import evaluate_lstm, train
from data import *
from arima import evaluate_arima
from var import evaluate_var
from lasso import evaluate_lasso


def evaluate(commission,display_plots = 0):
    lstm_fund=[]
    arima_fund=[]
    var_fund=[]
    lasso_fund=[]   
    test_case=246 
    for ticker in wig_20_stocks_tickers:
        
        if ticker !='ALE':
            #-6
            
            _,_,fund_status=evaluate_lstm(ticker,4,21,commission=commission,display_plots=display_plots,test_case=test_case-6)
            lstm_fund.append(fund_status)

            _,_,fund_status=evaluate_arima(ticker,1,commission=commission,display_plots=display_plots,test_case=test_case)
            arima_fund.append(fund_status)

            _,_,fund_status=evaluate_var(ticker,1,commission=commission,display_plots=display_plots,test_case=test_case)
            var_fund.append(fund_status)

            _,_,fund_status=evaluate_lasso(ticker,1,commission=commission,display_plots=display_plots,test_case=test_case)
            lasso_fund.append(fund_status)

    wig20 = load_data('wig20',test_case=test_case)[-test_case+7:]
    

    plt.title('Prowizja {}%'.format(commission))
    plt.ylabel('Zwrot (%)')
    plt.xlabel('Czas')
    plt.plot(100*np.mean(lstm_fund,axis=0), label=' LSTM WIG20') 
    plt.plot(100*np.mean(arima_fund,axis=0), label='ARIMA WIG20')
    plt.plot(100*np.mean(var_fund,axis=0), label='VAR WIG20')
    plt.plot(100*np.mean(lasso_fund,axis=0), label='LASSO WIG20')
    plt.plot(100*wig20['Zamkniecie']/wig20['Zamkniecie'][0], label='indeks WIG20')
    plt.xticks(np.arange(0, len(wig20), test_case/5))
    plt.legend()
    plt.savefig('../data/plots/Rezultat_{}%.pdf'.format(commission))
    plt.close()
    
    # save funds - wig20
    plt.title('Prowizja {}%'.format(commission))
    plt.ylabel('Zwrot (%)')
    plt.xlabel('Czas')
    plt.plot(100*np.mean(lstm_fund,axis=0)-100*wig20['Zamkniecie']/wig20['Zamkniecie'][0], label=' LSTM WIG20 - indeks WIG20') 
    plt.plot(100*np.mean(arima_fund,axis=0)-100*wig20['Zamkniecie']/wig20['Zamkniecie'][0], label='ARIMA WIG20 - indeks WIG20')
    plt.plot(100*np.mean(var_fund,axis=0)-100*wig20['Zamkniecie']/wig20['Zamkniecie'][0], label='VAR WIG20 - indeks WIG20')
    plt.plot(100*np.mean(lasso_fund,axis=0)-100*wig20['Zamkniecie']/wig20['Zamkniecie'][0], label='LASSO WIG20 - indeks WIG20')    
    plt.xticks(np.arange(0, len(wig20), test_case/5))
    plt.legend()
    plt.savefig('../data/plots/Rezultat_funds-wig_{}%.pdf'.format(commission))
    plt.close()

 

display_plots = 0
data=input('Download data (Y/N): ')
if data == 'Y':
    save_ticker('wig20')
    for ticker in wig_20_stocks_tickers:
        save_ticker(ticker)

data=input('Train all tickers (Y/N): ')
if data == 'Y':
    for ticker in wig_20_stocks_tickers:
        if ticker !='ALE':
            train(ticker,seq_len=4,epochs=25)
data=input('Display plots for every ticke (Y/N): ')

if data == 'Y':
    display_plots = 1            
#0% commison
evaluate(0.0,display_plots)

# 0.13% 
evaluate(0.13,display_plots)

#0.3% commison
evaluate(0.3,display_plots)

print('Finished')
