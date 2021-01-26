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
    test_case=250
    for ticker in wig_20_stocks_tickers:
        
        if ticker !='ALE':
            # Ewaluacja poszczegolnych podejsc
            _,_,fund_status=evaluate_lstm(ticker,4,21,commission=commission,display_plots=display_plots,test_case=test_case-6)
            lstm_fund.append(fund_status)

            _,_,fund_status=evaluate_arima(ticker,4,commission=commission,display_plots=display_plots,test_case=test_case)
            arima_fund.append(fund_status)

            _,_,fund_status=evaluate_var(ticker,4,commission=commission,display_plots=display_plots,test_case=test_case)
            var_fund.append(fund_status)

            _,_,fund_status=evaluate_lasso(ticker,4,commission=commission,display_plots=display_plots,test_case=test_case)
            lasso_fund.append(fund_status)
            
            if display_plots:
                # Tworzenie wykresów dla poszczegolnych akcji
                ticker_price = load_data(ticker,test_case=test_case)[-test_case+7:]
                plt.title(ticker)
                plt.ylabel('Zwrot (%)')
                plt.xlabel('Czas')
                plt.plot(100*np.mean(lstm_fund,axis=0), label=' LSTM') 
                plt.plot(100*np.mean(arima_fund,axis=0), label='ARIMA')
                plt.plot(100*np.mean(var_fund,axis=0), label='VAR')
                plt.plot(100*np.mean(lasso_fund,axis=0), label='LASSO')
                plt.plot(100*ticker_price['Zamkniecie']/ticker_price['Zamkniecie'][0], label='Orginalne dane')
                plt.xticks(np.arange(0, len(ticker_price), test_case/5))
                plt.legend()
                plt.savefig('../data/plots/{}-{}%-{}days.png'.format(ticker,commission,test_case))
                plt.close()
    # Pobierania danych odnosnie wig20
    wig20 = load_data('wig20',test_case=test_case)[-test_case+7:]
    
    # Tworzenie wykresu sredniogo zwrotu na tle indeksu wig20
    plt.title('Średnia zwrotu')
    plt.ylabel('Zwrot (%)')
    plt.xlabel('Czas')
    plt.plot(100*np.mean(lstm_fund,axis=0), label=' LSTM') 
    plt.plot(100*np.mean(arima_fund,axis=0), label='ARIMA')
    plt.plot(100*np.mean(var_fund,axis=0), label='VAR')
    plt.plot(100*np.mean(lasso_fund,axis=0), label='LASSO')
    plt.plot(100*wig20['Zamkniecie']/wig20['Zamkniecie'][0], label='indeks WIG20')
    plt.xticks(np.arange(0, len(wig20), test_case/5))
    plt.legend()
    plt.savefig('../data/plots/Rezultat_{}%_{}days.png'.format(commission,test_case))
    plt.close()

    # Tworzenei wykresu 
    plt.title('Zwrot w porównaniu do indeksu WIG20')
    plt.ylabel('Zwrot (%)')
    plt.xlabel('Czas')
    plt.plot(100*np.mean(lstm_fund,axis=0)-100*wig20['Zamkniecie']/wig20['Zamkniecie'][0], label=' LSTM') 
    plt.plot(100*np.mean(arima_fund,axis=0)-100*wig20['Zamkniecie']/wig20['Zamkniecie'][0], label='ARIMA ')
    plt.plot(100*np.mean(var_fund,axis=0)-100*wig20['Zamkniecie']/wig20['Zamkniecie'][0], label='VAR')
    plt.plot(100*np.mean(lasso_fund,axis=0)-100*wig20['Zamkniecie']/wig20['Zamkniecie'][0], label='LASSO')    
    plt.xticks(np.arange(0, len(wig20), test_case/5))
    plt.legend()
    plt.savefig('../data/plots/Rezultat_funds-wig_{}%_{}days.png'.format(commission,test_case))
    plt.close()


display_plots = 0
data=''
data=input('Pobierz dane z rynku (T/N): ')
if data == 'T':
    save_ticker('wig20')
    for ticker in wig_20_stocks_tickers:
        save_ticker(ticker)

data=input('Stworz modele LSTM (T/N): ')
if data == 'T':
    for ticker in wig_20_stocks_tickers:
        if ticker !='ALE':
            train(ticker,seq_len=4,epochs=21)

data=input('Wygeneruj wykresy dla poszczegolnych akcji (wszystkie podejscia) (T/N): ')
if data == 'T':
    display_plots = 1   
print('Podaj  prowizje dla ktorej chcesz przetestować modele.')
print('Zalecane, 0.0 ')
comm1 = int(input('Podaj prowizje: '))

evaluate(comm1,display_plots)


print('Finished')
