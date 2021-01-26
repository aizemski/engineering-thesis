def wheter_to_buy(data,predictions,commission=0.3):
    
    how_may_votes = len(predictions)# wartosc 1 dla VAR, LASSO, ARIMA, 5 dla LSTM
    to_buy =0 
    open_pirce = data[0] 
    
    # Sprawdzanie warunku zakupu
    for i in range(how_may_votes):
        if (predictions[i]*(1-commission/100)-(open_pirce)*(1+commission/100)) >0. :
            to_buy+=1

    if to_buy > how_may_votes//2 :
        return buy(data,commision=commission,take_profit=max(predictions)) 
    return 1.0


def buy(data,commision=0.3,stop_loss=0.95,take_profit=0):
    
    buy_price = data[0] # cena otwarcia
    high_price = data[1]
    sell_price  = data[3] # cena zamkniecia
    
    if high_price>= take_profit: # sprawdzanie czy przewidziana wartosc pojawiała się w ciagu notowan
        sell_price = take_profit

    add = buy_price*commision/100 # prowizja przy zakupie
    minus = sell_price*commision/100 # prowizja przy sprzedazy

    
    return (sell_price-minus)/(buy_price+add)
