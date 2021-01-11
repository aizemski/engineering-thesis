import numpy as np

def wheter_to_buy(data,predictions,commission=0.3):
    how_may_votes = len(predictions)
    to_buy =0 
    open_pirce = data[0] 
    low_price = data[2]
    for i in range(how_may_votes):
        if (predictions[i]*(1-commission/100)-(open_pirce)*(1+commission/100)) >0. :
            to_buy+=1
    if to_buy > how_may_votes//2 :
        return buy(data,commision=commission,take_profit=np.mean(predictions))
    return 1.0


def buy(data,commision=0.3,stop_loss=0.95,take_profit=0):
    
    buy_price = data[0] 
    high_price = data[1]
    low_price = data[2]
    close_price = data[3]
    sell_price = close_price
    if high_price>= take_profit> close_price:
        sell_price = take_profit

    if sell_price <= stop_loss*buy_price:
        sell_price = stop_loss*buy_price

    add = buy_price*commision/100
    minus = sell_price*commision/100

    
    return (sell_price-minus)/(buy_price+add)
