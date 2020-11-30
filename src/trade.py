def wheter_to_buy(data,predictions,commission=0.3):
    how_may_votes = len(predictions)
    to_buy =0 
    open_pirce = data[0] 
    low_price = data[2]
    for i in range(how_may_votes):
        if (predictions[i]*(1-commission/100)-(open_pirce)*(1+commission/100)) >.01:
            to_buy+=1
    if to_buy > how_may_votes//2 :
        return buy(data,commision=commission)
    return 1.0


def buy(data,commision=0.3,min_commision=3,stop_loss=0.9):
    open_price = data[0] 
    high_price = data[1]
    low_price = data[2]
    close_price = data[3]
    final_price = close_price

    if high_price > close_price:
        final_price = (high_price+close_price)/2 # slight chance to sell at highest 
    elif close_price <= stop_loss*open_price:
        final_price = stop_loss*open_price
    

    add = open_price*commision/100
    minus = final_price*commision/100

    
    return (final_price-minus)/(open_price+add)
