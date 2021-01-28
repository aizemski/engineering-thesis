# Wykorzystanie modeli statystycznych oraz metod uczenia maszynowego w tradingu na giełdzie papierów wartościowych.

Celem niniejszej projetku jest zbadanie: Czy przy wykorzystaniu wybranych modeli statystycznych oraz metod uczenia maszynowego w celu predykcji, można stworzyć efektywny system tradingowy pozwalający na zyski większe niż indeks?

Wykorzystane biblioteki wraz z ich wersjami:

-   Keras 2.4.3
-   Keras-Preprocessing 1.1.2
-   matplotlib 3.3.3
-   numpy 1.18.5
-   pandas 1.1.4
-   requests 2.23.0
-   scikit-learn 0.23.2-
-   scipy 1.5.4
-   statsmodels 0.12.1
-   tensorflow 2.3.1

Struktura projektu:

-   src:

    -   armima.py - funkcje odpowiedzialne za ewaluacje handlu za pomoca modelu arima (wraz z tworzeniem przewidywan).
    -   data.py - wstępna obróbka danych.
    -   get_stock_data.py - pobieranie notowan i zapisywanie do plików.
    -   lasso.py - funkcje odpowiedzialne za ewaluacje handlu za pomoca modelu lasso (wraz z tworzeniem przewidywan).
    -   lstm.py - funkcje odpowiedzialne za ewaluacje handlu za pomoca modelu lstm (wraz z tworzeniem przewidywan oraz uczeniem modelu).
    -   main.py - zarzadzanie aplikacja.
    -   tickers.py - spółki wchodzące w skład wig20 oraz przykladowe portfele.
    -   trade.py - algorytm podejmujacy decyzje o tradingu.
    -   var.py - funkcje odpowiedzialne za ewaluacje handlu za pomoca modelu var (wraz z tworzeniem przewidywan).

-   data:
    -   models - modele LSTM, w folderze są zapisane modele wykorzystane w pracy inzynierskiej,
    -   plots - wykresy wygenerowane przez aplikacje, w folderze znajdują się dane wykresy w pracy inzynierskiej,
    -   stock - dane odnosie notowan spółek, w folderze znajdują się dane wykorzystane w pracy inzynierskiej,

W celu uruchomieniu aplikacji nalezy przejsc do folderu src/ i uruchmić main.py.
Po uruchomieniu zostaniemy poproszeni o wybranie:

-   czy chcemy pobrac aktualne dane z rynku,
-   czy chcemy wygenerowac modele LSTM,
-   czy chchcemy zapisywac wyniki poszczegolnych akcji,
-   jaką prowizję płacimy za zlecenie.

W celu powtórzenia eksperymentu z pracy inzynierskiej nalezy w dwóch pierwszych pytaniach należy wybrać opcje N, w trzecim T a w czwartym prowizje na poziomie 0%.
