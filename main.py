from ClassETF.PortfolioTailCoRETF import PortfolioTailCoRETF
from ClassETF.Portfolio import Portfolio
from DataModels.MarketConfig import MarketConfig
from ClassETF.Market import Market
from Data.WIG20 import WIG20

from datetime import date, datetime

from Tools.save_pd_to_csv import save_pd_to_csv

if __name__ == "__main__":
    # 1. Konfiguracja Rynku
    conf = MarketConfig(start_date=date(2018, 1, 1), end_date=date(2021, 1, 31))
    market = Market(conf)

    # 2. Ścieżki
    directories = ["Data/wse stocks"]
    meta_path = "Data/poland.csv"

    # 3. Wczytanie rynku
    market.load_assets_from_directories(directories, meta_path)

    WIG_wagi=WIG20()
    data_start = date(2020, 1, 1)

    # 3. Inicjalizacja Portfela TailCoR

    my_tailcor_portfolio = PortfolioTailCoRETF(market, initial_capital=100000, data_start=data_start)
    my_tailcor_portfolio.rebalance_portfolio(WIG_wagi.get_index_weights(data_start))

    #3.5 Inicjalizacja stałego portfela (quasi index)
    my_portfolio = Portfolio(market, initial_capital=100000, data_start=data_start)
    my_portfolio.rebalance_portfolio(WIG_wagi.get_index_weights(data_start))


    # 4. Uruchomienie Backtestu
    my_tailcor_portfolio.run_backtest()
    print('---------------------------------')
    my_portfolio.run_backtest()

    # 5. Dostęp do wyników
    results_etf = my_tailcor_portfolio.get_history()
    results = my_portfolio.get_history()
    save_pd_to_csv(results_etf, 'results_etf.csv')
    save_pd_to_csv(results, 'results.csv')

