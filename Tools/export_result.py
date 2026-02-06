from ClassETF.PortfolioTailCoRETF import PortfolioTailCoRETF
from ClassETF.Portfolio import Portfolio
from DataModels.MarketConfig import MarketConfig
from ClassETF.Market import Market
from Data.WIG20 import WIG20
from datetime import date
import json
import os
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Funkcja pomocnicza do zapisu pod chmurę
def export_result(portfolio_obj, filename, run_name, params, market_obj):
    # 1. Pobierz historię jako DataFrame
    df = portfolio_obj.get_history()

    # 2. Utwórz folder storage jeśli nie istnieje
    os.makedirs('storage', exist_ok=True)

    # 3. Zapisz CSV
    csv_path = f"{ROOT}/storage/{filename}.csv"
    df.to_csv(csv_path)

    # 4. Wyciągnij mapę sektorów z obiektu Market dla wszystkich dostępnych aktywów
    # To kluczowe dla wykresów kołowych w aplikacji
    sector_map = {}
    asset_names = {}

    # Przeiteruj przez wszystkie aktywa w rynku, żeby zbudować słownik
    # Zakładam, że market.assets to słownik lub lista tickerów.
    # Jeśli market trzyma aktywa w market.assets słowniku:
    try:
        # Dostosuj tę pętlę do struktury swojego obiektu Market
        # Zakładam, że masz dostęp do listy tickerów np. przez keys()
        for ticker in market_obj.assets.keys():
            meta = market_obj.get_asset(ticker).metadata
            sector_map[ticker] = meta.market_sector
            asset_names[ticker] = meta.name
    except Exception as e:
        print(f"Warning: Could not extract metadata: {e}")

    # 5. Przygotuj Metadane
    metadata = {
        "id": filename,
        "name": run_name,
        "params": params,
        "generated_at": str(date.today()),
        "sector_map": sector_map,
        "asset_names": asset_names
    }

    # 6. Zapisz JSON
    json_path = f"{ROOT}/storage/{filename}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"[OK] Zapisano: {filename} (.csv + .json)")


if __name__ == "__main__":
    # 1. Konfiguracja

    conf = MarketConfig(start_date=date(2018, 1, 1), end_date=date(2022, 12, 31))
    market = Market(conf)

    directories = [f"{ROOT}/Data/wse stocks"]
    meta_path = f"{ROOT}/Data/poland.csv"
    market.load_assets_from_directories(directories, meta_path)

    WIG_wagi = WIG20()
    data_start = date(2019, 1, 1)

    # --- SCENARIUSZ 1: TailCoR ETF ---
    risk_param = 1.06
    print("Liczenie TailCoR...")
    tailcor_pf = PortfolioTailCoRETF(market, initial_capital=100000, data_start=data_start, risk_threshold=risk_param)
    tailcor_pf.rebalance_portfolio(WIG_wagi.get_index_weights(data_start))
    tailcor_pf.run_backtest()

    # Eksport TailCoR
    export_result(
        portfolio_obj=tailcor_pf,
        filename="tailcor_run_long3",
        run_name="TailCoR WIG20 (Safe)",
        params={"type": "TailCoR", "risk_threshold": risk_param, "start_date": str(data_start)},
        market_obj=market
    )

    # --- SCENARIUSZ 2: Benchmark (Zwykły ETF) ---
    print("Liczenie Benchmark...")
    benchmark_pf = Portfolio(market, initial_capital=100000, data_start=data_start)
    benchmark_pf.rebalance_portfolio(WIG_wagi.get_index_weights(data_start))
    benchmark_pf.run_backtest()

    # Eksport Benchmarku
    export_result(
        portfolio_obj=benchmark_pf,
        filename="benchmark_run_long3",
        run_name="Standard WIG20 ETF",
        params={"type": "Benchmark", "start_date": str(data_start)},
        market_obj=market
    )