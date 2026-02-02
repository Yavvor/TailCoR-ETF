import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import date

from DataModels.MarketConfig import MarketConfig
from DataModels.AssetMetadata import AssetMetadata

from ClassETF.TimeSeries import TimeSeries


class Market:
    def __init__(self, config: MarketConfig):
        self.config = config
        self.assets: Dict[str, TimeSeries] = {}  # Słownik ticker -> obiekt TimeSeries

    def _scan_date_range(self, file_path: Path):
        """
        Pomocnicza metoda: Wczytuje tylko kolumnę dat z pliku,
        aby szybko ustalić data_start i data_end bez ładowania całego pliku do pamięci.
        Zwraca (min_date, max_date) lub (None, None) w przypadku błędu.
        """
        try:
            # usecols=['<DATE>'] sprawia, że wczytywanie jest znacznie szybsze
            df = pd.read_table(file_path, sep=',', parse_dates=['<DATE>'], usecols=['<DATE>'])
            if df.empty:
                return None, None

            # Upewniamy się, że są posortowane
            dates = df['<DATE>'].sort_values()
            return dates.iloc[0].date(), dates.iloc[-1].date()
        except Exception:
            return None, None

    def load_assets_from_directories(self, data_dirs: List[str], metadata_csv_path: str):
        # 1. Wczytanie słownika metadanych (poland.csv)
        try:
            meta_df = pd.read_table(metadata_csv_path, sep=',')
            meta_df['ticker'] = meta_df['ticker'].astype(str).str.strip().str.upper()
            metadata_lookup = meta_df.set_index('ticker').to_dict(orient='index')
        except Exception as e:
            print(f"Błąd ładowania pliku metadanych: {e}")
            return

        loaded_count = 0

        for directory in data_dirs:
            dir_path = Path(directory)
            if not dir_path.exists():
                continue

            files = list(dir_path.glob("*.txt"))
            print(f"Skanowanie {directory}: {len(files)} plików.")

            for file_path in files:
                file_ticker = file_path.stem.strip().upper()

                # A. Pobranie informacji opisowych
                if file_ticker in metadata_lookup:
                    info = metadata_lookup[file_ticker]
                    asset_name = info.get('name', file_ticker)
                    asset_sector = info.get('sector', 'Unknown')
                else:
                    asset_name = file_ticker
                    asset_sector = 'Other'

                # B. Pobranie zakresu dat z pliku (Pre-scan)
                # Musimy to zrobić TERAZ, aby móc utworzyć AssetMetadata
                d_start, d_end = self._scan_date_range(file_path)

                if d_start is None or d_end is None:
                    print(f"Pominięto {file_ticker}: brak poprawnych dat w pliku.")
                    continue

                try:
                    # C. Tworzenie AssetMetadata (teraz mamy wszystkie wymagane pola)
                    meta = AssetMetadata(
                        ticker=file_ticker,
                        name=asset_name,
                        asset_type="equity",
                        market_sector=asset_sector,
                        data_start=d_start,  # Wstawiamy odczytane daty
                        data_end=d_end
                    )

                    # D. Tworzenie TimeSeries (przyjmuje gotowe metadane)
                    # TimeSeries wczyta sobie plik ponownie dla pełnych danych (OHLCV)
                    self.assets[file_ticker] = TimeSeries(meta, str(file_path))
                    loaded_count += 1

                except Exception as e:
                    print(f"Błąd przy tworzeniu aktywa {file_ticker}: {e}")

        print(f"Załadowano {loaded_count} aktywów.")

    def get_asset(self, ticker: str) -> TimeSeries:
        return self.assets[ticker]

    def get_assets_by_sector(self, sector: str) -> List[TimeSeries]:
        """Pomocnicza metoda: Zwraca listę aktywów z danego sektora."""
        return [asset for asset in self.assets.values() if asset.metadata.market_sector == sector]

    def get_active_tickers(self, check_date: date) -> List[str]:
        """
        WARIANT B: Sprawdza aktywność na podstawie metadanych (AssetMetadata).
        Jest to znacznie szybsze niż wchodzenie do DataFrame każdego aktywa.
        """
        active_tickers = []

        # Obsługa konwersji, gdyby check_date było pd.Timestamp
        target = check_date.date() if isinstance(check_date, pd.Timestamp) else check_date

        for ticker, asset in self.assets.items():
            # Bezpośrednie odwołanie do pól w Pydantic model
            if asset.metadata.data_start <= target <= asset.metadata.data_end:
                active_tickers.append(ticker)

        return active_tickers


    def get_market_returns_matrix(self, start: date, end: date) -> pd.DataFrame:
        """
        Zwraca DataFrame gdzie kolumny to tickery, a wiersze to stopy zwrotu.
        """
        data = {ticker: asset.get_returns_series(start, end)
                for ticker, asset in self.assets.items()}
        return pd.DataFrame(data).dropna()

if __name__=='__main__':
    from datetime import date
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]

    # 1. Konfiguracja
    conf = MarketConfig(start_date=date(2019, 1, 1), end_date=date(2023, 12, 31))
    market = Market(conf)

    # 2. Ścieżki
    directories = [f"{ROOT}/Data/wse stocks"]
    meta_path = f"{ROOT}/Data/poland.csv"

    # 3. Wczytanie wszystkiego jedną komendą
    market.load_assets_from_directories(directories, meta_path)

    # 4. Weryfikacja
    # Pobranie przykładowego aktywa i sprawdzenie czy sektor się wczytał
    if "06N" in market.assets:
        asset = market.get_asset("06N")
        print(f"Ticker: {asset.metadata.ticker}")
        print(f"Nazwa: {asset.metadata.name}")
        print(f"Sektor: {asset.metadata.market_sector}")  # Powinno wyświetlić "Miscellaneous"