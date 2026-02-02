from DataModels import AssetMetadata
import pandas as pd
from datetime import date, datetime


class TimeSeries:
    def __init__(self, metadata: AssetMetadata, csv_path: str):
        self.metadata = metadata
        self._data: pd.DataFrame = pd.DataFrame()
        self._returns: pd.Series = pd.Series(dtype=float)
        self.load_data_from_csv(csv_path)

    def load_data_from_csv(self, path: str):
        """
        Wczytanie szeregu z CSV zgodnie z podanym formatem:
        Data,Otwarcie,Najwyzszy,Najnizszy,Zamkniecie,Wolumen
        """
        try:
            df = pd.read_table(path, sep=',', parse_dates=['<DATE>'])

            # Standaryzacja nazw kolumn (aby reszta systemu była niezależna od nazw w CSV)
            df = df.rename(columns={
                '<DATE>': 'date',
                '<OPEN>': 'open',
                '<HIGH>': 'high',
                '<LOW>': 'low',
                '<CLOSE>': 'close',
                '<VOL>': 'volume'
            })

            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)

            # Wstępne czyszczenie danych
            df = df.ffill().dropna()

            self._data = df
            # Obliczenie stóp zwrotu od razu przy inicjalizacji (prosta analiza ilościowa)
            self._returns = self._data['close'].pct_change().dropna()

        except Exception as e:
            print(f"Błąd podczas wczytywania danych dla {self.metadata.ticker}: {e}")

    def get_price_on_date(self, query_date: date, price_type: str = 'close') -> float:
        """Dostęp do ceny na zadany dzień. Obsługuje brak danych (bierze ostatnią dostępną)."""
        ts_date = pd.Timestamp(query_date)

        # Jeśli Data jest w indeksie, zwróć cenę
        if ts_date in self._data.index:
            return self._data.loc[ts_date, price_type]

        # Jeśli nie (np. święto), weź ostatnią dostępną przed tą datą (asof)
        # To zapobiega "rozwlaniu się" kodu przy dziurach w danych
        idx = self._data.index.get_indexer([ts_date], method='pad')[0]
        if idx == -1:
            raise ValueError(f"Brak danych historycznych dla {self.metadata.ticker} przed {query_date}")

        real_date = self._data.index[idx]
        return self._data.loc[real_date, price_type]

    def get_returns_series(self, start: date, end: date) -> pd.Series:
        """Zwraca szereg stóp zwrotu w zadanym oknie czasowym (do analizy korelacji)."""
        mask = (self._returns.index >= pd.Timestamp(start)) & (self._returns.index <= pd.Timestamp(end))
        return self._returns.loc[mask]

    def normalize(self, start_value: float = 100.0) -> pd.Series:
        """Metoda normalizacji szeregu (np. do wizualizacji)."""
        return (self._data['close'] / self._data['close'].iloc[0]) * start_value

#
