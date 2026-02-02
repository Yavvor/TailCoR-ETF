from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union
from datetime import date, datetime
import pandas as pd
import numpy as np

# Zakładam, że te importy działają w Twoim środowisku
from ClassETF.Market import Market
from Data.WIG20 import WIG20


class Portfolio(ABC):
    def __init__(self, market: Market, initial_capital: float = 10000.0, data_start: date = '1900-12-31'):
        self.market = market
        self.initial_capital = initial_capital

        # Stan portfela
        self.cash = initial_capital
        self.holdings: Dict[str, float] = {}  # ticker -> liczba jednostek
        self.weights: Dict[str, float] = {}  # ticker -> waga docelowa (0-1)

        # Historia NAV (Net Asset Value)
        self.nav_history: List[Dict] = []

        if data_start == '1900-12-31':
            self.data_start = self.market.config.start_date
        else:
            self.data_start = data_start

        self.current_date: date = self.data_start
        self.WIG20 = WIG20()

    def calculate_current_nav(self) -> float:
        """Wylicza bieżącą wartość portfela (gotówka + wycena aktywów)."""
        assets_value = 0.0
        for ticker, units in self.holdings.items():
            price = self.market.get_asset(ticker).get_price_on_date(self.current_date)
            assets_value += units * price

        # Upewniamy się, że zwracamy float, a nie numpy scalar
        return float(self.cash + assets_value)

    def record_history(self):
        """Zapisuje bieżący stan do historii."""
        nav = self.calculate_current_nav()

        # Kopiujemy wagi, aby nie zapisywać referencji
        self.nav_history.append({
            'date': self.current_date,
            'nav': nav,
            'cash': float(self.cash),  # Dla pewności rzutujemy też gotówkę
            'weights': self.weights.copy()
        })

    def rebalance_portfolio(self, target_weights: Dict[str, float]):
        """
        Wersja z filtracją dostępności aktywów PRZED normalizacją.
        Zapobiega pozostawianiu niealokowanej gotówki, gdy brakuje danych dla składnika.
        """
        total_nav = self.calculate_current_nav()

        # 1. FILTRACJA: Sprawdzamy dostępność w Market
        # Odrzucamy instrumenty, których fizycznie nie mamy w danych (Market)
        valid_target_weights = {}
        for ticker, weight in target_weights.items():
            if ticker in self.market.assets:
                valid_target_weights[ticker] = weight
            else:
                # Opcjonalnie: log informacyjny
                # print(f"Info: Ticker {ticker} niedostępny w Market. Pomijam i redystrybuuję kapitał.")
                pass

        # 2. NORMALIZACJA WAG (tylko dla zweryfikowanych aktywów)
        total_weight_sum = sum(valid_target_weights.values())

        # Zabezpieczenie: jeśli po filtracji nic nie zostało, wychodzimy (zostajemy w gotówce/starym portfelu)
        if total_weight_sum <= 0:
            print(f"Warning: Brak dostępnych aktywów do rebalansu w dniu {self.current_date}.")
            return

        # Przeliczamy wagi tak, by sumowały się do 1.0 (100% alokacji w to co dostępne)
        # Rzutujemy na float(), by uniknąć typów numpy
        normalized_weights = {
            t: float(w / total_weight_sum)
            for t, w in valid_target_weights.items()
        }

        # Zapisujemy docelowe wagi do historii (to są wagi, które faktycznie próbujemy odwzorować)
        self.weights = normalized_weights

        # 3. TWORZYMY NOWY PORTFEL
        new_holdings = {}
        total_allocated_value = 0.0

        for ticker, weight in normalized_weights.items():
            # Tu mamy pewność, że ticker jest w self.market.assets (dzięki filtracji wyżej)
            price = self.market.get_asset(ticker).get_price_on_date(self.current_date)

            # Dodatkowe zabezpieczenie: jeśli cena jest błędna (<=0),
            # to mimo obecności w Market nie możemy kupić.
            if price <= 0:
                print(f"Warning: Cena {ticker} wynosi {price}. Pomijam zakup.")
                continue

            # Obliczamy wartość pozycji
            target_value = total_nav * weight
            target_units = target_value / price

            new_holdings[ticker] = float(target_units)

            # Dodajemy do sumy faktycznie alokowanych środków
            total_allocated_value += target_value

        # 4. Podmieniamy portfel na nowy
        self.holdings = new_holdings

        # 5. OBLICZAMY GOTÓWKĘ JAKO RESZTĘ
        # Powinna być minimalna (wynikająca z zaokrągleń), chyba że cena była <= 0
        self.cash = float(total_nav - total_allocated_value)

        # print(f'Rebalans {self.current_date}')

    def run_backtest(self):
        """Pętla symulacyjna - przechodzi dzień po dniu."""
        dates = pd.date_range(self.data_start, self.market.config.end_date, freq='B')

        for d in dates:
            self.current_date = d.date()

            # 1. Wywołanie logiki strategii (decyzja o zmianie wag)
            self.on_market_step()

            # 2. Zapisz wynik
            self.record_history()

    def on_market_step(self):
        """
        Sprawdzamy czy WIG20 się zaktualizował
        """
        # Pobieramy wagi z indeksu
        current_index_weights = self.WIG20.get_index_weights(self.current_date)

        # Opcjonalnie: Jeśli porównanie self.weights == current_index_weights
        # zawodzi przez typy (float vs np.float64) lub precyzję,
        # logika poniżej i tak wymusi rebalans przy najmniejszej zmianie.

        if self.weights == current_index_weights:
            pass
        else:
            self.rebalance_portfolio(current_index_weights)

    def get_nav_series(self) -> pd.Series:
        """Zwraca historię NAV jako pandas Series (do wizualizacji)."""
        df = pd.DataFrame(self.nav_history)
        df.set_index('date', inplace=True)
        return df['nav']

    def get_history(self) -> pd.DataFrame:
        df = pd.DataFrame(self.nav_history)
        return df