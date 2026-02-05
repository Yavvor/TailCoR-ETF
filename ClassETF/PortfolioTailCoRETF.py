import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from datetime import date

from ClassETF.Portfolio import Portfolio
from ClassETF.Market import Market
from Data.WIG20 import WIG20
from Tools.compute_log_returns_from_prices import compute_log_returns_from_prices
from Tools.tailCoR import tailCoR


class PortfolioTailCoRETF(Portfolio):
    def __init__(self, market: Market, initial_capital: float,
                 rebalance_window: int = 20,
                 lookback_period: int = 252,
                 zeta: float = 0.75,
                 tau: float = 0.95,
                 risk_threshold: float = 1.1,
                 max_portfolio_size: int = 20,
                 data_start: date = '1900-12-31'):

        super().__init__(market, initial_capital, data_start=data_start)

        self.rebalance_window = rebalance_window
        self.lookback_period = lookback_period
        self.zeta = zeta
        self.tau = tau
        self.risk_threshold = risk_threshold
        self.max_portfolio_size = max_portfolio_size

        self._days_since_rebalance = 0


        if data_start == '1900-12-31':
            self.data_start = self.market.config.start_date
        else:
            self.data_start = data_start
        self.current_date: date = self.data_start

        # Inicjalizacja listy rezerwowej
        self._substitute_list: List[str] = self._load_substitute_list("Data/SubstituteList.txt")
        # Filtrowanie rezerwy do aktywnych na starcie
        self._substitute_list = [t for t in self._substitute_list if t in market.get_active_tickers(self.current_date)]
        self._substitute_list = self.market.get_active_tickers(self.data_start)

        self.WIG20=WIG20()
        self.WIG_refactor_date=self.WIG20.get_last_update_date(self.current_date)

    def _load_substitute_list(self, filepath: str) -> List[str]:
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} not found. Substitute list is empty.")
            return []
        with open(filepath, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        return tickers

    def on_market_step(self):
        """
        Logika decyzyjna:
        1. Sprawdź ryzyko obecnego portfela.
        2. Wyrzuć aktywa przekraczające próg (High TailCoR).
        3. Znajdź najlepsze zastępstwo z rezerwy (Low TailCoR w kontekście reszty portfela).
        4. Rozdziel wagę wyrzuconych aktywów na nowe, promując te bezpieczniejsze.
        """
        self._days_since_rebalance += 1

        # Pobierz aktualny skład portfela (tickery z niezerową wagą/ilością)
        # Zakładam, że self.holdings to słownik {ticker: quantity} lub podobny z klasy Portfolio
        current_portfolio_tickers = list(self.holdings.keys())

        ####  Jeżeli WIG się zaktualizował, to wymuszamy zmianę wag do niego.
        WIG20_date=self.WIG20.get_last_update_date(self.current_date)

        if WIG20_date != self.WIG_refactor_date:
            self._substitute_list = self.market.get_active_tickers(self.data_start)
            self.WIG_refactor_date=WIG20_date
            self._days_since_rebalance =0
            self.rebalance_portfolio(self.WIG20.get_index_weights(self.current_date))
            return


        # --- 0. BOOTSTRAPPING (Start symulacji) ---
        if not current_portfolio_tickers:
            # Jeśli portfel jest pusty, pobierz startowe aktywa z rezerwy
            available_starters = [t for t in self._substitute_list
                                  if t in self.market.get_active_tickers(self.current_date)]

            # Jeśli mamy dostępnych, bierzemy max_portfolio_size
            if available_starters:
                initial_selection = available_starters[:self.max_portfolio_size]

                # Usuwamy wybrane z rezerwy
                for t in initial_selection:
                    self._substitute_list.remove(t)

                # Obliczamy wagi startowe (równowagowe lub wg ryzyka startowego)
                initial_scores = self._calculate_risk_scores(initial_selection)
                # Dystrybucja kapitału wg odwrotności ryzyka
                initial_weights = self._calculate_inverse_risk_weights(initial_scores, total_weight_to_distribute=1.0)

                self.rebalance_portfolio(initial_weights)
                self._days_since_rebalance = 0
            return

        # Wykonujemy sprawdzenie tylko w oknie rebalansu
        if self._days_since_rebalance < self.rebalance_window:
            return

        # --- 1. ANALIZA RYZYKA OBECNEGO PORTFELA ---
        portfolio_scores = self._calculate_risk_scores(current_portfolio_tickers)
        if portfolio_scores.empty:
            return

        # Identyfikacja "toksycznych" aktywów
        # Zgodnie z życzeniem: wyrzucamy te, dla których TailCoR jest za duży
        risky_assets = portfolio_scores[portfolio_scores > self.risk_threshold].index.tolist()

        # Jeśli brak aktywów do wyrzucenia, kończymy (trzymamy pozycje)
        # Opcjonalnie: można tu zrobić zwykły rebalans wag, jeśli minęło dużo czasu
        if not risky_assets:
            return

            # --- 2. LOGIKA WYMIANY (SWAP) ---

        # a) Oblicz ile "wagi" (kapitału) uwalniamy
        current_weights = self.get_current_weights()  # Zakładam, że masz taką metodę w Portfolio lub trzeba ją obliczyć
        freed_weight = sum(current_weights.get(t, 0.0) for t in risky_assets)

        # Lista aktywów, które zostają ("Keepers")
        keep_tickers = [t for t in current_portfolio_tickers if t not in risky_assets]

        # b) Znajdź kandydatów (tylko aktywni na rynku)
        active_substitutes = [t for t in self._substitute_list
                              if t in self.market.get_active_tickers(self.current_date)]

        if not active_substitutes:
            # Brak kandydatów - ewentualnie tylko sprzedajemy ryzykowne (gotówka)
            # Tutaj: zostawiamy to co mamy w 'keep_tickers', reszta idzie w cash (implicit)
            # lub rebalansujemy keep_tickers do 100%.
            # Przyjmijmy rebalans pozostałych:
            if keep_tickers:
                scores_keep = portfolio_scores.loc[keep_tickers]
                new_weights = self._calculate_inverse_risk_weights(scores_keep, 1.0)
                self.rebalance_portfolio(new_weights)

                # Ryzykowne wracają do rezerwy
                for ra in risky_assets:
                    if ra not in self._substitute_list:
                        self._substitute_list.append(ra)
            return

        # c) Wybór najlepszych zastępców
        # Analizujemy grupę: [Aktywa które zostają] + [Wszyscy kandydaci]
        # Chcemy sprawdzić, jak kandydaci zachowują się względem stabilnej części portfela
        analysis_pool = keep_tickers + active_substitutes
        full_analysis_scores = self._calculate_risk_scores(analysis_pool)

        if full_analysis_scores.empty:
            return

        # Wyciągamy wyniki tylko dla kandydatów z rezerwy
        # Sortujemy rosnąco (najmniejszy score = najbezpieczniejszy w tym kontekście)
        candidates_scores = full_analysis_scores.loc[full_analysis_scores.index.isin(active_substitutes)].sort_values()

        # Ile potrzebujemy nowych? Tyle ile wyrzuciliśmy
        num_needed = len(risky_assets)
        best_candidates = candidates_scores.index[:num_needed].tolist()

        # --- 3. DYSTRYBUCJA WAG (WEIGHTING) ---
        # Zgodnie z wymaganiem:
        # "nowe assety były do siebie w takich proporcjach jak poprzednie... z korzyścią dla mniejszego ryzyka"
        # Realizacja: Bierzemy 'freed_weight' i dzielimy go między 'best_candidates' wg Inverse Risk.

        # Pobieramy score ryzyka dla wybranej elity
        selected_candidates_scores = candidates_scores.loc[best_candidates]

        # Obliczamy wagi dla nowych aktywów (sumujące się do freed_weight)
        new_assets_weights = self._calculate_inverse_risk_weights(
            selected_candidates_scores,
            total_weight_to_distribute=freed_weight
        )

        # Budujemy finalny słownik wag
        final_weights = {}

        # 1. Przepisujemy wagi aktywów, które zostały (bez zmian!)
        for ticker in keep_tickers:
            final_weights[ticker] = current_weights.get(ticker, 0.0)

        # 2. Dodajemy wagi nowych aktywów
        final_weights.update(new_assets_weights)

        # --- 4. AKTUALIZACJA REZERWY I WYKONANIE ---

        # Aktualizacja listy rezerwowej (Swap)
        for bad_asset in risky_assets:
            if bad_asset not in self._substitute_list:
                self._substitute_list.append(bad_asset)

        for good_asset in best_candidates:
            if good_asset in self._substitute_list:
                self._substitute_list.remove(good_asset)

        # Wykonaj rebalans
        self.rebalance_portfolio(final_weights)
        self._days_since_rebalance = 0

    def _calculate_inverse_risk_weights(self, scores: pd.Series, total_weight_to_distribute: float) -> Dict[str, float]:
        """
        Oblicza wagi odwrotnie proporcjonalne do ryzyka (Inverse Risk),
        skalowane tak, aby ich suma wynosiła total_weight_to_distribute.

        Im mniejszy score (ryzyko), tym większa waga.
        """
        if scores.empty or total_weight_to_distribute <= 0:
            return {}

        # 1. Inwersja ryzyka (Inverse Risk)
        # Dodajemy mały epsilon, żeby nie dzielić przez 0, choć TailCoR zwykle > 0
        # abs() na wypadek dziwnych danych, choć TailCoR > 0
        inv_risk = 1.0 / (abs(scores) + 1e-6)

        # 2. Suma inwersji
        total_inv_risk = inv_risk.sum()

        if total_inv_risk == 0:
            return {}

        # 3. Normalizacja i skalowanie do zadanej puli kapitału
        # (Score_inv / Sum_inv) * Freed_Capital
        weights = (inv_risk / total_inv_risk) * total_weight_to_distribute

        return weights.to_dict()

    def _calculate_risk_scores(self, tickers: List[str]) -> pd.Series:
        """
        Liczy Systemic Risk Score (średnia korelacja ogonowa aktywa z resztą grupy).
        Zwraca pd.Series posortowane po tickerach.
        """
        # (Ta metoda pozostaje bez większych zmian logicznych, bo liczy samą metrykę)
        if len(tickers) < 2:
            # Dla pojedynczego aktywa ryzyko systemowe jest nieokreślone/zerowe
            return pd.Series(index=tickers, data=0.0)

        # 1. Pobierz dane
        analysis_start = self.current_date - pd.Timedelta(days=self.lookback_period * 2)
        prices_df = self._get_prices_matrix(analysis_start, self.current_date, tickers)

        if len(prices_df) < 50:  # Zabezpieczenie przed brakiem danych
            return pd.Series()

        # 2. Log returns
        log_returns = compute_log_returns_from_prices(prices_df)

        # 3. TailCoR Matrix
        try:
            tailcor_matrix = tailCoR(log_returns, zeta=self.zeta, tau=self.tau, mode="tailcor")
        except Exception:
            return pd.Series()

        # Imputacja NaN (średnią z macierzy)
        matrix_filled = tailcor_matrix.fillna(tailcor_matrix.mean().mean())

        # 4. Score = średnia korelacja z innymi
        # (Suma wiersza - 1) / (N - 1) -> odejmujemy przekątną (1.0)
        n_assets = len(matrix_filled.columns)
        if n_assets <= 1:
            return pd.Series(index=tickers, data=0.0)

        scores = (matrix_filled.sum(axis=1) - 1.0) / (n_assets - 1)
        return scores

    def _get_prices_matrix(self, start, end, tickers: List[str]) -> pd.DataFrame:
        """Pobiera ceny dla podanej listy tickerów."""
        data = {}
        for ticker in tickers:
            if ticker not in self.market.assets:
                continue
            asset = self.market.get_asset(ticker)
            # Szybki check dat
            if asset._data.index[-1] < pd.Timestamp(start):
                continue

            # Pobieramy wycinek
            mask = (asset._data.index >= pd.Timestamp(start)) & (asset._data.index < pd.Timestamp(end))
            series = asset._data.loc[mask, 'close']

            if not series.empty:
                data[ticker] = series

        df = pd.DataFrame(data)
        df.ffill(inplace=True)
        df.dropna(how='any', inplace=True)
        # Ograniczamy do lookback_period
        if len(df) > self.lookback_period:
            df = df.iloc[-self.lookback_period:]

        return df

    def get_current_weights(self) -> Dict[str, float]:
        """
        Pomocnicza metoda do obliczenia aktualnych wag w portfelu
        na podstawie self.holdings i aktualnych cen.
        """
        if not self.holdings:
            return {}

        # Oblicz wartość każdej pozycji
        values = {}
        total_value = 0.0

        for ticker, qty in self.holdings.items():
            price = self.market.get_asset(ticker).get_price_on_date(self.current_date)
            val = qty * price
            values[ticker] = val
            total_value += val

        if total_value == 0:
            return {}

        # Zamień na wagi
        weights = {t: v / total_value for t, v in values.items()}
        return weights