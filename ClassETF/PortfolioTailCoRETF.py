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
        # Filtrowanie rezerwy do aktywnych na starcie (tu był błąd w oryginale - nadpisywanie listy, poprawiłem logikę)
        loaded_list = [t for t in self._substitute_list if t in market.get_active_tickers(self.data_start)]
        # Jeśli lista z pliku pusta lub nieaktualna, bierzemy cały rynek jako rezerwę (bezpiecznik)
        if not loaded_list:
            self._substitute_list = self.market.get_active_tickers(self.data_start)
        else:
            self._substitute_list = loaded_list

        self.WIG20 = WIG20()
        self.WIG_refactor_date = self.WIG20.get_last_update_date(self.current_date)

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
        3. Znajdź najlepsze zastępstwo z rezerwy (Low TailCoR).
        4. NOWOŚĆ: Nowe aktywa przejmują wagi starych (sloty),
           gdzie najbezpieczniejszy kandydat dostaje największą dostępną wagę.
        """
        self._days_since_rebalance += 1

        # Pobierz aktualny skład portfela
        current_portfolio_tickers = list(self.holdings.keys())

        ####  Jeżeli WIG się zaktualizował, to wymuszamy zmianę wag do niego.
        WIG20_date = self.WIG20.get_last_update_date(self.current_date)

        if WIG20_date != self.WIG_refactor_date:
            self._substitute_list = self.market.get_active_tickers(self.data_start)
            self.WIG_refactor_date = WIG20_date
            self._days_since_rebalance = 0
            self.rebalance_portfolio(self.WIG20.get_index_weights(self.current_date))
            return

        # --- 0. BOOTSTRAPPING (Start symulacji jeśli portfel pusty) ---
        if not current_portfolio_tickers:
            available_starters = [t for t in self._substitute_list
                                  if t in self.market.get_active_tickers(self.current_date)]

            if available_starters:
                initial_selection = available_starters[:self.max_portfolio_size]
                for t in initial_selection:
                    if t in self._substitute_list:
                        self._substitute_list.remove(t)

                # Na start równe wagi lub wg ryzyka - tu zostawiam inverse risk jako start,
                # ale można zmienić na równe wagi: {t: 1.0/len(initial_selection) ...}
                initial_scores = self._calculate_risk_scores(initial_selection)
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
        risky_assets = portfolio_scores[portfolio_scores > self.risk_threshold].index.tolist()

        if not risky_assets:
            return

        # --- 2. LOGIKA WYMIANY (SWAP) ---

        # a) Pobieramy obecne wagi, żeby wiedzieć jakie "sloty" się zwalniają
        current_weights = self.get_current_weights()

        # Lista wag, które zostaną zwolnione (do recyklingu)
        # Sortujemy je malejąco, żeby największą wagę oddać najlepszemu kandydatowi
        freed_weights_values = [current_weights.get(t, 0.0) for t in risky_assets]
        freed_weights_values.sort(reverse=True)

        # Lista aktywów, które zostają ("Keepers")
        keep_tickers = [t for t in current_portfolio_tickers if t not in risky_assets]

        # b) Znajdź kandydatów (tylko aktywni na rynku)
        active_substitutes = [t for t in self._substitute_list
                              if t in self.market.get_active_tickers(self.current_date)]

        if not active_substitutes:
            # Brak kandydatów - rebalansujemy to co zostało
            if keep_tickers:
                scores_keep = portfolio_scores.loc[keep_tickers]
                # Tu musimy użyć inverse risk lub prostej normalizacji, bo nie mamy nowych
                new_weights = self._calculate_inverse_risk_weights(scores_keep, 1.0)
                self.rebalance_portfolio(new_weights)

                for ra in risky_assets:
                    if ra not in self._substitute_list:
                        self._substitute_list.append(ra)
            return

        # c) Wybór najlepszych zastępców
        # Analizujemy grupę: [Aktywa które zostają] + [Wszyscy kandydaci]
        analysis_pool = keep_tickers + active_substitutes
        full_analysis_scores = self._calculate_risk_scores(analysis_pool)

        if full_analysis_scores.empty:
            return

        # Wyciągamy wyniki tylko dla kandydatów z rezerwy
        # Sortujemy ROSNĄCO (TailCoR: im mniej tym lepiej) -> Priorytet wejścia
        candidates_scores = full_analysis_scores.loc[full_analysis_scores.index.isin(active_substitutes)].sort_values()

        # Potrzebujemy tylu nowych, ile wyrzuciliśmy (lub mniej, jeśli brakuje kandydatów)
        num_needed = len(risky_assets)
        best_candidates = candidates_scores.index[:num_needed].tolist()

        # --- 3. DYSTRYBUCJA WAG (NOWA LOGIKA) ---

        final_weights = {}

        # 1. Przepisujemy wagi aktywów, które zostały (bez zmian)
        for ticker in keep_tickers:
            final_weights[ticker] = current_weights.get(ticker, 0.0)

        # 2. Przypisujemy zwolnione wagi nowym aktywom
        # best_candidates są posortowani od najlepszego (najmniejsze ryzyko)
        # freed_weights_values są posortowane od największej (największy udział)
        # Robimy mapowanie: Najlepszy Kandydat -> Największa Waga

        for i, candidate in enumerate(best_candidates):
            # Zabezpieczenie, gdyby z jakiegoś powodu długości list się nie zgadzały
            if i < len(freed_weights_values):
                assigned_weight = freed_weights_values[i]
                final_weights[candidate] = assigned_weight
            else:
                # Jeśli mamy więcej kandydatów niż zwolnionych wag (teoretycznie niemożliwe w tej logice),
                # to nadmiarowi nie wchodzą.
                pass

        # Jeśli kandydatów było mniej niż wyrzuconych aktywów (rzadki przypadek),
        # część 'freed_weights_values' zostanie niewykorzystana (będzie w gotówce)
        # lub można by ją doliczyć do reszty. W tej implementacji - zostaje w gotówce do kolejnego kroku
        # lub do najbliższej aktualizacji WIG20.

        # --- 4. AKTUALIZACJA REZERWY I WYKONANIE ---

        # Aktualizacja listy rezerwowej (Swap)
        for bad_asset in risky_assets:
            if bad_asset not in self._substitute_list:
                self._substitute_list.append(bad_asset)

        # Usuwamy z rezerwy TYLKO tych kandydatów, którzy faktycznie weszli do final_weights
        added_assets = [t for t in final_weights.keys() if t not in keep_tickers]
        for good_asset in added_assets:
            if good_asset in self._substitute_list:
                self._substitute_list.remove(good_asset)

        # Wykonaj rebalans
        self.rebalance_portfolio(final_weights)
        self._days_since_rebalance = 0

    def _calculate_inverse_risk_weights(self, scores: pd.Series, total_weight_to_distribute: float) -> Dict[str, float]:
        """
        Pozostawiona jako pomocnicza (np. do startu), ale nie używana w głównym cyklu rebalansu.
        """
        if scores.empty or total_weight_to_distribute <= 0:
            return {}

        inv_risk = 1.0 / (abs(scores) + 1e-6)
        total_inv_risk = inv_risk.sum()

        if total_inv_risk == 0:
            return {}

        weights = (inv_risk / total_inv_risk) * total_weight_to_distribute
        return weights.to_dict()

    def _calculate_risk_scores(self, tickers: List[str]) -> pd.Series:
        # Bez zmian
        if len(tickers) < 2:
            return pd.Series(index=tickers, data=0.0)

        analysis_start = self.current_date - pd.Timedelta(days=self.lookback_period * 2)
        prices_df = self._get_prices_matrix(analysis_start, self.current_date, tickers)

        if len(prices_df) < 50:
            return pd.Series()

        log_returns = compute_log_returns_from_prices(prices_df)

        try:
            tailcor_matrix = tailCoR(log_returns, zeta=self.zeta, tau=self.tau, mode="tailcor")
        except Exception:
            return pd.Series()

        matrix_filled = tailcor_matrix.fillna(tailcor_matrix.mean().mean())
        n_assets = len(matrix_filled.columns)
        if n_assets <= 1:
            return pd.Series(index=tickers, data=0.0)

        scores = (matrix_filled.sum(axis=1) - 1.0) / (n_assets - 1)
        return scores

    def _get_prices_matrix(self, start, end, tickers: List[str]) -> pd.DataFrame:
        # Bez zmian
        data = {}
        for ticker in tickers:
            if ticker not in self.market.assets:
                continue
            asset = self.market.get_asset(ticker)
            if asset._data.index[-1] < pd.Timestamp(start):
                continue

            mask = (asset._data.index >= pd.Timestamp(start)) & (asset._data.index < pd.Timestamp(end))
            series = asset._data.loc[mask, 'close']

            if not series.empty:
                data[ticker] = series

        df = pd.DataFrame(data)
        df.ffill(inplace=True)
        df.dropna(how='any', inplace=True)
        if len(df) > self.lookback_period:
            df = df.iloc[-self.lookback_period:]

        return df

    def get_current_weights(self) -> Dict[str, float]:
        if not self.holdings:
            return {}

        values = {}
        total_value = 0.0

        for ticker, qty in self.holdings.items():
            price = self.market.get_asset(ticker).get_price_on_date(self.current_date)
            val = qty * price
            values[ticker] = val
            total_value += val

        if total_value == 0:
            return {}

        weights = {t: v / total_value for t, v in values.items()}
        return weights