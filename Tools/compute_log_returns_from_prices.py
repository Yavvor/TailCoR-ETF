import numpy as np
import pandas as pd
def compute_log_returns_from_prices(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Pomocnicza wersja Twojej funkcji dostosowana do macierzy cen (szerokiej).
    Oblicza logarytmiczne stopy zwrotu: ln(P_t / P_{t-1}).
    """
    # Logarytm z cen
    df_log = np.log(df_prices)
    # Różnica (diff) daje log returns
    df_log_returns = df_log.diff() #.dropna()
    return df_log_returns.iloc[1:]