import pandas as pd
import ast
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# --- KONFIGURACJA ---
ROOT = Path(__file__).resolve().parents[1]
FILE_1 = 'results.csv'  # Nazwa pierwszego pliku
FILE_2 = 'results_etf.csv'  # TUTAJ WPISZ NAZWĘ DRUGIEGO PLIKU


# --- FUNKCJA ŁADUJĄCA DANE ---
def load_and_process(filepath):
    """Wczytuje CSV, parsuje daty i kolumnę weights (str -> dict -> kolumny)."""
    full_path = f'{ROOT}/{filepath}'

    # Sprawdzenie czy plik istnieje, żeby nie wywaliło błędu
    if not Path(full_path).exists():
        print(f"Błąd: Plik {full_path} nie istnieje.")
        return None, None

    df = pd.read_csv(full_path)
    df['date'] = pd.to_datetime(df['date'])

    # Parsowanie wag (string -> dict)
    df['weights_dict'] = df['weights'].apply(lambda x: ast.literal_eval(x))

    # Rozbicie słownika na kolumny - tu mamy czyste wagi (np. 0.15), a nie kwoty
    weights_df = pd.json_normalize(df['weights_dict'])

    # Łączymy datę, nav, cash i czyste wagi
    final_df = pd.concat([df[['date', 'nav', 'cash']], weights_df], axis=1)

    return final_df, weights_df.columns.tolist()


# --- 1. WCZYTANIE OBU PORTFELI ---
df1, tickers1 = load_and_process(FILE_1)
df2, tickers2 = load_and_process(FILE_2)

# Jeśli któregoś pliku brakuje, przerywamy
if df1 is None or df2 is None:
    exit()

# --- 2. TWORZENIE WYKRESU Z DWIEMAOSIAMI (Dual Axis) ---
# Tworzymy wykres z dodatkową osią Y (secondary_y=True)
# Lewa oś (primary): NAV, Cash (wartości w PLN)
# Prawa oś (secondary): Wagi (wartości 0.0 - 1.0)
fig = make_subplots(specs=[[{"secondary_y": True}]])


# Funkcja pomocnicza do dodawania śladów (trace)
def add_portfolio_traces(df, tickers, label_suffix, line_style='solid'):
    # 1. NAV (Lewa oś)
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['nav'],
        mode='lines',
        name=f'NAV ({label_suffix})',
        line=dict(width=3, dash=line_style)
    ), secondary_y=False)

    # 2. Cash (Lewa oś)
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['cash'],
        mode='lines',
        name=f'Cash ({label_suffix})',
        line=dict(width=1, dash=line_style),
        visible='legendonly'  # Domyślnie ukryty, żeby nie zaśmiecać
    ), secondary_y=False)

    # 3. Wagi Tickerów (Prawa oś)
    for ticker in tickers:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df[ticker],
            mode='lines',
            name=f'{ticker} waga ({label_suffix})',
            line=dict(width=1, dash=line_style),
            # visible='legendonly' # Odkomentuj, jeśli chcesz domyślnie ukryć tickery
        ), secondary_y=True)


# Dodajemy dane do wykresu
# Portfel 1: Linia ciągła (solid)
add_portfolio_traces(df1, tickers1, "Portfel 1", line_style='solid')

# Portfel 2: Linia przerywana (dash) - dla odróżnienia
add_portfolio_traces(df2, tickers2, "Portfel 2", line_style='dash')

# --- 3. KONFIGURACJA WYGLĄDU ---
fig.update_layout(
    title="Porównanie Portfeli: NAV (Lewa Oś) vs Wagi Aktywów (Prawa Oś)",
    hovermode="x unified",
    template="plotly_white",
    legend=dict(groupclick="toggleitem")  # Ułatwia klikanie w legendę
)

# Opis osi
fig.update_yaxes(title_text="Wartość (PLN)", secondary_y=False)
fig.update_yaxes(title_text="Waga w portfelu (0-1)", secondary_y=True, tickformat=".1%")

fig.show()