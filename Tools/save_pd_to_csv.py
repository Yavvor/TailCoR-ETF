import pandas as pd
import os

def save_pd_to_csv(df, nazwa_pliku):
    """
    Zapisuje podany DataFrame do pliku CSV.
    """
    try:
        # index=False zapobiega dopisywaniu niepotrzebnej kolumny z numeracją wierszy
        df.to_csv(nazwa_pliku, index=False, encoding='utf-8-sig')
        print(f"Sukces! Plik zapisany jako: {nazwa_pliku}")
    except Exception as e:
        print(f"Wystąpił błąd podczas zapisu: {e}")



