import json
import pandas as pd
import ast
from datetime import date
from pathlib import Path


class ResultManager:
    def save_result(self, df, filename, metadata):
        """
        Zapisuje DataFrame do CSV i metadane do JSON.
        metadata: dict z parametrami (np. risk_threshold, typ portfela)
        """
        # Zapis CSV
        df.to_csv(f"storage/{filename}.csv")

        # Konwersja dat w metadanych na string (JSON nie lubi obiektów date)
        meta_ready = {k: (v.isoformat() if isinstance(v, (date)) else v) for k, v in metadata.items()}

        # Zapis JSON
        with open(f"storage/{filename}.json", 'w') as f:
            json.dump(meta_ready, f)

    def load_result(self, filename):
        """Ładuje dane i parsuje kolumnę weights"""
        df = pd.read_csv(f"storage/{filename}.csv", index_col=0)

        # Parsowanie stringa "{'PKO': 0.1...}" na prawdziwy słownik
        # Używamy ast.literal_eval, bo to bezpieczniejsze niż eval()
        df['weights'] = df['weights'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        with open(f"storage/{filename}.json", 'r') as f:
            meta = json.load(f)

        return df, meta