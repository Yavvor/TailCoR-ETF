from flask import Flask, render_template, jsonify, request
import os
import glob
import json
import pandas as pd
import ast
import numpy as np

app = Flask(__name__)
STORAGE_DIR = 'storage'


def get_metadata(file_id):
    """Ładuje plik JSON z metadanymi"""
    path = os.path.join(STORAGE_DIR, f"{file_id}.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


@app.route('/')
def index():
    # Pobieranie listy dostępnych analiz
    files = []
    for f in glob.glob(os.path.join(STORAGE_DIR, '*.json')):
        meta = json.load(open(f, 'r', encoding='utf-8'))
        files.append(meta)
    return render_template('dashboard.html', files=files)


@app.route('/api/series')
def get_series():
    """Zwraca całe szeregi czasowe (NAV, Drawdown) do wykresów liniowych"""
    file_id = request.args.get('file_id')
    try:
        df = pd.read_csv(os.path.join(STORAGE_DIR, f"{file_id}.csv"))

        # Obliczenia "w locie"
        roll_max = df['nav'].cummax()
        drawdown = (df['nav'] - roll_max) / roll_max

        return jsonify({
            'dates': df['date'].tolist(),
            'nav': df['nav'].tolist(),
            'drawdown': drawdown.tolist(),
            'cash': df['cash'].tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/composition')
def get_composition():
    """Zwraca skład portfela (Pie Chart) dla konkretnej daty"""
    file_id = request.args.get('file_id')
    target_date = request.args.get('date')  # Format YYYY-MM-DD

    try:
        # 1. Ładowanie metadanych (dla mapy sektorów)
        meta = get_metadata(file_id)
        sector_map = meta.get('sector_map', {})

        # 2. Ładowanie CSV
        df = pd.read_csv(os.path.join(STORAGE_DIR, f"{file_id}.csv"))

        # 3. Szukanie wiersza po dacie
        row = df.loc[df['date'] == target_date]

        if row.empty:
            # Fallback: jeśli data nie istnieje (np. weekend), weź najbliższą poprzednią
            # Tu upraszczamy - bierzemy po prostu pusty wynik lub błąd
            return jsonify({'error': 'Date not found'}), 404

        # 4. Parsowanie wag
        weights_str = row['weights'].values[0]
        weights = ast.literal_eval(weights_str)  # dict {'PKN': 0.15, ...}

        # 5. Agregacja Sektorowa
        sector_alloc = {}
        for ticker, weight in weights.items():
            # Pomijamy zerowe pozycje
            if weight < 0.001: continue

            sector = sector_map.get(ticker, 'Inne')
            sector_alloc[sector] = sector_alloc.get(sector, 0) + weight

        return jsonify({
            'date': target_date,
            'tickers': weights,
            'sectors': sector_alloc
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)