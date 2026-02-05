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
    # Pobieranie listy dostępnych analiz i segregacja
    tailcor_files = []
    benchmark_files = []

    for f in glob.glob(os.path.join(STORAGE_DIR, '*.json')):
        meta = json.load(open(f, 'r', encoding='utf-8'))
        # Sprawdzamy typ (zakładamy, że Benchmark może nie mieć pola type lub mieć 'Benchmark')
        f_type = meta.get('params', {}).get('type', 'Benchmark')

        if f_type == 'TailCoR':
            tailcor_files.append(meta)
        else:
            benchmark_files.append(meta)

    return render_template('dashboard.html', tailcor=tailcor_files, benchmark=benchmark_files)


@app.route('/api/series')
def get_series():
    """Zwraca całe szeregi czasowe (NAV, Drawdown)"""
    file_id = request.args.get('file_id')
    try:
        df = pd.read_csv(os.path.join(STORAGE_DIR, f"{file_id}.csv"))

        # Konwersja daty
        df['date'] = pd.to_datetime(df['date'])

        # Obliczenia "w locie"
        roll_max = df['nav'].cummax()
        drawdown = (df['nav'] - roll_max) / roll_max

        return jsonify({
            'dates': df['date'].dt.strftime('%Y-%m-%d').tolist(),
            'nav': df['nav'].tolist(),
            'drawdown': drawdown.tolist(),
            'cash': df['cash'].tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/composition')
def get_composition():
    """Zwraca skład portfela oraz mapowanie nazw spółek"""
    file_id = request.args.get('file_id')
    target_date = request.args.get('date')

    try:
        # 1. Metadane
        meta = get_metadata(file_id)
        sector_map = meta.get('sector_map', {})
        asset_names = meta.get('asset_names', {})  # Pełne nazwy

        # 2. CSV
        df = pd.read_csv(os.path.join(STORAGE_DIR, f"{file_id}.csv"))
        row = df.loc[df['date'] == target_date]

        if row.empty:
            return jsonify({'error': 'Date not found'}), 404

        # 3. Wagi
        weights_str = row['weights'].values[0]
        weights = ast.literal_eval(weights_str)

        # 4. Agregacja Sektorowa
        sector_alloc = {}
        for ticker, weight in weights.items():
            if weight < 0.001: continue
            sector = sector_map.get(ticker, 'Inne')
            sector_alloc[sector] = sector_alloc.get(sector, 0) + weight

        return jsonify({
            'date': target_date,
            'tickers': weights,
            'sectors': sector_alloc,
            'names': asset_names
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['POST'])
def calculate_stats():
    """Oblicza statystyki dla zadanego okresu"""
    data = request.json
    id1 = data.get('id1')
    id2 = data.get('id2')
    start_date = data.get('start_date')
    end_date = data.get('end_date')

    def get_metrics(fid, s_date, e_date):
        try:
            df = pd.read_csv(os.path.join(STORAGE_DIR, f"{fid}.csv"))
            df['date'] = pd.to_datetime(df['date'])

            mask = (df['date'] >= pd.to_datetime(s_date)) & (df['date'] <= pd.to_datetime(e_date))
            period_df = df.loc[mask].copy()

            if period_df.empty:
                return None

            period_df.set_index('date', inplace=True)

            # 1. Total Return
            start_val = period_df['nav'].iloc[0]
            end_val = period_df['nav'].iloc[-1]
            total_return = (end_val - start_val) / start_val

            # 2. Max Drawdown w tym okresie
            roll_max = period_df['nav'].cummax()
            dd = (period_df['nav'] - roll_max) / roll_max
            max_dd = dd.min()

            # 3. Volatility (Annualized)
            period_df['ret'] = period_df['nav'].pct_change()
            volatility = period_df['ret'].std() * np.sqrt(252)

            # 4. Sharpe Ratio (uproszczony, risk-free=0)
            avg_ret = period_df['ret'].mean() * 252
            sharpe = avg_ret / volatility if volatility != 0 else 0

            return {
                'return': round(total_return * 100, 2),
                'max_dd': round(max_dd * 100, 2),
                'volatility': round(volatility * 100, 2),
                'sharpe': round(sharpe, 2)
            }
        except Exception:
            return None

    return jsonify({
        'p1_stats': get_metrics(id1, start_date, end_date),
        'p2_stats': get_metrics(id2, start_date, end_date)
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)