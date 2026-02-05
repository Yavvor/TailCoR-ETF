import numpy as np


def calculate_metrics(df):
    # Log Returns
    df['log_ret'] = np.log(df['nav'] / df['nav'].shift(1))

    # Cumulative Return
    total_return = (df['nav'].iloc[-1] / df['nav'].iloc[0]) - 1

    # Max Drawdown (MDD)
    roll_max = df['nav'].cummax()
    df['drawdown'] = (df['nav'] - roll_max) / roll_max
    max_drawdown = df['drawdown'].min()

    # Volatility (Annualized - assuming daily data 252 days)
    volatility = df['log_ret'].std() * np.sqrt(252)

    return {
        'total_return': round(total_return * 100, 2),
        'max_drawdown': round(max_drawdown * 100, 2),
        'volatility': round(volatility * 100, 2)
    }