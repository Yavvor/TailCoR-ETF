# TailCoR-ETF

TailCoR-ETF is a Python project for working with ETF data and performing tail-risk analysis for Exchange-Traded Funds (ETFs). The repository contains a main script designed to process ETF return data with a particular focus on downside risk and tail behavior.

More about TailCor:

> Chiapino, F., Ortobelli Lozza, S., Rachev, S. T., & Sframe, S. (2022).  
> *TailCoR: A dependence measure for financial contagion*. PLOS ONE, 17(12), e0278599.  
> [https://doi.org/10.1371/journal.pone.0278599](https://doi.org/10.1371/journal.pone.0278599)

## Features

- Python-based ETF data analysis
- Tail-risk and extreme-return metrics
- Simple, script-driven workflow
- Easily extendable for further financial research

## Installation

Install the required dependencies using:

pip install -r requirements.txt

## Usage

1. Set `folder_path` to the directory containing the input data.
2. Set `save_path` to the directory where results should be stored.
3. Optionally, set `WINDOW_SIZE` and `STEP` manually.  
   If left as `None`, the script will calculate them automatically.
4. Run the script:

python main.py

## Notes

This project was developed as part of the *Methods of Statistical Arbitrage* coursework.
