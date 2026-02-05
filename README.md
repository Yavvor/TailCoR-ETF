# TailCoR-ETF

TailCoR-ETF is a Python project for analyzing Exchange-Traded Fund (ETF) return data with a focus on tail risk and extreme dependence between assets. The project implements workflows to process ETF returns and compute TailCoR-based metrics, which capture both linear and nonlinear dependencies in the tails of return distributions.

More about TailCor:

> Chiapino, F., Ortobelli Lozza, S., Rachev, S. T., & Sframe, S. (2022).  
> *TailCoR: A dependence measure for financial contagion*. PLOS ONE, 17(12), e0278599.  
> [https://doi.org/10.1371/journal.pone.0278599](https://doi.org/10.1371/journal.pone.0278599)


This repository was developed as part of an academic project and is intended for research and educational purposes. It can be extended for further studies in portfolio risk management and financial data analysis.

## Overview

The goal of this project is to investigate whether tail correlation measures can be used to improve portfolio risk management compared to traditional correlation-based approaches. The software enables users to:

- Load and preprocess ETF return data  
- Compute TailCoR statistics for selected assets  
- Analyze extreme co-movement behavior  
- Export results for further research or visualization  

## Installation

Clone the repository and install the required dependencies:

git clone https://github.com/Yavvor/TailCoR-ETF.git  
cd TailCoR-ETF  
pip install -r requirements.txt  

Python version 3.7 or higher is recommended.

## Usage

1. Prepare ETF return data and place it in the desired input directory.  
2. Open `main.py` and set the following variables:  
   - `folder_path` – path to the directory containing input data  
   - `save_path` – path where output results will be saved  
   - `WINDOW_SIZE` and `STEP` (optional). If set to `None`, they are calculated automatically.  
3. Run the script:

python main.py  

Results will be saved as CSV files in the specified output directory.

## Project Structure

TailCoR-ETF/  
│  
├── main.py              # Main entry point  
├── ClassETF/            # Core ETF data handling logic  
├── DataModels/          # Data structures and models  
├── Tools/               # Utility and helper functions  
├── results.csv          # Example output file  
├── results_etf.csv      # Example output file  
├── requirements.txt  
└── README.md  

## Extensions and Future Work

Possible extensions of this project include:

- Adding more tail dependence measures  
- Integrating real-time or API-based market data  
- Implementing portfolio optimization based on TailCoR  
- Developing visualization tools or dashboards  
- Comparing TailCoR with traditional correlation and covariance-based strategies  

