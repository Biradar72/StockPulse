import streamlit as st,pandas as pd
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'results'/'StockPulse_25_Stock_Results.csv';st.title('StockPulse — BSE Direction Prediction');st.dataframe(pd.read_csv(p),use_container_width=True) if p.exists() else st.info('Run python run_25_stocks.py first.')
