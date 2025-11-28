import os
from datetime import datetime
import pandas as pd
import streamlit as st

# Sayfa ayarı
st.set_page_config(page_title="Enerji Verimliliği", layout="wide", page_icon=None, initial_sidebar_state="expanded")

@st.cache_data
def load_sheets():
    file_name = "dc_saf_soru_tablosu.xlsx"
    try:
        sheets = pd.read_excel(file_name, sheet_name=None, header=0)
    except FileNotFoundError:
        st.error("HATA: 'dc_saf_soru_tablosu.xlsx' bulunamadı. Dosyayı app.py ile aynı klasöre koyun.")
        return None
    except Exception as e:
        st.error(f"Excel okunurken hata oluştu: {e}")
        return None

    cleaned = {}
    for name, df in sheets.items():
        if df is not None:
            df = df.dropna(how="all").dropna(axis=1, how="all")
            if not df.empty:
                cleaned[name] = df
    return cleaned

def show_input_stats(sheets):
    st.sidebar.subheader("🧮 Veri Giriş Durumu")

    total_cells = 0
    filled_cells = 0
    required_cells = 0
    filled_required = 0
    missing_required_entries = []

    for sheet_name, df in sheets.items():
        if df is None or df.empty or df.shape[1] < 4:
            continue
        col_D = df.columns[3]
        if "Önem" not in df.columns:
            continue
        for idx, row in df.iterrows():
            val = row[col_D]
            importance = str(row["Önem"]).strip()
            total_cells += 1
            if pd.notna(val) and str(val).strip() != "":
                filled_cells += 1
                if importance == "1":
                    filled_required += 1
            elif importance == "1":
                missing_required_entries.append((sheet_name, row[0], row[1]))
                required_cells += 1

    overall_pct = int(100 * filled_cells / total_cells) if total_cells else 0
    required_pct = int(100 * filled_required / required_cells) if required_cells else 0

    st.sidebar.metric("Toplam Giriş Oranı", f"{overall_pct}%")
    st.sidebar.progress(overall_pct / 100)

    st.sidebar.metric("Zorunlu Veri Girişi", f"{required_pct}%")
    st.sidebar.progress(required_pct / 100)

    if m
