import os
from datetime import datetime
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Enerji Verimliliği", layout="wide")

@st.cache_data
def load_sheets():
    try:
        sheets = pd.read_excel("dc_saf_soru_tablosu.xlsx", sheet_name=None, header=0)
    except Exception as e:
        st.error(f"Excel okunurken hata oluştu: {e}")
        return {}
    cleaned = {}
    for name, df in sheets.items():
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if not df.empty:
            cleaned[name] = df
    return cleaned

def show_form():
    st.title("📥 Enerji Verimliliği Formu")
    st.markdown("""
    Bu form **dc_saf_soru_tablosu.xlsx** dosyasına göre hazırlanmıştır.  
    1. Girişi sadece **Set Değeri** alanına yapınız.  
    2. 🔴 Zorunlu (Önem: 1), 🟡 Faydalı (Önem: 2), ⚪ Opsiyonel (Önem: 3) olarak belirtilmiştir.  
    3. Detaylı bilgi ve açıklama için ℹ️ simgesine tıklayınız.
    """)

    sheets = load_sheets()
    if not sheets:
        return

    if "clicked_info" not in st.session_state:
        st.session_state.clicked_info = ""

    for sheet_idx, (sheet_name, df) in enumerate(sheets.items(), start=1):
        with st.expander(f"{sheet_idx}. {sheet_name}", expanded=(sheet_idx == 1)):

            if df.shape[1] < 4:
                st.warning("Bu sayfa için gerekli sütunlar eksik.")
                continue

            col_A, col_B, col_C, col_D = df.columns[:4]
            detail_cols = df.columns[4:]

            for idx, row in df.iterrows():
                row_key = f"{sheet_name}_{idx}"
                c1, c2, c3, c4, c5 = st.columns([2, 3, 3, 2, 1])
                with c1:
                    st.markdown(f"**{row[col_A]}**")
                with c2:
                    emoji = {"1": "🔴", "2": "🟡", "3": "⚪"}.get(str(row.get("Önem", "3")), "")
                    st.markdown(f"{emoji} {row[col_B]}")
                with c3:
                    st.markdown(str(row[col_C]) if pd.notna(row[col_C]) else "")
                with c4:
                    df.at[idx, col_D] = st.text_input(label="", value=str(row[col_D]) if pd.notna(row[col_D]) else "", key=f"val_{row_key}")
                with c5:
                    if st.button("ℹ️", key=f"info_{row_key}"):
                        st.session_state.clicked_info = row_key

                if st.session_state.clicked_info == row_key:
                    explanations = []
                    for dc_idx, col in enumerate(detail_cols):
                        val = row[col]
                        if pd.notna(val):
                            prefix = ["📘 Açıklama:", "📌 Kaynak:", "⏱ Kayıt Aralığı:"][dc_idx] if dc_idx < 3 else f"🔹 {col}:"
                            explanations.append(f"{prefix} {val}")
                    if explanations:
                        st.info("\n\n".join(explanations))

def main():
    show_form()

if __name__ == "__main__":
    main()
