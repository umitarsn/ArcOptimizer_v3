import os
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Enerji Verimliliği",
    layout="wide",
    page_icon=None,
    initial_sidebar_state="expanded",
)

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
        if df is None:
            continue
        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")
        if not df.empty:
            cleaned[name] = df

    return cleaned

def show_energy_form():
    st.title("Enerji Verimliliği Formu")
    st.markdown("""
    Bu form **dc_saf_soru_tablosu.xlsx** dosyasına göre hazırlanmıştır.

    1. Girişi sadece **Set Değeri** alanına yapınız.
    2. 🔴 Zorunlu ("Önem: 1"), 🟡 Faydalı ("Önem: 2"), ⚪ Opsiyonel ("Önem: 3") olarak belirtilmiştir.
    3. Detaylı bilgi ve açıklama için 🔹 simgesine tıklayınız.
    """)

    sheets = load_sheets()
    if sheets is None or len(sheets) == 0:
        return

    if "user_inputs" not in st.session_state:
        st.session_state.user_inputs = {}

    for sheet_name, df in sheets.items():
        st.markdown(f"### {sheet_name}")

        headers = df.columns.tolist()
        tag_col, var_col, desc_col = headers[0], headers[1], headers[2]
        set_col = headers[3] if len(headers) > 3 else None
        detail_cols = headers[4:] if len(headers) > 4 else []

        for idx, row in df.iterrows():
            col1, col2, col3, col4, col5 = st.columns([2, 3, 4, 2, 1])

            with col1:
                st.markdown(f"**{row[tag_col]}**")

            with col2:
                importance = str(row.get("Önem", "")).strip()
                marker = ""
                if importance == "1":
                    marker = "🔴"
                elif importance == "2":
                    marker = "🟡"
                elif importance == "3":
                    marker = "⚪"
                st.markdown(f"{marker} {row[var_col]}")

            with col3:
                st.markdown(f"{row[desc_col]}")

            with col4:
                tag = row[tag_col]
                unit = str(row[set_col]) if set_col in row and pd.notna(row[set_col]) else ""
                default_value = st.session_state.user_inputs.get(tag, "")
                user_input = st.text_input("", value=default_value, key=f"input_{sheet_name}_{idx}")
                if unit:
                    user_input = user_input.strip()
                    if user_input and not user_input.endswith(unit):
                        user_input = f"{user_input} {unit}"
                st.session_state.user_inputs[tag] = user_input

            with col5:
                if detail_cols:
                    if st.button("ℹ️", key=f"info_{sheet_name}_{idx}"):
                        details = []
                        for col in detail_cols:
                            val = row.get(col, "")
                            if pd.notna(val) and str(val).strip():
                                details.append(f"- **{col}**: {val}")
                        if details:
                            st.info("\n".join(details))

    if st.button("Kaydet"):
        try:
            df_out = pd.DataFrame([st.session_state.user_inputs]).T.reset_index()
            df_out.columns = ["Tag", "Input"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = os.path.join("data", f"energy_inputs_{timestamp}.xlsx")
            os.makedirs("data", exist_ok=True)
            df_out.to_excel(out_file, index=False)
            st.success(f"Veriler kaydedildi: {out_file}")
        except Exception as e:
            st.error(f"Kayıt hatası: {e}")

def main():
    show_energy_form()

if __name__ == "__main__":
    main()
