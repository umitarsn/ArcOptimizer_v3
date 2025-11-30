import os
import json
import pandas as pd
import streamlit as st

# ----------------------------------------------
# GENEL AYARLAR
# ----------------------------------------------
st.set_page_config(
    page_title="1. Veri Girişi",
    layout="wide",
)

SAVE_PATH = "data/saved_inputs.json"
os.makedirs("data", exist_ok=True)

if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, "r") as f:
        saved_inputs = json.load(f)
else:
    saved_inputs = {}

if "info_state" not in st.session_state:
    st.session_state.info_state = {}

# ----------------------------------------------
# EXCEL OKUMA
# ----------------------------------------------
@st.cache_data
def load_sheets():
    file_name = "dc_saf_soru_tablosu.xlsx"
    try:
        xls = pd.read_excel(file_name, sheet_name=None)
        return {k: v.dropna(how="all") for k, v in xls.items() if not v.empty}
    except Exception as e:
        st.error(f"Excel dosyası yüklenemedi: {e}")
        return {}

# ----------------------------------------------
# FORM GÖSTERİMİ
# ----------------------------------------------
def show_energy_form():
    st.markdown("## 🧐 1. Veri Girişi")
    st.markdown("""Bu form **dc_saf_soru_tablosu.xlsx** dosyasına göre hazırlanmıştır.

1. Girişi sadece **Set Değeri** alanına yapınız.  
2. 🔴 Zorunlu (Önem: 1), 🟡 Faydalı (Önem: 2), \26aa Opsiyonel (Önem: 3) olarak belirtilmiştir.  
3. Detaylı bilgi ve açıklama için ℹ️ simgesine tıklayınız.
    """)

    sheets = load_sheets()
    if not sheets:
        return

    total_fields = 0
    total_filled = 0
    required_fields = 0
    required_filled = 0

    for sheet_idx, (sheet_name, df) in enumerate(sheets.items(), start=1):
        with st.expander(f"{sheet_idx}. {sheet_name}", expanded=(sheet_idx == 1)):
            df = df.replace({pd.NA: None})

            for idx, row in df.iterrows():
                row_key = f"{sheet_idx}_{idx}"
                önem = int(row.get("Önem") or 3)
                renk = {1: "🔴", 2: "🟡", 3: "⚪"}.get(önem, "⚪")
                birim = str(row.get("Set") or "").strip()

                tag = row.get("Tag") or ""
                val_key = f"{sheet_name}|{tag}"

                cols = st.columns([2.2, 2.5, 4.0, 2.5, 0.7])
                cols[0].markdown(f"**{tag}**")
                cols[1].markdown(f"{renk} {row.get('Değişken', '')}")
                cols[2].markdown(row.get("Açıklama", ""))

                current_val = saved_inputs.get(val_key, "")

                with cols[3]:
                    input_col, unit_col = st.columns([5, 2])
                    with input_col:
                        new_val = st.text_input(
                            label="",
                            value=current_val,
                            key=val_key,
                            label_visibility="collapsed",
                            placeholder=""
                        )
                        if new_val != current_val:
                            saved_inputs[val_key] = new_val
                            with open(SAVE_PATH, "w") as f:
                                json.dump(saved_inputs, f)

                    with unit_col:
                        if birim.lower() not in ["", "none", "nan"]:
                            st.markdown(f"**{birim}**")

                with cols[4]:
                    if st.button("ℹ️", key=f"info_{row_key}"):
                        st.session_state.info_state[row_key] = not st.session_state.info_state.get(row_key, False)

                if st.session_state.info_state.get(row_key, False):
                    detaylar = []
                    if row.get("Detaylı Açıklama"):
                        detaylar.append(f"🔷 **Detaylı Açıklama:** {row['Detaylı Açıklama']}")
                    if row.get("Veri Kaynağı"):
                        detaylar.append(f"📌 **Kaynak:** {row['Veri Kaynağı']}")
                    if row.get("Kayıt Aralığı"):
                        detaylar.append(f"⏱️ **Kayıt Aralığı:** {row['Kayıt Aralığı']}")
                    if row.get("Önem") is not None:
                        detaylar.append(f"🔵 **Önem:** {int(row['Önem'])}")
                    st.info("  \n".join(detaylar))

                total_fields += 1
                if new_val.strip():
                    total_filled += 1
                    if önem == 1:
                        required_filled += 1
                if önem == 1:
                    required_fields += 1

    st.sidebar.subheader("📊 Veri Girişi Durumu")
    pct_all = round(100 * total_filled / total_fields, 1) if total_fields else 0
    pct_required = round(100 * required_filled / required_fields,
