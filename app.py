import os
from datetime import datetime

import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# SAYFA AYARLARI
# ------------------------------------------------------------
st.set_page_config(
    page_title="Enerji Verimliliği",
    layout="wide",
    page_icon=None,
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# LOGO (sol üst)
# ------------------------------------------------------------
LOGO_FILE = "logo.png"
if os.path.exists(LOGO_FILE):
    st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"]::before {{
                content: "";
                display: block;
                background-image: url("{LOGO_FILE}");
                background-size: contain;
                background-repeat: no-repeat;
                background-position: center;
                height: 120px;
                margin: 16px 16px 24px 16px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        st.image(LOGO_FILE, width=160)


# ------------------------------------------------------------
# EXCEL OKUMA (0. satır başlık, duplicate fix)
# ------------------------------------------------------------
@st.cache_data
def load_sheets():
    """
    dc_saf_soru_tablosu.xlsx içindeki sheet'leri okur.

    Beklenen yapı:
    - 0. satır: Değer | Açıklama | Birim | Açıklama | Kaynak | Veri Kaynağı | Kayıt Aralığı
    - 1+ satırlar: parametre satırları

    Yapılanlar:
    - header=None ile ham okuma
    - 0. satır -> original_header
    - duplicate kolon isimleri benzersiz yapılır (Açıklama, Açıklama -> Açıklama, Açıklama_1)
    - 1. satırdan itibaren veri alınır
    """
    file_name = "dc_saf_soru_tablosu.xlsx"

    try:
        raw = pd.read_excel(file_name, sheet_name=None, header=None)
    except FileNotFoundError:
        st.error(
            "HATA: 'dc_saf_soru_tablosu.xlsx' bulunamadı. Dosyayı app.py ile aynı klasöre koyun."
        )
        return None
    except Exception as e:
        st.error(f"Excel okunurken hata oluştu: {e}")
        return None

    sheets_fixed = {}

    for name, df_raw in raw.items():
        if df_raw is None or df_raw.empty:
            continue

        # 0. satır -> orijinal başlık
        original_header = df_raw.iloc[0].tolist()
        df = df_raw.iloc[1:].copy()

        # Duplicate kolonları benzersiz yap (Açıklama, Açıklama -> Açıklama, Açıklama_1)
        seen = {}
        dedup_header = []
        for h in original_header:
            if h in seen:
                dedup_header.append(f"{h}_{seen[h]}")
                seen[h] += 1
            else:
                dedup_header.append(h)
                seen[h] = 1

        df.columns = dedup_header

        # Boş satı
