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
# LOGO
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
    dc_saf_soru_tablosu.xlsx içindeki TÜM sheet'leri okur.

    Beklenen yapı:
    0. satır: Değer | Açıklama | Birim | Açıklama | Kaynak | Veri Kaynağı | Kayıt Aralığı
    1+ satırlar: parametre satırları

    - header=None ile ham okuruz
    - 0. satırı header yaparız
    - Duplicate kolon isimlerini benzersizleştiririz (Açıklama, Açıklama → Açıklama, Açıklama_1)
    """
    file_name = "dc_saf_soru_tablosu.xlsx"

    try:
        raw = pd.read_excel(file_name, sheet_name=None, header=None)
    except FileNotFoundError:
        st.error("❌ 'dc_saf_soru_tablosu.xlsx' bulunamadı. app.py ile aynı klasör_
