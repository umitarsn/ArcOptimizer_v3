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
# EXCEL YÜKLEYEN FONKSİYON
# ------------------------------------------------------------
@st.cache_data
def load_sheets():
    """
    dc_saf_soru_tablosu.xlsx dosyasını sheet'leriyle birlikte okur.
    - Senin dosyanın gerçek yapısına göre:
      * Satır 1: sheet başlığı
      * Satır 2: boş/kaymış hücreler
      * Satır 3: gerçek kolon başlıkları → header=2
    """
    file_name = "dc_saf_soru_tablosu.xlsx"

    try:
        sheets = pd.read_excel(
            file_name,
            sheet_name=None,
            header=2   # ✔ senin dosyana %100 uygun
        )
    except FileNotFoundError:
        st.error("❌ Excel dosyası bulunamadı. app.py ile aynı klasöre koymalısın.")
        return None
    except Exception as e:
        st.error(f"❌ Excel okunurken hata oluştu: {e}")
        return None

    # Boş satır/kolon temizliği
    cleaned = {}
    for name, df in sheets.items():
        if df is None:
            continue
        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")
        if not df.empty:
            cleaned[name] = df

    return cleaned


# ------------------------------------------------------------
# FORM EKRANI
# ------------------------------------------------------------
def show_energy_efficiency_form():
    st.title("⚡ Enerji Verimliliği")
    st.write("Bu form Excel dosyanızın birebir düzenlenebilir halidir.")

    sheets = load_sheets()
    if sheets is None:
        return

    edited_sheets = {}

    with st.form("energy_form"):
        for i, (sheet_name, df) in enumerate(sheets.items(), start=1):
            with st.expander(f"{i}. {sheet_name}", expanded=(i == 1)):
                st.caption("Excel sheet'inin birebir düzenlenebilir hali")
                edited_df = st.data_editor(
                    df,
                    use_container_width=True,
                    num_rows="dynamic",
                    key=f"sheet_{i}",
                )
                edited_sheets[sheet_name] = edited_df

        submitted = st.form_submit_button("Kaydet")

    # --------------------------------------------------------
    # KAYDETME
    # --------------------------------------------------------
    if submitted:
        os.makedirs("data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outfile = os.path.join("data", f"energy_efficiency_{timestamp}.xlsx")

        try:
            with pd.ExcelWriter(outfile, engine="openpyxl") as writer:
                for name, df in edited_sheets.items():
                    df.to_excel(writer, sheet_name=name[:31], index=False)
        except Exception as e:
            st.error(f"❌ Kaydedilirken hata oluştu: {e}")
            return

        st.success("✅ Tüm sheet'ler kaydedildi.")
        st.write(f"Kaydedilen dosya: `data/{os.path.basename(outfile)}`")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    show_energy_efficiency_form()


if __name__ == "__main__":
    main()
