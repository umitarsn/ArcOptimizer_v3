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
# LOGO / MARKALAMA (Sol üst taraf)
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
else:
    st.sidebar.header("Enerji Verimliliği")


# ------------------------------------------------------------
# EXCEL YÜKLEME — 0. SATIR BAŞLIK OLACAK
# ------------------------------------------------------------
@st.cache_data
def load_sheets():
    """
    dc_saf_soru_tablosu.xlsx içindeki TÜM sheet'leri yükler.

    Beklenen yapı:
    - Satır 0: Değer | Açıklama | Birim | Açıklama | Kaynak | Veri Kaynağı | Kayıt Aralığı
    - Satır 1+: Parametre satırları (Cevher FeO/Fe2O3 içeriği, Nem, vb.)

    Yapılan işlem:
    - header=None ile tüm satırları ham okuyoruz
    - 0. satırı kolon başlığı olarak atıyoruz
    - Sonrasında 0. satırı drop edip sadece veri satırlarını bırakıyoruz
    """
    file_name = "dc_saf_soru_tablosu.xlsx"

    try:
        raw_sheets = pd.read_excel(
            file_name,
            sheet_name=None,
            header=None  # ✔ başlıkları kendimiz set edeceğiz
        )
    except FileNotFoundError:
        st.error(f"❌ '{file_name}' bulunamadı. Lütfen app.py ile aynı klasöre ekleyin.")
        return None
    except Exception as e:
        st.error(f"❌ Excel okunurken hata oluştu: {e}")
        return None

    cleaned = {}
    for name, df_raw in raw_sheets.items():
        if df_raw is None or df_raw.empty:
            continue

        # İlk satırı başlık olarak al
        header = df_raw.iloc[0].tolist()
        df = df_raw.iloc[1:].copy()
        df.columns = header

        # Tamamen boş satır/kolonları temizle
        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")

        if not df.empty:
            cleaned[name] = df

    return cleaned


# ------------------------------------------------------------
# FORM SAYFASI
# ------------------------------------------------------------
def show_energy_efficiency_form():
    st.title("⚡ Enerji Verimliliği")
    st.markdown(
        """
        Bu form, **dc_saf_soru_tablosu.xlsx** dosyanızın **birebir edit edilebilir** halidir.  

        Her sheet, aşağıda açılır bir bölüm (expander) içinde:
        - Kolon başlıkları: **Değer, Açıklama, Birim, Açıklama, Kaynak, Veri Kaynağı, Kayıt Aralığı**
        - Satırlar: Excel'deki tüm parametreler

        Hücrelere tıklayıp değerleri doğrudan düzenleyebilirsiniz.
        """
    )

    sheets = load_sheets()
    if sheets is None or len(sheets) == 0:
        return

    total_rows = sum(len(df) for df in sheets.values())
    with st.sidebar:
        st.subheader("Form Bilgisi")
        st.info(
            f"Toplam satır (parametre + açıklama): {total_rows}\n\n"
            "Her sheet Excel’dekiyle aynı yapıda gösterilir ve düzenlenebilir."
        )

    edited_sheets = {}

    # --------------------------------------------------------
    # FORM
    # --------------------------------------------------------
    with st.form("energy_eff_form"):
        st.subheader("Müşteri Girdileri")
        st.write("Her başlığa tıklayıp ilgili tabloyu açın ve hücreleri düzenleyin:")

        for i, (sheet_name, df) in enumerate(sheets.items(), start=1):
            with st.expander(f"{i}. {sheet_name}", expanded=(i == 1)):
                st.caption("Bu tablo, ilgili sheet’in Excel’deki haliyle birebir aynıdır.")
                edited_df = st.data_editor(
                    df,
                    use_container_width=True,
                    num_rows="dynamic",
                    key=f"sheet_{i}",
                )
                edited_sheets[sheet_name] = edited_df

        submitted = st.form_submit_button("Kaydet")

    # --------------------------------------------------------
    # KAYDET
    # --------------------------------------------------------
    if submitted:
        if not edited_sheets:
            st.warning("Kaydedilecek veri bulunamadı.")
            return

        os.makedirs("data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outfile = os.path.join("data", f"energy_efficiency_responses_{timestamp}.xlsx")

        try:
            with pd.ExcelWriter(outfile, engine="openpyxl") as writer:
                for name, df in edited_sheets.items():
                    safe_name = str(name)[:31]  # Excel sheet adı limiti
                    df.to_excel(writer, sheet_name=safe_name, index=False)
        except Exception as e:
            st.error(f"❌ Excel dosyası yazılırken hata oluştu: {e}")
            return

        st.success("✅ Tüm sheet'ler başarıyla kaydedildi.")
        st.write(f"Kaydedilen dosya: `data/{os.path.basename(outfile)}`")
        st.info("Bu dosyayı Railway/sunucu tarafında indirip kullanabilirsiniz.")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    show_energy_efficiency_form()


if __name__ == "__main__":
    main()
